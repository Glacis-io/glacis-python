"""
Tests for the Glacis client.
"""

import pytest
from pytest_httpx import HTTPXMock

from glacis import AsyncGlacis, Glacis
from glacis.crypto import hash_payload
from glacis.models import GlacisApiError, GlacisRateLimitError


class TestGlacisSync:
    """Tests for the synchronous Glacis client."""

    def test_requires_api_key(self):
        """API key is required."""
        with pytest.raises(ValueError, match="api_key is required"):
            Glacis(api_key="")

    def test_accepts_valid_config(self):
        """Valid config is accepted."""
        glacis = Glacis(
            api_key="glsk_live_test123",
            base_url="https://custom.api.glacis.io",
            debug=True,
        )
        assert glacis.api_key == "glsk_live_test123"
        assert glacis.base_url == "https://custom.api.glacis.io"

    def test_hash_produces_64_char_hex(self):
        """Hash produces 64-character hex string."""
        glacis = Glacis(api_key="test")
        result = glacis.hash({"test": "data"})
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_key_order_invariant(self):
        """Hash is invariant to key order."""
        glacis = Glacis(api_key="test")
        hash1 = glacis.hash({"b": 2, "a": 1})
        hash2 = glacis.hash({"a": 1, "b": 2})
        assert hash1 == hash2

    def test_attest_sends_hash_not_payload(self, httpx_mock: HTTPXMock):
        """Attest sends payload hash, not the actual payload."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.glacis.io/v1/attest",
            json={
                "attestationId": "att_test123",
                "evidenceHash": "a" * 64,
                "timestamp": 1704067200000,
                "leafIndex": 42,
                "treeSize": 100,
                "samplingDecision": {"level": "L0"},
            },
        )

        with Glacis(api_key="glsk_live_test123") as glacis:
            receipt = glacis.attest(
                service_id="my-service",
                operation_type="inference",
                input={"prompt": "Hello, world!"},
                output={"response": "Hi there!"},
                metadata={"model": "gpt-4"},
            )

        # Verify request
        request = httpx_mock.get_request()
        assert request is not None
        assert request.headers["X-Glacis-Key"] == "glsk_live_test123"

        import json

        body = json.loads(request.content)
        assert body["serviceId"] == "my-service"
        assert body["operationType"] == "inference"
        assert len(body["payloadHash"]) == 64  # SHA-256 hex

        # Zero-egress: user data should NOT be in the request
        assert "input" not in body
        assert "output" not in body
        assert "prompt" not in body
        # User metadata (model=gpt-4) not sent; no operational metadata without phase/correlation_id
        assert "metadata" not in body

        # Verify response parsing
        assert receipt.id == "att_test123"
        assert receipt.leaf_index == 42
        assert receipt.sampling_decision is not None
        assert receipt.sampling_decision.level == "L0"

    def test_attest_sends_operational_metadata(self, httpx_mock: HTTPXMock):
        """Attest sends phase/correlationId as operational metadata."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.glacis.io/v1/attest",
            json={
                "attestationId": "att_test456",
                "evidenceHash": "b" * 64,
                "timestamp": 1704067200000,
                "leafIndex": 1,
                "treeSize": 10,
                "samplingDecision": {"level": "L1", "sampleValue": "00ff" * 4},
            },
        )

        with Glacis(api_key="glsk_live_test123") as glacis:
            receipt = glacis.attest(
                service_id="my-service",
                operation_type="inference",
                input={"prompt": "test"},
                output={"response": "test"},
                metadata={"model": "gpt-4"},  # User metadata - NOT sent
                phase="input",
                correlation_id="corr-123",
            )

        import json

        body = json.loads(httpx_mock.get_request().content)

        # Operational metadata IS sent
        assert "metadata" in body
        assert body["metadata"]["phase"] == "input"
        assert body["metadata"]["correlationId"] == "corr-123"
        assert body["metadata"]["sequenceIndex"] == 0

        # User metadata (model=gpt-4) is NOT in the operational metadata
        assert "model" not in body["metadata"]

        # User data still not sent (zero-egress)
        assert "input" not in body
        assert "output" not in body

        # Sampling decision parsed
        assert receipt.sampling_decision.level == "L1"
        assert receipt.sampling_decision.sample_value == "00ff" * 4

    def test_attest_sends_control_plane_results_in_body(self, httpx_mock: HTTPXMock):
        """When control_plane_results is provided, it's sent in the request body."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.glacis.io/v1/attest",
            json={
                "attestationId": "att_cpr_test",
                "evidenceHash": "c" * 64,
                "timestamp": 1704067200000,
                "leafIndex": 1,
                "treeSize": 10,
            },
        )

        from glacis.models import (
            ControlPlaneResults,
            Determination,
            PolicyContext,
            PolicyScope,
            SafetyScores,
        )

        cpr = ControlPlaneResults(
            policy=PolicyContext(
                id="policy-1",
                version="1.0",
                scope=PolicyScope(environment="production", tags=[]),
            ),
            determination=Determination(action="forwarded", confidence=1.0),
            controls=[],
            safety=SafetyScores(overall_risk=0.1),
        )

        with Glacis(api_key="glsk_live_test123") as glacis:
            receipt = glacis.attest(
                service_id="my-service",
                operation_type="inference",
                input={"prompt": "Hello"},
                output={"response": "Hi"},
                control_plane_results=cpr,
            )

        import json

        body = json.loads(httpx_mock.get_request().content)

        # controlPlaneResults should be in the request body
        assert "controlPlaneResults" in body
        assert body["controlPlaneResults"]["schema_version"] == "1.0"
        assert body["controlPlaneResults"]["determination"]["action"] == "forwarded"

        # cprHash should be in the request body (separate from payloadHash)
        assert "cprHash" in body
        assert len(body["cprHash"]) == 64  # valid hex hash

        # User data still not sent (zero-egress)
        assert "input" not in body
        assert "output" not in body

        # Receipt should have control_plane_results attached locally
        assert receipt.control_plane_results is not None
        assert receipt.control_plane_results.determination.action == "forwarded"

    def test_verify_public_endpoint(self, httpx_mock: HTTPXMock):
        """Verify is a public endpoint (no auth header)."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.glacis.io/v1/verify/att_test123",
            json={
                "valid": True,
                "attestation": {
                    "entryId": "att_test123",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "orgId": "org_xxx",
                    "serviceId": "my-service",
                    "operationType": "inference",
                    "payloadHash": "abc123",
                    "signature": "sig123",
                    "leafIndex": 42,
                    "leafHash": "hash123",
                },
                "verification": {
                    "signatureValid": True,
                    "proofValid": True,
                    "verifiedAt": "2024-01-01T00:00:00Z",
                },
                "proof": {"leafIndex": 42, "treeSize": 100, "hashes": []},
                "treeHead": {
                    "treeSize": 100,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "rootHash": "root",
                    "signature": "sig",
                },
            },
        )

        with Glacis(api_key="test") as glacis:
            result = glacis.verify("att_test123")

        request = httpx_mock.get_request()
        assert request is not None
        # Verify endpoint should NOT have auth header
        assert "X-Glacis-Key" not in request.headers

        assert result.valid is True
        assert result.verification.signature_valid is True

    def test_query_log_params(self, httpx_mock: HTTPXMock):
        """Query log builds params correctly."""
        httpx_mock.add_response(
            method="GET",
            json={
                "entries": [],
                "hasMore": False,
                "count": 0,
                "treeHead": {
                    "treeSize": 0,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "rootHash": "",
                    "signature": "",
                },
            },
        )

        with Glacis(api_key="test") as glacis:
            glacis.query_log(
                org_id="org_test",
                service_id="svc_test",
                start="2024-01-01T00:00:00Z",
                end="2024-12-31T23:59:59Z",
                limit=100,
                cursor="cursor123",
            )

        request = httpx_mock.get_request()
        assert request is not None
        url = str(request.url)
        assert "orgId=org_test" in url
        assert "serviceId=svc_test" in url
        assert "limit=100" in url
        assert "cursor=cursor123" in url

    def test_rate_limit_error(self, httpx_mock: HTTPXMock):
        """Rate limit returns appropriate error."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.glacis.io/v1/attest",
            status_code=429,
            headers={"Retry-After": "60"},
        )

        with Glacis(api_key="test", max_retries=0) as glacis:
            with pytest.raises(GlacisRateLimitError) as exc:
                glacis.attest(
                    service_id="test",
                    operation_type="inference",
                    input={},
                    output={},
                )
            assert exc.value.retry_after_ms == 60000

    def test_client_error_no_retry(self, httpx_mock: HTTPXMock):
        """Client errors (4xx) should not be retried."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.glacis.io/v1/attest",
            status_code=400,
            json={"error": "Bad request"},
        )

        with Glacis(api_key="test", max_retries=3) as glacis:
            with pytest.raises(GlacisApiError) as exc:
                glacis.attest(
                    service_id="test",
                    operation_type="inference",
                    input={},
                    output={},
                )
            assert exc.value.status == 400

        # Should only make one request (no retries)
        assert len(httpx_mock.get_requests()) == 1


class TestGlacisAsync:
    """Tests for the asynchronous Glacis client."""

    @pytest.mark.asyncio
    async def test_async_attest(self, httpx_mock: HTTPXMock):
        """Async attest works correctly."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.glacis.io/v1/attest",
            json={
                "attestationId": "att_async",
                "evidenceHash": "a" * 64,  # Spec: evidenceHash
                "timestamp": 1704067200000,  # Spec: Unix ms
                "leafIndex": 1,
                "treeSize": 1,
            },
        )

        async with AsyncGlacis(api_key="test") as glacis:
            receipt = await glacis.attest(
                service_id="test",
                operation_type="inference",
                input={"data": "test"},
                output={"result": "ok"},
            )

        assert receipt.id == "att_async"

    @pytest.mark.asyncio
    async def test_async_attest_with_control_plane_results(self, httpx_mock: HTTPXMock):
        """Async attest sends controlPlaneResults and includes in hash."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.glacis.io/v1/attest",
            json={
                "attestationId": "att_async_cpr",
                "evidenceHash": "d" * 64,
                "timestamp": 1704067200000,
                "leafIndex": 1,
                "treeSize": 10,
            },
        )

        from glacis.models import (
            ControlPlaneResults,
            Determination,
            PolicyContext,
            PolicyScope,
            SafetyScores,
        )

        cpr = ControlPlaneResults(
            policy=PolicyContext(
                id="policy-1",
                version="1.0",
                scope=PolicyScope(environment="production", tags=[]),
            ),
            determination=Determination(action="forwarded", confidence=1.0),
            controls=[],
            safety=SafetyScores(overall_risk=0.0),
        )

        async with AsyncGlacis(api_key="test") as glacis:
            receipt = await glacis.attest(
                service_id="test",
                operation_type="inference",
                input={"data": "test"},
                output={"result": "ok"},
                control_plane_results=cpr,
            )

        import json

        body = json.loads(httpx_mock.get_request().content)

        # controlPlaneResults should be in the request body
        assert "controlPlaneResults" in body
        assert body["controlPlaneResults"]["schema_version"] == "1.0"

        # cprHash should be in the request body (separate from payloadHash)
        assert "cprHash" in body
        assert len(body["cprHash"]) == 64

        # Receipt should have control_plane_results attached locally
        assert receipt.control_plane_results is not None
        assert receipt.control_plane_results.schema_version == "1.0"

        # payloadHash sent to server is now I/O-only (CPR has its own cpr_hash)
        hash_io_only = glacis.hash({
            "input": {"data": "test"},
            "output": {"result": "ok"},
        })
        assert body["payloadHash"] == hash_io_only

    @pytest.mark.asyncio
    async def test_async_verify(self, httpx_mock: HTTPXMock):
        """Async verify works correctly."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.glacis.io/v1/verify/att_test",
            json={
                "valid": True,
                "verification": {
                    "signatureValid": True,
                    "proofValid": True,
                    "verifiedAt": "2024-01-01T00:00:00Z",
                },
                "proof": {"leafIndex": 1, "treeSize": 1, "hashes": []},
                "treeHead": {
                    "treeSize": 1,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "rootHash": "root",
                    "signature": "sig",
                },
            },
        )

        async with AsyncGlacis(api_key="test") as glacis:
            result = await glacis.verify("att_test")

        assert result.valid is True

    def test_async_requires_api_key(self):
        """Empty API key raises ValueError."""
        with pytest.raises(ValueError, match="api_key is required"):
            AsyncGlacis(api_key="")

    @pytest.mark.asyncio
    async def test_async_rate_limit_error(self, httpx_mock: HTTPXMock):
        """429 response raises GlacisRateLimitError."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.glacis.io/v1/attest",
            status_code=429,
            headers={"Retry-After": "30"},
        )

        async with AsyncGlacis(api_key="test", max_retries=0) as glacis:
            with pytest.raises(GlacisRateLimitError) as exc:
                await glacis.attest(
                    service_id="test",
                    operation_type="inference",
                    input={},
                    output={},
                )
            assert exc.value.retry_after_ms == 30000

    @pytest.mark.asyncio
    async def test_async_client_error_no_retry(self, httpx_mock: HTTPXMock):
        """Client errors (4xx) are not retried."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.glacis.io/v1/attest",
            status_code=400,
            json={"error": "Bad request"},
        )

        async with AsyncGlacis(api_key="test", max_retries=3) as glacis:
            with pytest.raises(GlacisApiError) as exc:
                await glacis.attest(
                    service_id="test",
                    operation_type="inference",
                    input={},
                    output={},
                )
            assert exc.value.status == 400

        assert len(httpx_mock.get_requests()) == 1

    @pytest.mark.asyncio
    async def test_async_query_log(self, httpx_mock: HTTPXMock):
        """Async query_log builds params and parses result."""
        httpx_mock.add_response(
            method="GET",
            json={
                "entries": [],
                "hasMore": False,
                "count": 0,
                "treeHead": {
                    "treeSize": 0,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "rootHash": "",
                    "signature": "",
                },
            },
        )

        async with AsyncGlacis(api_key="test") as glacis:
            result = await glacis.query_log(
                org_id="org_test",
                service_id="svc_test",
                limit=50,
            )

        request = httpx_mock.get_request()
        assert request is not None
        url = str(request.url)
        assert "orgId=org_test" in url
        assert "serviceId=svc_test" in url
        assert "limit=50" in url
        assert result.count == 0

    @pytest.mark.asyncio
    async def test_async_get_tree_head(self, httpx_mock: HTTPXMock):
        """Async get_tree_head parses response."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.glacis.io/v1/root",
            json={
                "tree_size": 42,
                "timestamp": "2024-06-01T00:00:00Z",
                "root_hash": "abc123",
                "signature": "sig456",
            },
        )

        async with AsyncGlacis(api_key="test") as glacis:
            head = await glacis.get_tree_head()

        assert head.tree_size == 42
        assert head.root_hash == "abc123"

    @pytest.mark.asyncio
    async def test_async_context_manager_closes(self, httpx_mock: HTTPXMock):
        """async with calls aclose() on exit."""
        glacis = AsyncGlacis(api_key="test")
        async with glacis:
            assert glacis._client is not None

        # After context manager exit, client should be closed
        assert glacis._client.is_closed

    def test_async_hash_consistency(self):
        """AsyncGlacis.hash() matches Glacis.hash() and hash_payload()."""
        payload = {"input": {"prompt": "test"}, "output": {"response": "ok"}}

        sync_glacis = Glacis(api_key="test")
        async_glacis = AsyncGlacis(api_key="test")

        assert sync_glacis.hash(payload) == async_glacis.hash(payload)
        assert async_glacis.hash(payload) == hash_payload(payload)

        sync_glacis.close()

    @pytest.mark.asyncio
    async def test_async_server_error_retries(self, httpx_mock: HTTPXMock):
        """Server errors (5xx) are retried, succeeding on second attempt."""
        # First request: 500 error
        httpx_mock.add_response(
            method="POST",
            url="https://api.glacis.io/v1/attest",
            status_code=500,
        )
        # Second request: success
        httpx_mock.add_response(
            method="POST",
            url="https://api.glacis.io/v1/attest",
            json={
                "attestationId": "att_retry",
                "evidenceHash": "b" * 64,
                "timestamp": 1704067200000,
                "leafIndex": 5,
                "treeSize": 10,
            },
        )

        async with AsyncGlacis(
            api_key="test", max_retries=2, base_delay=0.01, max_delay=0.02
        ) as glacis:
            receipt = await glacis.attest(
                service_id="test",
                operation_type="inference",
                input={"data": "test"},
                output={"result": "ok"},
            )

        assert receipt.id == "att_retry"
        assert len(httpx_mock.get_requests()) == 2
