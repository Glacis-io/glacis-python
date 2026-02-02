"""
Tests for the Glacis client.
"""

import pytest
from pytest_httpx import HTTPXMock

from glacis import AsyncGlacis, Glacis
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
                "attestation_hash": "abc123def456",
                "timestamp": "2024-01-01T00:00:00Z",
                "leafIndex": 42,
                "treeSize": 100,
                "leafHash": "abc123",
                "merkleProof": {"leafIndex": 42, "treeSize": 100, "hashes": ["def456"]},
                "signedTreeHead": {
                    "treeSize": 100,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "rootHash": "root123",
                    "signature": "sig123",
                },
                "badgeUrl": "https://api.glacis.io/badge/att_test123.svg",
                "verifyUrl": "https://api.glacis.io/v1/verify/att_test123",
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
        assert body["metadata"] == {"model": "gpt-4"}

        # Payload should NOT be in the request
        assert "input" not in body
        assert "output" not in body
        assert "prompt" not in body

        # Verify response parsing
        assert receipt.attestation_id == "att_test123"
        assert receipt.leaf_index == 42

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
                "attestation_hash": "hash123",
                "timestamp": "2024-01-01T00:00:00Z",
                "leafIndex": 1,
                "treeSize": 1,
                "leafHash": "hash",
                "merkleProof": {"leafIndex": 1, "treeSize": 1, "hashes": []},
                "signedTreeHead": {
                    "treeSize": 1,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "rootHash": "root",
                    "signature": "sig",
                },
                "badgeUrl": "",
                "verifyUrl": "",
            },
        )

        async with AsyncGlacis(api_key="test") as glacis:
            receipt = await glacis.attest(
                service_id="test",
                operation_type="inference",
                input={"data": "test"},
                output={"result": "ok"},
            )

        assert receipt.attestation_id == "att_async"

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
