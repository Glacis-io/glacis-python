"""Integration tests for hosted (server-attested) minting, gateway mocked.

The mock reproduces the real gateway's behavior (lite demo backend
``gateway/app/governance_api.py``): ``POST /v1/govern`` takes exactly
``{task_class, request_sha256}``, echoes the commitment into
``receipt.commitments.request``, and — with ``sync_anchor=true`` — returns an
included transparency record, or a pending record that anchors later via
``GET /transparency/proof``.
"""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from glacis import Glacis
from glacis.crypto import offline_signed_payload_for
from glacis.models import GlacisApiError, GlacisMintError, HostedArtifact
from glacis.witness import is_projected_witness_receipt
from tests._reference_log import ReferenceLog

API_KEY = "glsk_live_test_key"
BASE = "https://api.glacis.io"


def _make_client(**kw) -> Glacis:
    kw.setdefault("api_key", API_KEY)
    kw.setdefault("mode", "hosted")
    kw.setdefault("signing_seed", b"\x07" * 32)
    return Glacis(**kw)


def _receipt_for(request_sha256: str, receipt_hash: str, task: str = "default") -> dict:
    """A projected receipt exactly as gateway/app/projection.py constructs it."""
    return {
        "receipt_id": receipt_hash,
        "prev": "0" * 64,
        "task": task,
        "outcome": "ADMITTED",
        "governed": True,
        "charter_version": "1.0.0",
        "charter_hash": "b" * 64,
        "commitments": {"request": request_sha256, "response": None},
        "latency_ms": 2,
        "at_ms": 1_800_000_000_000,
    }


@pytest.fixture
def log() -> ReferenceLog:
    log = ReferenceLog()
    for i in range(5):
        log.append(hashlib.sha256(b"prior-%d" % i).hexdigest())
    return log


def _mock_govern_included(httpx_mock, log: ReferenceLog):
    """Mock a sync-anchored mint that echoes the SDK's real commitment."""

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert set(body) == {"task_class", "request_sha256"}  # deny-unknown
        assert request.headers["X-Glacis-Key"] == API_KEY
        receipt_hash = hashlib.sha256(request.content).hexdigest()
        idx = log.append(receipt_hash)
        return httpx.Response(
            200,
            json={
                "receipt": _receipt_for(body["request_sha256"], receipt_hash,
                                        body["task_class"]),
                "inclusion": log.inclusion(idx),
            },
        )

    httpx_mock.add_callback(respond, url=f"{BASE}/v1/govern?sync_anchor=true")


class TestHostedMint:
    def test_witnessed_end_to_end(self, httpx_mock, log: ReferenceLog):
        _mock_govern_included(httpx_mock, log)
        client = _make_client(log_public_keys=[log.public_key_hex])
        artifact = client.attest(
            service_id="svc",
            operation_type="inference",
            input={"prompt": "hi"},
            output={"response": "yo"},
        )
        assert isinstance(artifact, HostedArtifact)
        assert artifact.witness_status == "WITNESSED"
        assert artifact.verification.inclusion_verified
        assert artifact.verification.sth_signature_verified
        assert artifact.verification.log_public_key_hex == log.public_key_hex

    def test_binding_preimage_is_the_signed_bytes(self, httpx_mock, log: ReferenceLog):
        """request_sha256 == sha256 of the attestation's exact signed bytes,
        re-derivable by any holder of the artifact — and the gateway's echo
        (receipt.commitments.request) matches it."""
        _mock_govern_included(httpx_mock, log)
        client = _make_client(log_public_keys=[log.public_key_hex])
        artifact = client.attest(
            service_id="svc", operation_type="inference",
            input={"a": 1}, output={"b": 2},
        )
        rederived = hashlib.sha256(
            offline_signed_payload_for(artifact.attestation)
        ).hexdigest()
        assert artifact.binding.request_sha256 == rederived
        assert artifact.receipt["commitments"]["request"] == rederived

    def test_artifact_json_parses_as_permalink_envelope(self, httpx_mock, log):
        """The saved artifact must open at glacis.io/verify: its top level
        carries a receipt object (unwrapEnvelope keys on that and ignores
        unknown fields) and the inner receipt is witness-projected."""
        _mock_govern_included(httpx_mock, log)
        client = _make_client(log_public_keys=[log.public_key_hex])
        artifact = client.attest(
            service_id="svc", operation_type="inference", input=1, output=2,
        )
        doc = json.loads(artifact.to_json())
        assert doc["v"] == 1
        assert isinstance(doc["receipt"], dict)
        assert isinstance(doc["inclusion"], dict)
        assert is_projected_witness_receipt(doc["receipt"])
        assert doc["attestation_mode"] == "server-attested"
        # The local attestation rides along, itself offline-verifiable.
        assert doc["attestation"]["id"].startswith("oatt_")
        assert doc["attestation"]["signature"]

    def test_no_log_key_mints_but_never_witnessed(self, httpx_mock, log):
        _mock_govern_included(httpx_mock, log)
        client = _make_client(log_public_keys=[])
        artifact = client.attest(
            service_id="svc", operation_type="inference", input=1, output=2,
        )
        assert artifact.witness_status == "LOGGED_UNVERIFIED"
        assert "GLACIS_LOG_PUBLIC_KEY_HEX" in artifact.verification.reason

    def test_commitment_mismatch_fails_closed(self, httpx_mock, log):
        def respond(request: httpx.Request) -> httpx.Response:
            receipt_hash = hashlib.sha256(b"other").hexdigest()
            idx = log.append(receipt_hash)
            return httpx.Response(200, json={
                "receipt": _receipt_for("e" * 64, receipt_hash),
                "inclusion": log.inclusion(idx),
            })

        httpx_mock.add_callback(respond, url=f"{BASE}/v1/govern?sync_anchor=true")
        client = _make_client(log_public_keys=[log.public_key_hex])
        with pytest.raises(GlacisMintError, match="different request"):
            client.attest(service_id="s", operation_type="t", input=1, output=2)

    def test_401_is_surfaced(self, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE}/v1/govern?sync_anchor=true",
            status_code=401,
            json={"error": "invalid key"},
        )
        client = _make_client()
        with pytest.raises(GlacisApiError) as exc:
            client.attest(service_id="s", operation_type="t", input=1, output=2)
        assert exc.value.status == 401
        assert "GLACIS_API_KEY" in str(exc.value)

    def test_401_key_revoked_is_distinct(self, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE}/v1/govern?sync_anchor=true",
            status_code=401,
            json={"error": "key_revoked"},
        )
        client = _make_client()
        with pytest.raises(GlacisApiError, match="revoked"):
            client.attest(service_id="s", operation_type="t", input=1, output=2)
        assert len(httpx_mock.get_requests()) == 1

    def test_429_abuse_limit_surfaced_not_retried(self, httpx_mock):
        from glacis.models import GlacisRateLimitError

        httpx_mock.add_response(
            url=f"{BASE}/v1/govern?sync_anchor=true",
            status_code=429,
            json={"error": "abuse rate limit"},
            headers={"Retry-After": "30"},
        )
        client = _make_client()
        with pytest.raises(GlacisRateLimitError) as exc:
            client.attest(service_id="s", operation_type="t", input=1, output=2)
        assert exc.value.retry_after_ms == 30_000
        assert len(httpx_mock.get_requests()) == 1

    def test_429_without_retry_after_header(self, httpx_mock):
        """The gateway sets no Retry-After on its abuse 429; the error must
        carry retry_after_ms=None gracefully, not crash or invent a window."""
        from glacis.models import GlacisRateLimitError

        httpx_mock.add_response(
            url=f"{BASE}/v1/govern?sync_anchor=true",
            status_code=429,
            json={"error": "rate_limited",
                  "detail": "abuse control on a free witness — not a quota; retry shortly"},
        )
        client = _make_client()
        with pytest.raises(GlacisRateLimitError) as exc:
            client.attest(service_id="s", operation_type="t", input=1, output=2)
        assert exc.value.retry_after_ms is None
        assert len(httpx_mock.get_requests()) == 1

    def test_503_key_validation_unavailable_is_a_refusal(self, httpx_mock):
        """A gateway-side refusal, not a network error: no receipt was minted,
        nothing is retried, and the message says which it was."""
        httpx_mock.add_response(
            url=f"{BASE}/v1/govern?sync_anchor=true",
            status_code=503,
            json={"error": "key_validation_unavailable"},
        )
        client = _make_client()
        with pytest.raises(GlacisApiError, match="gateway-side refusal") as exc:
            client.attest(service_id="s", operation_type="t", input=1, output=2)
        assert exc.value.status == 503
        assert len(httpx_mock.get_requests()) == 1

    def test_glsk_test_key_accepted_labels_unaffected(self, httpx_mock, log):
        """glsk_test_ keys authenticate like glsk_live_ ones; the key is
        opaque to the SDK and never influences the label state machine."""

        def respond(request: httpx.Request) -> httpx.Response:
            assert request.headers["X-Glacis-Key"] == "glsk_test_some_key"
            body = json.loads(request.content)
            receipt_hash = hashlib.sha256(request.content).hexdigest()
            idx = log.append(receipt_hash)
            return httpx.Response(200, json={
                "receipt": _receipt_for(body["request_sha256"], receipt_hash),
                "inclusion": log.inclusion(idx),
            })

        httpx_mock.add_callback(respond, url=f"{BASE}/v1/govern?sync_anchor=true")
        client = _make_client(
            api_key="glsk_test_some_key", log_public_keys=[log.public_key_hex]
        )
        artifact = client.attest(
            service_id="s", operation_type="t", input=1, output=2,
        )
        assert artifact.witness_status == "WITNESSED"

    def test_timeout_after_send_is_not_retried(self, httpx_mock):
        """/v1/govern is not idempotent: a timed-out mint may have minted.
        Exactly one request goes out; the error says why."""
        httpx_mock.add_exception(httpx.ReadTimeout("deadline"))
        client = _make_client()
        with pytest.raises(GlacisApiError, match="not idempotent"):
            client.attest(service_id="s", operation_type="t", input=1, output=2)
        assert len(httpx_mock.get_requests()) == 1

    def test_pending_then_anchored_via_poll(self, httpx_mock, log):
        state = {}

        def govern(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            receipt_hash = hashlib.sha256(request.content).hexdigest()
            state["idx"] = log.append(receipt_hash)
            state["receipt_hash"] = receipt_hash
            return httpx.Response(200, json={
                "receipt": _receipt_for(body["request_sha256"], receipt_hash),
                "inclusion": {"status": "pending", "eta_ms": 100,
                              "poll": "/transparency/proof?receipt_id=" + receipt_hash},
            })

        def proof(request: httpx.Request) -> httpx.Response:
            inc = log.inclusion(state["idx"])
            return httpx.Response(200, json=inc)

        httpx_mock.add_callback(govern, url=f"{BASE}/v1/govern?sync_anchor=true")
        # The receipt_id in the poll URL is only known after the mint, so the
        # proof endpoint is matched by method rather than full URL.
        httpx_mock.add_callback(proof, method="GET", is_reusable=True)

        client = _make_client(log_public_keys=[log.public_key_hex])
        artifact = client.attest(
            service_id="s", operation_type="t", input=1, output=2,
        )
        assert artifact.inclusion["status"] == "included"
        assert artifact.witness_status == "WITNESSED"

    def test_pending_forever_is_honest(self, httpx_mock, log):
        def govern(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            receipt_hash = hashlib.sha256(request.content).hexdigest()
            return httpx.Response(200, json={
                "receipt": _receipt_for(body["request_sha256"], receipt_hash),
                "inclusion": {"status": "pending", "eta_ms": 100},
            })

        def proof(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"status": "unknown"})

        httpx_mock.add_callback(govern, url=f"{BASE}/v1/govern?sync_anchor=true")
        httpx_mock.add_callback(proof, method="GET", is_reusable=True)

        client = _make_client(log_public_keys=[log.public_key_hex], timeout=0.6)
        artifact = client.attest(
            service_id="s", operation_type="t", input=1, output=2,
        )
        assert artifact.witness_status == "LOGGED_UNVERIFIED"
        assert artifact.inclusion["status"] == "pending"


class TestHostedConfig:
    def test_missing_api_key_is_a_clear_auth_error(self, monkeypatch):
        monkeypatch.delenv("GLACIS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GLACIS_API_KEY"):
            Glacis(mode="hosted", signing_seed=b"\x07" * 32)

    def test_env_configuration(self, monkeypatch, httpx_mock, log):
        monkeypatch.setenv("GLACIS_API_KEY", API_KEY)
        monkeypatch.setenv("GLACIS_WITNESS_API_BASE", "https://alt.glacis.io")
        monkeypatch.setenv("GLACIS_LOG_PUBLIC_KEY_HEX", log.public_key_hex)
        monkeypatch.setenv("GLACIS_SIGNING_SEED_HEX", "07" * 32)

        def respond(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            receipt_hash = hashlib.sha256(request.content).hexdigest()
            idx = log.append(receipt_hash)
            return httpx.Response(200, json={
                "receipt": _receipt_for(body["request_sha256"], receipt_hash),
                "inclusion": log.inclusion(idx),
            })

        httpx_mock.add_callback(
            respond, url="https://alt.glacis.io/v1/govern?sync_anchor=true"
        )
        client = Glacis(mode="hosted")
        assert client.base_url == "https://alt.glacis.io"
        assert client.timeout == 8.0  # portal mint-client parity
        artifact = client.attest(
            service_id="s", operation_type="t", input=1, output=2,
        )
        assert artifact.witness_status == "WITNESSED"

    def test_unknown_task_class_rejected_before_spending_a_mint(self, httpx_mock):
        client = _make_client()
        with pytest.raises(ValueError, match="public-safe"):
            client.attest(
                service_id="s", operation_type="t", input=1, output=2,
                task_class="totally-made-up",
            )
        assert len(httpx_mock.get_requests()) == 0

    def test_offline_and_online_modes_unchanged(self):
        offline = Glacis(mode="offline", signing_seed=b"\x07" * 32)
        assert offline.timeout == 30.0
        assert offline.base_url == "https://api.glacis.io"
        offline.close()
