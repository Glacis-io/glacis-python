"""
GLACIS Client implementations (sync and async).

The Glacis client provides a simple interface for attesting AI operations
to the public transparency log. Input and output data are hashed locally
using RFC 8785 canonical JSON + SHA-256 - the actual payload never leaves
your infrastructure.

Supports three modes:
- Hosted: Mints a server-attested artifact via the api.glacis.io gateway
  (local attestation + transparency-log inclusion, verified locally)
- Online: Sends attestations to api.glacis.io for witnessing (legacy)
- Offline: Signs attestations locally using Ed25519

Example (hosted):
    >>> from glacis import Glacis
    >>> glacis = Glacis(mode="hosted")  # GLACIS_API_KEY + GLACIS_LOG_PUBLIC_KEY_HEX
    >>> artifact = glacis.attest(
    ...     service_id="my-ai-service",
    ...     operation_type="inference",
    ...     input={"prompt": "Hello"},
    ...     output={"response": "Hi there!"},
    ... )
    >>> artifact.witness_status  # "WITNESSED" only after local verification
    >>> artifact.save("receipt.json")  # paste at glacis.io/verify

Example (offline):
    >>> glacis = Glacis(mode="offline", signing_seed=my_32_byte_seed)
    >>> receipt = glacis.attest(...)
    >>> result = glacis.verify(receipt)  # witness_status="SELF_SIGNED"
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import httpx

from glacis.config import SamplingConfig
from glacis.crypto import hash_payload, offline_signed_payload
from glacis.models import (
    Attestation,
    ControlPlaneResults,
    Evidence,
    GlacisApiError,
    GlacisMintError,
    GlacisRateLimitError,
    HostedArtifact,
    LogQueryResult,
    OfflineVerifyResult,
    SamplingDecision,
    TreeHeadResponse,
    VerifyResult,
    WitnessBinding,
)
from glacis.verify import verify_attestation
from glacis.verify import verify_offline as verify_offline_receipt

if TYPE_CHECKING:
    from glacis.crypto import Ed25519Runtime
    from glacis.storage import StorageBackend


class GlacisMode(str, Enum):
    """Operating mode for the Glacis client."""

    ONLINE = "online"
    OFFLINE = "offline"
    HOSTED = "hosted"


class OperationContext:
    """Tracks operation_id and auto-increments operation_sequence.

    Usage:
        >>> op = glacis.operation()
        >>> r1 = glacis.attest(..., operation_id=op.operation_id,
        ...                    operation_sequence=op.next_sequence())
        >>> r2 = glacis.attest(..., operation_id=op.operation_id,
        ...                    operation_sequence=op.next_sequence())
    """

    def __init__(self, operation_id: Optional[str] = None):
        self.operation_id = operation_id or str(uuid.uuid4())
        self._sequence = 0

    def next_sequence(self) -> int:
        seq = self._sequence
        self._sequence += 1
        return seq


logger = logging.getLogger("glacis")

DEFAULT_BASE_URL = "https://api.glacis.io"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0

# Hosted mint: one 8s deadline covers the POST and any anchor polling —
# matches the portal's mint client so the two surfaces time out together.
DEFAULT_HOSTED_DEADLINE = 8.0

ENV_API_KEY = "GLACIS_API_KEY"
ENV_WITNESS_API_BASE = "GLACIS_WITNESS_API_BASE"
ENV_LOG_PUBLIC_KEY_HEX = "GLACIS_LOG_PUBLIC_KEY_HEX"
ENV_SIGNING_SEED_HEX = "GLACIS_SIGNING_SEED_HEX"


def _normalize_server_response(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a server attest response (possibly camelCase) to Attestation fields.

    The server may still return the old flat format with camelCase keys.
    This maps it to the v1.2 Attestation model fields.
    """
    return {
        "id": data.get("attestationId", data.get("id", "")),
        "operation_id": data.get("operationId", data.get("operation_id", "")),
        "operation_sequence": data.get("operationSequence", data.get("operation_sequence", 0)),
        "service_id": data.get("serviceId", data.get("service_id", "")),
        "operation_type": data.get("operationType", data.get("operation_type", "")),
        "evidence_hash": data.get("evidenceHash", data.get("evidence_hash",
                         data.get("payloadHash", data.get("payload_hash", "")))),
        "cpr_hash": data.get("cprHash", data.get("cpr_hash")),
        "supersedes": data.get("supersedes"),
        "control_plane_results": data.get("controlPlaneResults", data.get("control_plane_results")),
        "public_key": data.get("publicKey", data.get("public_key", "")),
        "signature": data.get("signature", ""),
        "timestamp": data.get("timestamp"),
        "sampling_decision": _normalize_sampling(data),
    }


def _normalize_sampling(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Extract sampling decision from server response."""
    sd = data.get("samplingDecision", data.get("sampling_decision"))
    if sd is None:
        return None
    return {
        "level": sd.get("level", "L0"),
        "sample_value": sd.get("sampleValue", sd.get("sample_value", 0)),
        "prf_tag": sd.get("prfTag", sd.get("prf_tag", [])),
    }


class Glacis:
    """
    Synchronous GLACIS client.

    Provides attestation, verification, and log querying for the public
    transparency log. Supports both online (server-witnessed) and offline
    (locally-signed) modes.

    Args:
        api_key: API key for authenticated endpoints (required for online mode)
        base_url: Base URL for the API (default: https://api.glacis.io)
        debug: Enable debug logging
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries for transient errors
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay in seconds
        mode: Operating mode - "online" (default) or "offline"
        signing_seed: 32-byte Ed25519 signing seed (required for offline mode)
        db_path: Path to SQLite database for offline receipts (default: ~/.glacis/glacis.db)
        storage_backend: Storage backend type - "sqlite" (default) or "json"
        storage_path: Base path for storage. For sqlite: .db file path.
                      For json: directory containing .jsonl files. Overrides db_path.
        sampling_config: Sampling configuration (l1_rate, l2_rate). If None, defaults to
                         l1_rate=1.0 (review all), l2_rate=0.0 (no deep inspection).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        debug: bool = False,
        timeout: Optional[float] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        mode: Literal["online", "offline", "hosted"] = "online",
        signing_seed: Optional[bytes] = None,
        policy_key: Optional[bytes] = None,
        db_path: Optional[Path] = None,
        storage_backend: str = "sqlite",
        storage_path: Optional[Path] = None,
        sampling_config: Optional[SamplingConfig] = None,
        log_public_keys: Optional[list[str]] = None,
    ):
        self._sampling_config = sampling_config or SamplingConfig()
        self.mode = GlacisMode(mode)
        if base_url is None:
            if self.mode == GlacisMode.HOSTED:
                base_url = os.environ.get(ENV_WITNESS_API_BASE) or DEFAULT_BASE_URL
            else:
                base_url = DEFAULT_BASE_URL
        self.base_url = base_url.rstrip("/")
        self.debug = debug
        if timeout is None:
            timeout = (
                DEFAULT_HOSTED_DEADLINE
                if self.mode == GlacisMode.HOSTED
                else DEFAULT_TIMEOUT
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        if policy_key is not None and len(policy_key) != 32:
            raise ValueError("policy_key must be exactly 32 bytes")

        if self.mode == GlacisMode.HOSTED:
            self._init_hosted(api_key, signing_seed, policy_key, log_public_keys)
        elif self.mode == GlacisMode.ONLINE:
            if not api_key:
                raise ValueError("api_key is required for online mode")
            self.api_key = api_key
            self._client: Optional[httpx.Client] = httpx.Client(timeout=timeout)
            self._storage: Optional["StorageBackend"] = None
            self._signing_seed: Optional[bytes] = None
            self._policy_key: Optional[bytes] = policy_key
            self._public_key: Optional[str] = None
            self._ed25519: Optional["Ed25519Runtime"] = None
        else:
            # Offline mode
            if not signing_seed:
                raise ValueError("signing_seed is required for offline mode")
            if len(signing_seed) != 32:
                raise ValueError("signing_seed must be exactly 32 bytes")

            self.api_key = ""  # Not used in offline mode
            self._signing_seed = signing_seed
            self._policy_key = policy_key
            self._client = None  # No HTTP client needed

            # Initialize Ed25519 runtime and derive public key
            from glacis.crypto import get_ed25519_runtime

            self._ed25519 = get_ed25519_runtime()
            self._public_key = self._ed25519.get_public_key_hex(signing_seed)

            # Initialize storage (storage_path overrides db_path)
            from glacis.storage import create_storage

            effective_path = storage_path or db_path
            self._storage = create_storage(
                backend=storage_backend,
                path=effective_path,
            )

        if debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

    def _init_hosted(
        self,
        api_key: Optional[str],
        signing_seed: Optional[bytes],
        policy_key: Optional[bytes],
        log_public_keys: Optional[list[str]],
    ) -> None:
        """Hosted (server-attested) mode: local attestation + gateway mint."""
        self._ephemeral_signing_key = False
        api_key = api_key or os.environ.get(ENV_API_KEY)
        if not api_key:
            raise ValueError(
                "api_key is required for hosted mode — pass api_key= or set "
                f"{ENV_API_KEY} (a glsk_live_... key)"
            )
        self.api_key = api_key
        self._client = httpx.Client(timeout=self.timeout)
        self._storage = None
        self._policy_key = policy_key

        if signing_seed is None:
            seed_hex = os.environ.get(ENV_SIGNING_SEED_HEX, "").strip()
            if seed_hex:
                try:
                    signing_seed = bytes.fromhex(seed_hex)
                except ValueError:
                    raise ValueError(f"{ENV_SIGNING_SEED_HEX} is not valid hex")
            else:
                # Ephemeral per-client key: the local signature still binds the
                # attested content; the server-attested part of the artifact
                # carries the trust. Pass signing_seed for a stable identity.
                signing_seed = os.urandom(32)
                self._ephemeral_signing_key = True
        if len(signing_seed) != 32:
            raise ValueError("signing_seed must be exactly 32 bytes")
        self._signing_seed = signing_seed

        from glacis.crypto import get_ed25519_runtime

        self._ed25519 = get_ed25519_runtime()
        self._public_key = self._ed25519.get_public_key_hex(signing_seed)

        if log_public_keys is None:
            key_hex = os.environ.get(ENV_LOG_PUBLIC_KEY_HEX, "").strip()
            log_public_keys = [key_hex] if key_hex else []
        self._log_public_keys = [k.strip().lower() for k in log_public_keys if k.strip()]

    def __enter__(self) -> "Glacis":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client and/or storage."""
        if self._client:
            self._client.close()
        if self._storage:
            self._storage.close()

    def operation(self, operation_id: Optional[str] = None) -> OperationContext:
        """Create an OperationContext for grouping related attestations.

        Args:
            operation_id: Optional explicit operation ID (default: auto-generated UUID)

        Returns:
            OperationContext with auto-incrementing sequence
        """
        return OperationContext(operation_id)

    def attest(
        self,
        service_id: str,
        operation_type: str,
        input: Any,
        output: Any,
        metadata: Optional[dict[str, str]] = None,
        control_plane_results: Optional[Union[ControlPlaneResults, dict[str, Any]]] = None,
        operation_id: Optional[str] = None,
        operation_sequence: Optional[int] = None,
        supersedes: Optional[str] = None,
        task_class: str = "default",
    ) -> Union[Attestation, HostedArtifact]:
        """
        Attest an AI operation.

        The input, output, and control_plane_results are hashed locally using RFC 8785
        canonical JSON + SHA-256. Only the hash is sent to the server - the actual
        data never leaves your infrastructure (zero egress).

        Args:
            service_id: Service identifier (e.g., "my-ai-service")
            operation_type: Type of operation (inference, embedding, completion, classification)
            input: Input data (hashed locally, never sent)
            output: Output data (hashed locally, never sent)
            metadata: Optional metadata (stored locally for evidence)
            control_plane_results: Optional control plane results (typed model or dict)
            operation_id: UUID linking attestations in the same operation
            operation_sequence: Ordinal sequence within the operation
            supersedes: Attestation ID this replaces (revision chains)
            task_class: Hosted mode only — governance task class sent to the
                gateway. Must be one of glacis.witness.HOSTED_TASK_CLASSES.
                Ignored in online/offline mode.

        Returns:
            Attestation (online/offline) or HostedArtifact (hosted)

        Raises:
            GlacisApiError: On API errors (online/hosted mode)
            GlacisRateLimitError: When rate limited
            GlacisMintError: Hosted mode — the gateway's answer does not bind
                to the request that was sent
        """
        # I/O-only hash (evidence_hash)
        evidence_hash = self.hash({"input": input, "output": output})

        # Serialize CPR to dict if typed model
        cpr_dict: Optional[dict[str, Any]] = None
        if control_plane_results is not None:
            if hasattr(control_plane_results, "model_dump"):
                cpr_dict = control_plane_results.model_dump()
            else:
                cpr_dict = control_plane_results

        # Separate CPR hash (independently verifiable, signed in Merkle leaf)
        cpr_hash: Optional[str] = None
        if cpr_dict:
            cpr_hash = self.hash(cpr_dict)

        if self.mode == GlacisMode.OFFLINE:
            return self._attest_offline(
                service_id, operation_type, evidence_hash,
                input, output, metadata, cpr_dict, cpr_hash,
                operation_id, operation_sequence, supersedes,
            )

        if self.mode == GlacisMode.HOSTED:
            return self._attest_hosted(
                service_id, operation_type, evidence_hash,
                cpr_dict, cpr_hash,
                operation_id, operation_sequence, supersedes,
                task_class,
            )

        return self._attest_online(
            service_id, operation_type, evidence_hash,
            input, output, cpr_hash, cpr_dict,
            operation_id, operation_sequence, supersedes,
        )

    def _attest_online(
        self,
        service_id: str,
        operation_type: str,
        evidence_hash: str,
        input_data: Any,
        output_data: Any,
        cpr_hash: Optional[str] = None,
        control_plane_results: Optional[dict[str, Any]] = None,
        operation_id: Optional[str] = None,
        operation_sequence: Optional[int] = None,
        supersedes: Optional[str] = None,
    ) -> Attestation:
        """Create a server-witnessed attestation."""
        self._debug(f"Attesting (online): service_id={service_id}, hash={evidence_hash[:16]}...")

        body: dict[str, Any] = {
            "service_id": service_id,
            "operation_type": operation_type,
            "evidence_hash": evidence_hash,
        }

        if cpr_hash:
            body["cpr_hash"] = cpr_hash
        if control_plane_results:
            body["control_plane_results"] = control_plane_results
        if operation_id:
            body["operation_id"] = operation_id
        if operation_sequence is not None:
            body["operation_sequence"] = operation_sequence
        if supersedes:
            body["supersedes"] = supersedes

        response = self._request_with_retry(
            "POST",
            f"{self.base_url}/v1/attest",
            json=body,
            headers={"X-Glacis-Key": self.api_key},
        )

        # Normalize server response (may be camelCase from older server)
        normalized = _normalize_server_response(response)
        attestation = Attestation.model_validate(normalized)

        # Attach CPR locally
        if control_plane_results:
            attestation.control_plane_results = control_plane_results

        # L1/L2 Evidence: populate with raw I/O data for local retention
        if (
            attestation.sampling_decision
            and attestation.sampling_decision.level in ("L1", "L2")
        ):
            ev_data: dict[str, Any] = {
                "input": input_data,
                "output": output_data,
            }
            _prob = (
                self._sampling_config.l2_rate
                if attestation.sampling_decision.level == "L2"
                else self._sampling_config.l1_rate
            )
            attestation.evidence = Evidence(
                sample_probability=_prob,
                data=ev_data,
            )
            self._debug(
                f"L1 evidence populated (level={attestation.sampling_decision.level})"
            )

        self._debug(f"Attestation successful: {attestation.id}")
        return attestation

    def _attest_offline(
        self,
        service_id: str,
        operation_type: str,
        evidence_hash: str,
        input: Any,
        output: Any,
        metadata: Optional[dict[str, str]],
        control_plane_results: Optional[dict[str, Any]] = None,
        cpr_hash: Optional[str] = None,
        operation_id: Optional[str] = None,
        operation_sequence: Optional[int] = None,
        supersedes: Optional[str] = None,
    ) -> Attestation:
        """Create a locally-signed attestation."""
        self._debug(f"Attesting (offline): service_id={service_id}, hash={evidence_hash[:16]}...")

        attestation_id = f"oatt_{uuid.uuid4()}"
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        op_id = operation_id or str(uuid.uuid4())
        op_seq = operation_sequence if operation_sequence is not None else 0

        # The exact bytes that get signed. `glacis.verify.verify_offline()`
        # rebuilds them with the same function, so signing and verification
        # cannot drift apart.
        message = offline_signed_payload(
            service_id=service_id,
            operation_type=operation_type,
            evidence_hash=evidence_hash,
            timestamp_ms=str(timestamp_ms),
            operation_id=op_id,
            operation_sequence=op_seq,
            control_plane_results=control_plane_results,
            supersedes=supersedes,
        )

        assert self._ed25519 is not None
        assert self._signing_seed is not None
        assert self._public_key is not None

        signature_hex = self._ed25519.sign(self._signing_seed, message).hex()

        attestation = Attestation(
            id=attestation_id,
            operation_id=op_id,
            operation_sequence=op_seq,
            service_id=service_id,
            operation_type=operation_type,
            evidence_hash=evidence_hash,
            cpr_hash=cpr_hash,
            supersedes=supersedes,
            control_plane_results=control_plane_results,
            public_key=self._public_key,
            signature=signature_hex,
            is_offline=True,
            timestamp=timestamp_ms,
        )

        # Store in SQLite
        assert self._storage is not None
        self._storage.store_receipt(
            attestation,
            input_preview=str(input)[:100] if input else None,
            output_preview=str(output)[:100] if output else None,
            metadata=metadata,
        )

        self._debug(f"Offline attestation created: {attestation_id}")
        return attestation

    def _attest_hosted(
        self,
        service_id: str,
        operation_type: str,
        evidence_hash: str,
        control_plane_results: Optional[dict[str, Any]],
        cpr_hash: Optional[str],
        operation_id: Optional[str],
        operation_sequence: Optional[int],
        supersedes: Optional[str],
        task_class: str,
    ) -> HostedArtifact:
        """Mint a server-attested artifact via ``POST /v1/govern``.

        The local attestation is computed exactly as offline mode computes it
        (same signed-payload builder, same Ed25519 signing, ``oatt_`` id).
        The gateway never sees payload text: it receives only ``task_class``
        and ``request_sha256`` — SHA-256 over the attestation's exact signed
        bytes — and echoes that commitment back as
        ``receipt.commitments.request``.
        """
        from glacis.witness import HOSTED_TASK_CLASSES, classify_envelope

        if task_class not in HOSTED_TASK_CLASSES:
            raise ValueError(
                f"task_class {task_class!r} is not in the gateway's public-safe "
                f"label set: {sorted(HOSTED_TASK_CLASSES)}"
            )

        deadline = time.monotonic() + self.timeout

        # --- local attestation, exactly as offline mode ---
        attestation_id = f"oatt_{uuid.uuid4()}"
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        op_id = operation_id or str(uuid.uuid4())
        op_seq = operation_sequence if operation_sequence is not None else 0

        message = offline_signed_payload(
            service_id=service_id,
            operation_type=operation_type,
            evidence_hash=evidence_hash,
            timestamp_ms=str(timestamp_ms),
            operation_id=op_id,
            operation_sequence=op_seq,
            control_plane_results=control_plane_results,
            supersedes=supersedes,
        )

        assert self._ed25519 is not None
        assert self._signing_seed is not None
        assert self._public_key is not None
        signature_hex = self._ed25519.sign(self._signing_seed, message).hex()

        attestation = Attestation(
            id=attestation_id,
            operation_id=op_id,
            operation_sequence=op_seq,
            service_id=service_id,
            operation_type=operation_type,
            evidence_hash=evidence_hash,
            cpr_hash=cpr_hash,
            supersedes=supersedes,
            control_plane_results=control_plane_results,
            public_key=self._public_key,
            signature=signature_hex,
            is_offline=True,
            timestamp=timestamp_ms,
        )

        # The binding: the gateway's request commitment IS the hash of the
        # exact bytes the local attestation signed.
        request_sha256 = hashlib.sha256(message).hexdigest()

        response = self._mint_govern(task_class, request_sha256, deadline)

        receipt = response.get("receipt")
        inclusion = response.get("inclusion")
        if not isinstance(receipt, dict) or not isinstance(inclusion, dict):
            raise GlacisMintError(
                "the gateway's /v1/govern response has no receipt/inclusion "
                f"objects: {response!r}"
            )

        echoed = (receipt.get("commitments") or {}).get("request")
        if echoed != request_sha256:
            raise GlacisMintError(
                "the gateway's receipt commits to a different request than "
                f"this SDK sent (sent {request_sha256}, receipt carries "
                f"{echoed!r}) — the artifact would not bind, so none is issued"
            )

        # G-INGEST-LAG: a just-minted receipt may be pending; poll for the
        # anchor within what is left of the deadline.
        if inclusion.get("status") != "included":
            polled = self._poll_inclusion(receipt.get("receipt_id"), deadline)
            if polled is not None:
                inclusion = polled

        envelope = {"v": 1, "receipt": receipt, "inclusion": inclusion}
        verification = classify_envelope(envelope, self._log_public_keys)
        self._debug(
            f"Hosted mint {receipt.get('receipt_id', '?')[:16]}...: "
            f"{verification.witness_status}"
        )

        return HostedArtifact(
            receipt=receipt,
            inclusion=inclusion,
            attestation=attestation,
            binding=WitnessBinding(request_sha256=request_sha256),
            verification=verification,
        )

    def _mint_govern(
        self, task_class: str, request_sha256: str, deadline: float
    ) -> dict[str, Any]:
        """One mint POST. Never blindly retried: each /v1/govern call mints a
        NEW receipt (the translog dedupes by receipt hash, but a retry does
        not share one), so only a connect error — where the request was never
        sent — is retried, once."""
        assert self._client is not None
        url = f"{self.base_url}/v1/govern"
        body = {"task_class": task_class, "request_sha256": request_sha256}
        headers = {"X-Glacis-Key": self.api_key}
        params = {"sync_anchor": "true"}

        for attempt in (0, 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GlacisApiError("hosted mint deadline exceeded", 408)
            try:
                response = self._client.post(
                    url,
                    json=body,
                    params=params,
                    headers=headers,
                    timeout=remaining,
                )
            except httpx.ConnectError as e:
                if attempt == 0:
                    continue
                raise GlacisApiError(f"could not reach the mint gateway: {e}", 0)
            except httpx.TimeoutException as e:
                # The request may have been processed; a retry would mint a
                # second receipt. Surface it instead.
                raise GlacisApiError(
                    f"mint request timed out after being sent — not retried "
                    f"because /v1/govern is not idempotent: {e}",
                    408,
                )

            if response.is_success:
                result: dict[str, Any] = response.json()
                return result

            try:
                err_body = response.json()
            except Exception:
                err_body = {}
            message = err_body.get("error", f"mint failed with status {response.status_code}")
            if response.status_code == 401:
                message = (
                    "the mint gateway rejected this API key (401). Check "
                    f"{ENV_API_KEY} / api_key= (expects glsk_live_...): {message}"
                )
            raise GlacisApiError(message, response.status_code, err_body.get("code"), err_body)

        raise GlacisApiError("mint failed", 0)  # unreachable

    def _poll_inclusion(
        self, receipt_id: Optional[str], deadline: float
    ) -> Optional[dict[str, Any]]:
        """Poll ``/transparency/proof`` until the leaf anchors or the deadline
        passes. Returns the included record, or None (caller keeps the honest
        pending state, which classifies as LOGGED_UNVERIFIED)."""
        assert self._client is not None
        if not isinstance(receipt_id, str) or not receipt_id:
            return None
        url = f"{self.base_url}/transparency/proof"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                response = self._client.get(
                    url,
                    params={"receipt_id": receipt_id},
                    timeout=min(remaining, 2.0),
                )
                if response.status_code == 200:
                    out = response.json()
                    if out.get("status") == "included":
                        return {
                            "status": "included",
                            "leaf_index": out.get("leaf_index"),
                            "inclusion_proof": out.get("inclusion_proof"),
                            "sth": out.get("sth"),
                        }
            except httpx.HTTPError:
                pass  # transient; the deadline bounds the loop
            if deadline - time.monotonic() <= 0.25:
                return None
            time.sleep(0.25)

    def decompose(
        self,
        attestation: Attestation,
        items: list[dict[str, Any]],
        operation_type: str = "item",
        source_data: Any = None,
    ) -> list[Attestation]:
        """Decompose a batch attestation into individual item attestations.

        All decomposed items share the same operation_id as the parent,
        with incrementing operation_sequence starting after the parent's sequence.

        Args:
            attestation: The parent batch attestation
            items: List of individual items to attest (e.g., QA pairs)
            operation_type: Operation type for decomposed items (default: "item")
            source_data: Optional shared input data for all items

        Returns:
            List of Attestation objects, one per item
        """
        op_id = attestation.operation_id
        base_seq = attestation.operation_sequence + 1

        results: list[Attestation] = []
        for i, item in enumerate(items):
            r = self.attest(
                service_id=attestation.service_id,
                operation_type=operation_type,
                input=source_data or {"parent_attestation_id": attestation.id},
                output=item,
                operation_id=op_id,
                operation_sequence=base_seq + i,
            )
            results.append(r)

        return results

    def should_review(
        self,
        attestation: Attestation,
        sampling_rate: Optional[float] = None,
    ) -> SamplingDecision:
        """Deterministic sampling decision using nested L1/L2 tiers.

        Uses HMAC-SHA256 with domain separator per spec v1.2:
          prf_tag = HMAC-SHA256(policy_key, "sample:v1" || evidence_hash_bytes)

        If policy_key was not provided, falls back to signing_seed.

        Tier logic (nested — L2 implies L1):
        - L2 if sample_value <= l2_rate threshold (deep inspection)
        - L1 if sample_value <= l1_rate threshold (evidence collection)
        - L0 otherwise (control plane results only)

        Args:
            attestation: The attestation to evaluate for sampling.
            sampling_rate: Explicit L1 probability override (0.0-1.0).
                          If None, uses l1_rate from sampling config.

        Returns:
            SamplingDecision with level="L2", "L1", or "L0".
        """
        import hashlib
        import hmac
        import math
        import struct

        key = self._policy_key or self._signing_seed
        if not key:
            raise ValueError(
                "should_review requires policy_key or signing_seed (offline mode)"
            )

        l1_rate = sampling_rate if sampling_rate is not None else self._sampling_config.l1_rate
        l2_rate = self._sampling_config.l2_rate

        # Spec v1.2: HMAC-SHA256(policy_key, "sample:v1" || evidence_hash_bytes)
        evidence_bytes = bytes.fromhex(attestation.evidence_hash)
        message = b"sample:v1" + evidence_bytes
        tag = hmac.new(key, message, hashlib.sha256).digest()
        sample_value = struct.unpack(">Q", tag[:8])[0]

        # Nested sampling: L2 ⊂ L1
        if l2_rate > 0.0:
            if l2_rate >= 1.0:
                level = "L2"
            else:
                l2_threshold = math.floor(l2_rate * ((2**64) - 1))
                if sample_value <= l2_threshold:
                    level = "L2"
                elif l1_rate >= 1.0:
                    level = "L1"
                elif l1_rate <= 0.0:
                    level = "L0"
                else:
                    l1_threshold = math.floor(l1_rate * ((2**64) - 1))
                    level = "L1" if sample_value <= l1_threshold else "L0"
        elif l1_rate >= 1.0:
            level = "L1"
        elif l1_rate <= 0.0:
            level = "L0"
        else:
            l1_threshold = math.floor(l1_rate * ((2**64) - 1))
            level = "L1" if sample_value <= l1_threshold else "L0"

        return SamplingDecision(
            level=level,
            sample_value=sample_value,
            prf_tag=list(tag),
        )

    def verify(
        self,
        receipt: Union[str, Attestation],
    ) -> Union[VerifyResult, OfflineVerifyResult]:
        """
        Verify an attestation.

        Given an **id string**, this is a lookup: an ``oatt_`` id is read back
        from local storage and its signature checked, anything else is verified
        by the server.

        Given an **Attestation object**, the object's own Ed25519 signature is
        always checked, whatever its unsigned ``is_offline`` flag says.
        ``is_offline=False`` adds a server lookup of the object's id on top,
        and that answer is applied to the object only if the object binds to
        the log entry it returns. See ``glacis.verify.verify_attestation`` for
        the whole rule and for what the previous dispatch let through.

        Args:
            receipt: Attestation ID string or Attestation object

        Returns:
            VerifyResult (the log entry's verdict, for an object bound to it or
            an id looked up directly) or OfflineVerifyResult (the supplied
            object's own signature check)
        """
        if isinstance(receipt, str):
            if receipt.startswith("oatt_"):
                if self._storage:
                    stored = self._storage.get_receipt(receipt)
                    if stored:
                        return self._verify_offline(stored)
                raise ValueError(f"Offline receipt not found: {receipt}")
            return self._verify_online(receipt)
        elif isinstance(receipt, Attestation):
            self._debug(
                f"Verifying (object): {receipt.id} is_offline={receipt.is_offline}"
            )
            return verify_attestation(receipt, self._verify_online)
        else:
            raise TypeError(f"Invalid receipt type: {type(receipt)}")

    def _verify_online(self, attestation_id: str) -> VerifyResult:
        """Verify an online attestation via server API."""
        self._debug(f"Verifying (online): {attestation_id}")

        response = self._request_with_retry(
            "GET",
            f"{self.base_url}/v1/verify/{attestation_id}",
        )

        return VerifyResult.model_validate(response)

    def _verify_offline(self, attestation: Attestation) -> OfflineVerifyResult:
        """Verify an offline attestation's Ed25519 signature locally.

        This is a real cryptographic check against the payload rebuilt from
        the receipt's signed fields — see ``glacis.verify.verify_offline``,
        which is the single implementation the CLI uses as well. It needs no
        signing seed, so it works on a receipt someone else signed, and it
        fails when any signed field has been altered since signing.
        """
        self._debug(f"Verifying (offline): {attestation.id}")
        return verify_offline_receipt(attestation)

    def get_last_receipt(self) -> Optional[Attestation]:
        """
        Get the most recent offline attestation.

        Only available in offline mode.

        Returns:
            The most recent Attestation, or None if none exist

        Raises:
            RuntimeError: If called in online mode
        """
        if self.mode != GlacisMode.OFFLINE:
            raise RuntimeError("get_last_receipt() is only available in offline mode")

        assert self._storage is not None
        return self._storage.get_last_receipt()

    def query_log(
        self,
        org_id: Optional[str] = None,
        service_id: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        operation_id: Optional[str] = None,
    ) -> LogQueryResult:
        """
        Query the public transparency log.

        Args:
            org_id: Filter by organization ID
            service_id: Filter by service ID
            start: Start timestamp (ISO 8601)
            end: End timestamp (ISO 8601)
            limit: Maximum results (default: 50, max: 1000)
            cursor: Pagination cursor
            operation_id: Filter by operation ID

        Returns:
            Paginated log entries
        """
        params: dict[str, Any] = {}
        if org_id:
            params["orgId"] = org_id
        if service_id:
            params["serviceId"] = service_id
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if operation_id:
            params["operation_id"] = operation_id
        if limit:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor

        self._debug(f"Querying log: {params}")

        response = self._request_with_retry(
            "GET",
            f"{self.base_url}/v1/log",
            params=params,
        )

        return LogQueryResult.model_validate(response)

    def get_tree_head(self) -> TreeHeadResponse:
        """
        Get the current signed tree head.

        This is a public endpoint that does not require authentication.
        """
        response = self._request_with_retry(
            "GET",
            f"{self.base_url}/v1/root",
        )

        return TreeHeadResponse.model_validate(response)

    def hash(self, payload: Any) -> str:
        """
        Hash a payload using RFC 8785 canonical JSON + SHA-256.

        Args:
            payload: Any JSON-serializable value

        Returns:
            Hex-encoded SHA-256 hash (64 characters)
        """
        return hash_payload(payload)

    def get_api_key(self) -> str:
        """Get the API key."""
        return self.api_key

    def _request_with_retry(
        self,
        method: str,
        url: str,
        json: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Make a request with exponential backoff retry."""
        assert self._client is not None, "HTTP client not initialized"
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(
                    method,
                    url,
                    json=json,
                    params=params,
                    headers=headers,
                )

                if response.is_success:
                    result: dict[str, Any] = response.json()
                    return result

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    retry_after_ms = int(retry_after) * 1000 if retry_after else None
                    raise GlacisRateLimitError("Rate limited", retry_after_ms)

                if 400 <= response.status_code < 500:
                    # Client errors should not be retried
                    try:
                        body = response.json()
                    except Exception:
                        body = {}
                    raise GlacisApiError(
                        body.get("error", f"Request failed with status {response.status_code}"),
                        response.status_code,
                        body.get("code"),
                        body,
                    )

                # Server errors can be retried
                last_error = GlacisApiError(
                    f"Request failed with status {response.status_code}",
                    response.status_code,
                )

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e

            # Wait before retry with exponential backoff + jitter
            if attempt < self.max_retries:
                delay = min(self.base_delay * (2**attempt), self.max_delay)
                jitter = random.random() * 0.3 * delay
                time.sleep(delay + jitter)

        if last_error:
            raise last_error
        raise GlacisApiError("Request failed", 500)

    def _debug(self, message: str) -> None:
        """Log a debug message."""
        if self.debug:
            logger.debug(f"[glacis] {message}")


class AsyncGlacis:
    """
    Asynchronous GLACIS client.

    Provides async attestation, verification, and log querying for the public
    transparency log.

    Args:
        api_key: API key for authenticated endpoints
        base_url: Base URL for the API (default: https://api.glacis.io)
        debug: Enable debug logging
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries for transient errors
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay in seconds
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        debug: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
    ):
        if not api_key:
            raise ValueError("api_key is required")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.debug = debug
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        self._client = httpx.AsyncClient(timeout=timeout)

        if debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

    async def __aenter__(self) -> "AsyncGlacis":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def attest(
        self,
        service_id: str,
        operation_type: str,
        input: Any,
        output: Any,
        metadata: Optional[dict[str, str]] = None,
        control_plane_results: Optional[Union[ControlPlaneResults, dict[str, Any]]] = None,
        operation_id: Optional[str] = None,
        operation_sequence: Optional[int] = None,
        supersedes: Optional[str] = None,
    ) -> Attestation:
        """
        Attest an AI operation (async).

        Args:
            service_id: Service identifier
            operation_type: Type of operation
            input: Input data (hashed locally, never sent)
            output: Output data (hashed locally, never sent)
            metadata: Optional metadata
            control_plane_results: Optional control plane results
            operation_id: UUID linking attestations in the same operation
            operation_sequence: Ordinal sequence within the operation
            supersedes: Attestation ID this replaces (revision chains)

        Returns:
            Attestation
        """
        # I/O-only hash (evidence_hash)
        evidence_hash = self.hash({"input": input, "output": output})

        # Serialize CPR to dict if typed model
        cpr_dict: Optional[dict[str, Any]] = None
        if control_plane_results is not None:
            if hasattr(control_plane_results, "model_dump"):
                cpr_dict = control_plane_results.model_dump()
            else:
                cpr_dict = control_plane_results

        cpr_hash: Optional[str] = None
        if cpr_dict:
            cpr_hash = self.hash(cpr_dict)

        self._debug(f"Attesting: service_id={service_id}, hash={evidence_hash[:16]}...")

        body: dict[str, Any] = {
            "service_id": service_id,
            "operation_type": operation_type,
            "evidence_hash": evidence_hash,
        }

        if cpr_hash:
            body["cpr_hash"] = cpr_hash
        if cpr_dict:
            body["control_plane_results"] = cpr_dict
        if operation_id:
            body["operation_id"] = operation_id
        if operation_sequence is not None:
            body["operation_sequence"] = operation_sequence
        if supersedes:
            body["supersedes"] = supersedes

        response = await self._request_with_retry(
            "POST",
            f"{self.base_url}/v1/attest",
            json=body,
            headers={"X-Glacis-Key": self.api_key},
        )

        normalized = _normalize_server_response(response)
        attestation = Attestation.model_validate(normalized)

        if cpr_dict:
            attestation.control_plane_results = cpr_dict

        # L1/L2 Evidence (online: server determines sampling, probability unknown)
        if (
            attestation.sampling_decision
            and attestation.sampling_decision.level in ("L1", "L2")
        ):
            attestation.evidence = Evidence(
                sample_probability=0.0,
                data={"input": input, "output": output},
            )
            self._debug(
                f"L1 evidence populated (level={attestation.sampling_decision.level})"
            )

        self._debug(f"Attestation successful: {attestation.id}")
        return attestation

    async def verify(self, attestation_id: str) -> VerifyResult:
        """Verify an attestation."""
        self._debug(f"Verifying: {attestation_id}")

        response = await self._request_with_retry(
            "GET",
            f"{self.base_url}/v1/verify/{attestation_id}",
        )

        return VerifyResult.model_validate(response)

    async def query_log(
        self,
        org_id: Optional[str] = None,
        service_id: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        operation_id: Optional[str] = None,
    ) -> LogQueryResult:
        """Query the public transparency log."""
        params: dict[str, Any] = {}
        if org_id:
            params["orgId"] = org_id
        if service_id:
            params["serviceId"] = service_id
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if operation_id:
            params["operation_id"] = operation_id
        if limit:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor

        self._debug(f"Querying log: {params}")

        response = await self._request_with_retry(
            "GET",
            f"{self.base_url}/v1/log",
            params=params,
        )

        return LogQueryResult.model_validate(response)

    async def get_tree_head(self) -> TreeHeadResponse:
        """Get the current signed tree head."""
        response = await self._request_with_retry(
            "GET",
            f"{self.base_url}/v1/root",
        )

        return TreeHeadResponse.model_validate(response)

    def hash(self, payload: Any) -> str:
        """Hash a payload using RFC 8785 canonical JSON + SHA-256."""
        return hash_payload(payload)

    def get_api_key(self) -> str:
        """Get the API key."""
        return self.api_key

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        json: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Make a request with exponential backoff retry."""
        import asyncio

        assert self._client is not None, "HTTP client not initialized"
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(
                    method,
                    url,
                    json=json,
                    params=params,
                    headers=headers,
                )

                if response.is_success:
                    result: dict[str, Any] = response.json()
                    return result

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    retry_after_ms = int(retry_after) * 1000 if retry_after else None
                    raise GlacisRateLimitError("Rate limited", retry_after_ms)

                if 400 <= response.status_code < 500:
                    try:
                        body = response.json()
                    except Exception:
                        body = {}
                    raise GlacisApiError(
                        body.get("error", f"Request failed with status {response.status_code}"),
                        response.status_code,
                        body.get("code"),
                        body,
                    )

                last_error = GlacisApiError(
                    f"Request failed with status {response.status_code}",
                    response.status_code,
                )

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e

            if attempt < self.max_retries:
                delay = min(self.base_delay * (2**attempt), self.max_delay)
                jitter = random.random() * 0.3 * delay
                await asyncio.sleep(delay + jitter)

        if last_error:
            raise last_error
        raise GlacisApiError("Request failed", 500)

    def _debug(self, message: str) -> None:
        """Log a debug message."""
        if self.debug:
            logger.debug(f"[glacis] {message}")
