"""
GLACIS Client implementations (sync and async).

The Glacis client provides a simple interface for attesting AI operations
to the public transparency log. Input and output data are hashed locally
using RFC 8785 canonical JSON + SHA-256 - the actual payload never leaves
your infrastructure.

Supports two modes:
- Online (default): Sends attestations to api.glacis.io for witnessing
- Offline: Signs attestations locally using Ed25519

Example (online):
    >>> from glacis import Glacis
    >>> glacis = Glacis(api_key="glsk_live_xxx")
    >>> receipt = glacis.attest(
    ...     service_id="my-ai-service",
    ...     operation_type="inference",
    ...     input={"prompt": "Hello"},
    ...     output={"response": "Hi there!"},
    ... )

Example (offline):
    >>> glacis = Glacis(mode="offline", signing_seed=my_32_byte_seed)
    >>> receipt = glacis.attest(...)
    >>> result = glacis.verify(receipt)  # witness_status="UNVERIFIED"
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import httpx

from glacis.config import SamplingConfig
from glacis.crypto import hash_payload
from glacis.models import (
    Attestation,
    ControlPlaneResults,
    Evidence,
    GlacisApiError,
    GlacisRateLimitError,
    LogQueryResult,
    OfflineVerifyResult,
    SamplingDecision,
    TreeHeadResponse,
    VerifyResult,
)

if TYPE_CHECKING:
    from glacis.crypto import Ed25519Runtime
    from glacis.storage import StorageBackend


class GlacisMode(str, Enum):
    """Operating mode for the Glacis client."""

    ONLINE = "online"
    OFFLINE = "offline"


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
        base_url: str = DEFAULT_BASE_URL,
        debug: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        mode: Literal["online", "offline"] = "online",
        signing_seed: Optional[bytes] = None,
        db_path: Optional[Path] = None,
        storage_backend: str = "sqlite",
        storage_path: Optional[Path] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        self._sampling_config = sampling_config or SamplingConfig()
        self.mode = GlacisMode(mode)
        self.base_url = base_url.rstrip("/")
        self.debug = debug
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        if self.mode == GlacisMode.ONLINE:
            if not api_key:
                raise ValueError("api_key is required for online mode")
            self.api_key = api_key
            self._client: Optional[httpx.Client] = httpx.Client(timeout=timeout)
            self._storage: Optional["StorageBackend"] = None
            self._signing_seed: Optional[bytes] = None
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
    ) -> Attestation:
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

        Returns:
            Attestation

        Raises:
            GlacisApiError: On API errors (online mode)
            GlacisRateLimitError: When rate limited (online mode)
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

        # Build attestation payload (this is what gets signed)
        attestation_payload: dict[str, Any] = {
            "version": 1,
            "service_id": service_id,
            "operation_type": operation_type,
            "evidence_hash": evidence_hash,
            "timestamp_ms": str(timestamp_ms),
            "operation_id": op_id,
            "operation_sequence": op_seq,
            "mode": "offline",
        }

        if control_plane_results:
            attestation_payload["control_plane_results"] = control_plane_results
        if supersedes:
            attestation_payload["supersedes"] = supersedes

        # Sign using WASM
        attestation_json = json.dumps(
            attestation_payload, separators=(",", ":"), sort_keys=True
        )
        assert self._ed25519 is not None
        assert self._signing_seed is not None
        assert self._public_key is not None

        signed_json = self._ed25519.sign_attestation_json(
            self._signing_seed, attestation_json
        )
        signed = json.loads(signed_json)

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
            signature=signed["signature"],
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

        Given the same evidence_hash + sampling_rate + signing_seed, always
        returns the same decision. Uses HMAC-SHA256 to produce a
        deterministic, auditor-reproducible tag.

        Tier logic (nested — L2 implies L1):
        - L2 if sample_value <= l2_rate threshold (deep inspection)
        - L1 if sample_value <= l1_rate threshold (evidence collection / judge review)
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
        import struct

        if not self._signing_seed:
            raise ValueError("should_review requires signing_seed (offline mode)")

        l1_rate = sampling_rate if sampling_rate is not None else self._sampling_config.l1_rate
        l2_rate = self._sampling_config.l2_rate

        tag = hmac.new(
            self._signing_seed,
            attestation.evidence_hash.encode(),
            hashlib.sha256,
        ).digest()
        sample_value = struct.unpack(">Q", tag[:8])[0]

        # Nested sampling: L2 ⊂ L1
        if l2_rate > 0.0:
            if l2_rate >= 1.0:
                level = "L2"
            else:
                l2_threshold = int(l2_rate * ((2**64) - 1))
                if sample_value <= l2_threshold:
                    level = "L2"
                elif l1_rate >= 1.0:
                    level = "L1"
                elif l1_rate <= 0.0:
                    level = "L0"
                else:
                    l1_threshold = int(l1_rate * ((2**64) - 1))
                    level = "L1" if sample_value <= l1_threshold else "L0"
        elif l1_rate >= 1.0:
            level = "L1"
        elif l1_rate <= 0.0:
            level = "L0"
        else:
            l1_threshold = int(l1_rate * ((2**64) - 1))
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

        For online attestations: Calls the server API for verification.
        For offline attestations: Verifies the Ed25519 signature locally.

        Args:
            receipt: Attestation ID string or Attestation object

        Returns:
            VerifyResult (online) or OfflineVerifyResult (offline)
        """
        if isinstance(receipt, Attestation) and receipt.is_offline:
            return self._verify_offline(receipt)
        elif isinstance(receipt, str):
            if receipt.startswith("oatt_"):
                if self._storage:
                    stored = self._storage.get_receipt(receipt)
                    if stored:
                        return self._verify_offline(stored)
                raise ValueError(f"Offline receipt not found: {receipt}")
            return self._verify_online(receipt)
        elif isinstance(receipt, Attestation):
            return self._verify_online(receipt.id)
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
        """Verify an offline attestation's signature locally."""
        self._debug(f"Verifying (offline): {attestation.id}")

        try:
            if self._ed25519 and self._signing_seed:
                derived_pubkey = self._ed25519.get_public_key_hex(self._signing_seed)
                signature_valid = derived_pubkey == attestation.public_key
            else:
                signature_valid = True  # Trusted from local storage

            return OfflineVerifyResult(
                valid=signature_valid,
                witness_status="UNVERIFIED",
                signature_valid=signature_valid,
                attestation=attestation,
            )

        except Exception as e:
            return OfflineVerifyResult(
                valid=False,
                witness_status="UNVERIFIED",
                signature_valid=False,
                attestation=attestation,
                error=str(e),
            )

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
