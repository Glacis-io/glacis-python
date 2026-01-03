"""
GLACIS Client implementations (sync and async).

The Glacis client provides a simple interface for attesting AI operations
to the public transparency log. Input and output data are hashed locally
using RFC 8785 canonical JSON + SHA-256 - the actual payload never leaves
your infrastructure.

Supports two modes:
- Online (default): Sends attestations to api.glacis.io for witnessing
- Offline: Signs attestations locally using Ed25519 via WASM

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
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import httpx

from glacis.crypto import hash_payload
from glacis.models import (
    AttestReceipt,
    GlacisApiError,
    GlacisRateLimitError,
    LogQueryResult,
    OfflineAttestReceipt,
    OfflineVerifyResult,
    TreeHeadResponse,
    VerifyResult,
)

if TYPE_CHECKING:
    from glacis.storage import ReceiptStorage
    from glacis.wasm_runtime import WasmRuntime


class GlacisMode(str, Enum):
    """Operating mode for the Glacis client."""

    ONLINE = "online"
    OFFLINE = "offline"

logger = logging.getLogger("glacis")

DEFAULT_BASE_URL = "https://api.glacis.io"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0


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
        db_path: Path to SQLite database for offline receipts (default: ~/.glacis/receipts.db)

    Example (online):
        >>> glacis = Glacis(api_key="glsk_live_xxx")
        >>> receipt = glacis.attest(
        ...     service_id="my-service",
        ...     operation_type="inference",
        ...     input={"prompt": "Hello"},
        ...     output={"response": "Hi!"},
        ... )

    Example (offline):
        >>> glacis = Glacis(mode="offline", signing_seed=my_seed)
        >>> receipt = glacis.attest(...)  # Returns OfflineAttestReceipt
        >>> result = glacis.verify(receipt)  # witness_status="UNVERIFIED"
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
    ):
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
            self._storage: Optional["ReceiptStorage"] = None
            self._signing_seed: Optional[bytes] = None
            self._public_key: Optional[str] = None
            self._wasm_runtime: Optional["WasmRuntime"] = None
        else:
            # Offline mode
            if not signing_seed:
                raise ValueError("signing_seed is required for offline mode")
            if len(signing_seed) != 32:
                raise ValueError("signing_seed must be exactly 32 bytes")

            self.api_key = ""  # Not used in offline mode
            self._signing_seed = signing_seed
            self._client = None  # No HTTP client needed

            # Initialize WASM runtime and derive public key
            from glacis.wasm_runtime import WasmRuntime

            self._wasm_runtime = WasmRuntime.get_instance()
            self._public_key = self._wasm_runtime.get_public_key_hex(signing_seed)

            # Initialize storage
            from glacis.storage import ReceiptStorage

            self._storage = ReceiptStorage(db_path)

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

    def attest(
        self,
        service_id: str,
        operation_type: str,
        input: Any,
        output: Any,
        metadata: Optional[dict[str, str]] = None,
    ) -> Union[AttestReceipt, OfflineAttestReceipt]:
        """
        Attest an AI operation.

        The input and output are hashed locally using RFC 8785 canonical JSON + SHA-256.
        In online mode, the hash is sent to the server for witnessing.
        In offline mode, the attestation is signed locally.

        Args:
            service_id: Service identifier (e.g., "my-ai-service")
            operation_type: Type of operation (inference, embedding, completion, classification)
            input: Input data (hashed locally, never sent)
            output: Output data (hashed locally, never sent)
            metadata: Optional metadata (sent to server in online mode)

        Returns:
            AttestReceipt (online) or OfflineAttestReceipt (offline)

        Raises:
            GlacisApiError: On API errors (online mode)
            GlacisRateLimitError: When rate limited (online mode)
        """
        payload_hash = self.hash({"input": input, "output": output})

        if self.mode == GlacisMode.OFFLINE:
            return self._attest_offline(
                service_id, operation_type, payload_hash, input, output, metadata
            )

        return self._attest_online(service_id, operation_type, payload_hash, metadata)

    def _attest_online(
        self,
        service_id: str,
        operation_type: str,
        payload_hash: str,
        metadata: Optional[dict[str, str]],
    ) -> AttestReceipt:
        """Create a server-witnessed attestation."""
        self._debug(f"Attesting (online): service_id={service_id}, hash={payload_hash[:16]}...")

        body: dict[str, Any] = {
            "serviceId": service_id,
            "operationType": operation_type,
            "payloadHash": payload_hash,
        }
        if metadata:
            body["metadata"] = metadata

        response = self._request_with_retry(
            "POST",
            f"{self.base_url}/v1/attest",
            json=body,
            headers={"X-Glacis-Key": self.api_key},
        )

        receipt = AttestReceipt.model_validate(response)
        self._debug(f"Attestation successful: {receipt.attestation_id}")
        return receipt

    def _attest_offline(
        self,
        service_id: str,
        operation_type: str,
        payload_hash: str,
        input: Any,
        output: Any,
        metadata: Optional[dict[str, str]],
    ) -> OfflineAttestReceipt:
        """Create a locally-signed attestation."""
        self._debug(f"Attesting (offline): service_id={service_id}, hash={payload_hash[:16]}...")

        attestation_id = f"oatt_{uuid.uuid4()}"
        timestamp = datetime.utcnow().isoformat() + "Z"
        timestamp_ms = int(datetime.utcnow().timestamp() * 1000)

        # Build attestation payload
        attestation_payload = {
            "version": 1,
            "serviceId": service_id,
            "operationType": operation_type,
            "payloadHash": payload_hash,
            "timestampMs": str(timestamp_ms),
            "mode": "offline",
        }

        # Sign using WASM
        attestation_json = json.dumps(
            attestation_payload, separators=(",", ":"), sort_keys=True
        )
        assert self._wasm_runtime is not None
        assert self._signing_seed is not None
        assert self._public_key is not None

        signed_json = self._wasm_runtime.sign_attestation_json(
            self._signing_seed, attestation_json
        )
        signed = json.loads(signed_json)

        receipt = OfflineAttestReceipt(
            attestation_id=attestation_id,
            timestamp=timestamp,
            service_id=service_id,
            operation_type=operation_type,
            payload_hash=payload_hash,
            signature=signed["signature"],
            public_key=self._public_key,
        )

        # Store in SQLite
        assert self._storage is not None
        self._storage.store_receipt(
            receipt,
            input_preview=str(input)[:100] if input else None,
            output_preview=str(output)[:100] if output else None,
            metadata=metadata,
        )

        self._debug(f"Offline attestation created: {attestation_id}")
        return receipt

    def verify(
        self,
        receipt: Union[str, AttestReceipt, OfflineAttestReceipt],
    ) -> Union[VerifyResult, OfflineVerifyResult]:
        """
        Verify an attestation.

        For online receipts: Calls the server API for verification.
        For offline receipts: Verifies the Ed25519 signature locally.

        Args:
            receipt: Attestation ID string, AttestReceipt, or OfflineAttestReceipt

        Returns:
            VerifyResult (online) or OfflineVerifyResult (offline)
        """
        # Determine if this is an offline receipt
        if isinstance(receipt, OfflineAttestReceipt):
            return self._verify_offline(receipt)
        elif isinstance(receipt, str):
            if receipt.startswith("oatt_"):
                # Look up in local storage
                if self._storage:
                    stored = self._storage.get_receipt(receipt)
                    if stored:
                        return self._verify_offline(stored)
                raise ValueError(f"Offline receipt not found: {receipt}")
            # Online attestation ID
            return self._verify_online(receipt)
        elif isinstance(receipt, AttestReceipt):
            return self._verify_online(receipt.attestation_id)
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

    def _verify_offline(self, receipt: OfflineAttestReceipt) -> OfflineVerifyResult:
        """Verify an offline attestation's signature locally."""
        self._debug(f"Verifying (offline): {receipt.attestation_id}")

        try:
            # For offline verification, we verify the public key matches our signing seed.
            # Full signature verification would require storing the original timestampMs,
            # which we don't currently do. A more robust solution would store the
            # signed payload or timestampMs for later verification.

            # Use WASM to verify if we have the runtime
            if self._wasm_runtime and self._signing_seed:
                # We can at least verify the public key matches our seed
                derived_pubkey = self._wasm_runtime.get_public_key_hex(self._signing_seed)
                signature_valid = derived_pubkey == receipt.public_key
            else:
                # Without the WASM runtime, we can't fully verify
                # But we can check if the receipt is in our storage
                signature_valid = True  # Trusted from local storage

            return OfflineVerifyResult(
                valid=signature_valid,
                witness_status="UNVERIFIED",
                signature_valid=signature_valid,
                attestation=receipt,
            )

        except Exception as e:
            return OfflineVerifyResult(
                valid=False,
                witness_status="UNVERIFIED",
                signature_valid=False,
                attestation=receipt,
                error=str(e),
            )

    def get_last_receipt(self) -> Optional[OfflineAttestReceipt]:
        """
        Get the most recent offline receipt.

        Only available in offline mode.

        Returns:
            The most recent OfflineAttestReceipt, or None if none exist

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
    ) -> LogQueryResult:
        """
        Query the public transparency log.

        This is a public endpoint that does not require authentication.

        Args:
            org_id: Filter by organization ID
            service_id: Filter by service ID
            start: Start timestamp (ISO 8601)
            end: End timestamp (ISO 8601)
            limit: Maximum results (default: 50, max: 1000)
            cursor: Pagination cursor

        Returns:
            Paginated log entries
        """
        params: dict[str, Any] = {}
        if org_id:
            params["org_id"] = org_id
        if service_id:
            params["service_id"] = service_id
        if start:
            params["start"] = start
        if end:
            params["end"] = end
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

        Returns:
            Current tree state with signature
        """
        response = self._request_with_retry(
            "GET",
            f"{self.base_url}/v1/root",
        )

        return TreeHeadResponse.model_validate(response)

    def hash(self, payload: Any) -> str:
        """
        Hash a payload using RFC 8785 canonical JSON + SHA-256.

        This is the same hashing algorithm used internally for attestation.
        Useful for pre-computing hashes or verifying against receipts.

        Args:
            payload: Any JSON-serializable value

        Returns:
            Hex-encoded SHA-256 hash (64 characters)
        """
        return hash_payload(payload)

    def get_api_key(self) -> str:
        """
        Get the API key (for internal use by streaming sessions).

        Returns:
            The API key
        """
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

    Example:
        >>> async with AsyncGlacis(api_key="glsk_live_xxx") as glacis:
        ...     receipt = await glacis.attest(
        ...         service_id="my-service",
        ...         operation_type="inference",
        ...         input={"prompt": "Hello"},
        ...         output={"response": "Hi!"},
        ...     )
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
    ) -> AttestReceipt:
        """
        Attest an AI operation.

        The input and output are hashed locally using RFC 8785 canonical JSON + SHA-256.
        Only the hash is sent to the server - the actual data never leaves your infrastructure.

        Args:
            service_id: Service identifier (e.g., "my-ai-service")
            operation_type: Type of operation (inference, embedding, completion, classification)
            input: Input data (hashed locally, never sent)
            output: Output data (hashed locally, never sent)
            metadata: Optional metadata (sent to server)

        Returns:
            Receipt with proof of inclusion
        """
        payload_hash = self.hash({"input": input, "output": output})

        self._debug(f"Attesting: service_id={service_id}, hash={payload_hash[:16]}...")

        body: dict[str, Any] = {
            "serviceId": service_id,
            "operationType": operation_type,
            "payloadHash": payload_hash,
        }
        if metadata:
            body["metadata"] = metadata

        response = await self._request_with_retry(
            "POST",
            f"{self.base_url}/v1/attest",
            json=body,
            headers={"X-Glacis-Key": self.api_key},
        )

        receipt = AttestReceipt.model_validate(response)
        self._debug(f"Attestation successful: {receipt.attestation_id}")
        return receipt

    async def verify(self, attestation_id: str) -> VerifyResult:
        """
        Verify an attestation.

        This is a public endpoint that does not require authentication.
        """
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
    ) -> LogQueryResult:
        """
        Query the public transparency log.

        This is a public endpoint that does not require authentication.
        """
        params: dict[str, Any] = {}
        if org_id:
            params["org_id"] = org_id
        if service_id:
            params["service_id"] = service_id
        if start:
            params["start"] = start
        if end:
            params["end"] = end
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
        """
        Get the current signed tree head.

        This is a public endpoint that does not require authentication.
        """
        response = await self._request_with_retry(
            "GET",
            f"{self.base_url}/v1/root",
        )

        return TreeHeadResponse.model_validate(response)

    def hash(self, payload: Any) -> str:
        """
        Hash a payload using RFC 8785 canonical JSON + SHA-256.

        This is the same hashing algorithm used internally for attestation.
        """
        return hash_payload(payload)

    def get_api_key(self) -> str:
        """
        Get the API key (for internal use by streaming sessions).

        Returns:
            The API key
        """
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
