"""
RFC 8785 Canonical JSON and SHA-256 Hashing

This module provides deterministic JSON serialization and hashing that produces
identical output to the TypeScript and Rust implementations.

The canonical JSON format follows RFC 8785:
- Object keys are sorted lexicographically by Unicode code point
- No whitespace between elements
- Numbers without unnecessary precision
- Recursive canonicalization of nested structures

Example:
    >>> from glacis.crypto import hash_payload
    >>> hash1 = hash_payload({"b": 2, "a": 1})
    >>> hash2 = hash_payload({"a": 1, "b": 2})
    >>> assert hash1 == hash2  # Keys are sorted
"""

import hashlib
import json
from typing import Any, Optional


def canonical_json(data: Any) -> str:
    """
    Serialize data to RFC 8785 canonical JSON.

    This produces deterministic JSON that is identical across all runtimes
    (Python, TypeScript, Rust).

    Args:
        data: Any JSON-serializable value

    Returns:
        Canonical JSON string

    Raises:
        ValueError: If data contains non-serializable values (NaN, Infinity)

    Example:
        >>> canonical_json({"b": 2, "a": 1})
        '{"a":1,"b":2}'
    """
    return _canonicalize_value(data)


def _canonicalize_value(value: Any) -> str:
    """Recursively canonicalize a value."""
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        # Check for non-finite numbers (not valid in JSON)
        if isinstance(value, float):
            if value != value:  # NaN check
                raise ValueError("Cannot canonicalize NaN")
            if value == float("inf") or value == float("-inf"):
                raise ValueError("Cannot canonicalize Infinity")

        # Use Python's default number serialization
        # For integers, this produces no decimal point
        # For floats, this matches JavaScript's behavior
        return json.dumps(value)

    if isinstance(value, str):
        # Use json.dumps for proper string escaping
        return json.dumps(value)

    if isinstance(value, (list, tuple)):
        elements = [_canonicalize_value(item) for item in value]
        return "[" + ",".join(elements) + "]"

    if isinstance(value, dict):
        # Sort keys lexicographically by Unicode code point (RFC 8785)
        sorted_keys = sorted(value.keys())
        pairs = []
        for key in sorted_keys:
            pairs.append(f"{json.dumps(key)}:{_canonicalize_value(value[key])}")
        return "{" + ",".join(pairs) + "}"

    raise ValueError(f"Cannot canonicalize value of type: {type(value).__name__}")


def hash_payload(data: Any) -> str:
    """
    Hash data using RFC 8785 canonical JSON + SHA-256.

    This is the primary hashing function for the transparency log.
    Produces identical hashes across Python, TypeScript, and Rust runtimes.

    Args:
        data: Any JSON-serializable value

    Returns:
        Hex-encoded SHA-256 hash (64 characters)

    Example:
        >>> hash_payload({"b": 2, "a": 1})
        'a1b2c3...'  # 64 hex characters
    """
    canonical = canonical_json(data)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_bytes(data: bytes) -> str:
    """
    Hash raw bytes using SHA-256.

    Args:
        data: Raw bytes to hash

    Returns:
        Hex-encoded SHA-256 hash (64 characters)
    """
    return hashlib.sha256(data).hexdigest()


# =============================================================================
# Ed25519 Signing (for offline attestations)
# =============================================================================

class CryptoError(Exception):
    """Error from cryptographic operations."""
    pass


class Ed25519Runtime:
    """
    Ed25519 cryptographic runtime using PyNaCl (libsodium bindings).

    Provides signing and verification for offline attestations.
    """

    def __init__(self) -> None:
        try:
            import nacl.exceptions
            import nacl.signing
            self._nacl_signing = nacl.signing
            self._nacl_exceptions = nacl.exceptions
        except ImportError:
            raise CryptoError(
                "PyNaCl not installed. Install with: pip install pynacl"
            )

    def get_public_key_hex(self, seed: bytes) -> str:
        """Get hex-encoded public key from a 32-byte seed."""
        if len(seed) != 32:
            raise ValueError("Seed must be exactly 32 bytes")
        signing_key = self._nacl_signing.SigningKey(seed)
        return bytes(signing_key.verify_key).hex()

    def sign(self, seed: bytes, message: bytes) -> bytes:
        """Sign a message with Ed25519, returning 64-byte signature."""
        if len(seed) != 32:
            raise ValueError("Seed must be exactly 32 bytes")
        signing_key = self._nacl_signing.SigningKey(seed)
        signed = signing_key.sign(message)
        return signed.signature

    def verify(self, public_key_hex: str, message: bytes, signature_hex: str) -> bool:
        """Check an Ed25519 signature over ``message`` under ``public_key_hex``.

        Verification needs no seed and no secret: the public key on the receipt
        is enough. That is the whole point — a third party who holds only the
        receipt can run exactly this check.

        Args:
            public_key_hex: 32-byte Ed25519 public key, hex-encoded.
            message: The exact bytes that were signed.
            signature_hex: 64-byte Ed25519 signature, hex-encoded.

        Returns:
            True only when the signature verifies over those exact bytes.
            A well-formed signature that does not verify returns False.

        Raises:
            CryptoError: the key or the signature is not decodable at the right
                length, so no verification could be attempted at all. "I could
                not check" is a different answer from "the signature is wrong",
                and callers name the two differently.
        """
        try:
            key_bytes = bytes.fromhex(public_key_hex)
            signature_bytes = bytes.fromhex(signature_hex)
        except (ValueError, TypeError) as e:
            raise CryptoError(f"public_key/signature is not valid hex: {e}") from e

        if len(key_bytes) != 32:
            raise CryptoError(
                f"public_key must decode to 32 bytes, got {len(key_bytes)}"
            )
        if len(signature_bytes) != 64:
            raise CryptoError(
                f"signature must decode to 64 bytes, got {len(signature_bytes)}"
            )

        try:
            self._nacl_signing.VerifyKey(key_bytes).verify(message, signature_bytes)
            return True
        except self._nacl_exceptions.BadSignatureError:
            return False

    def sign_attestation_json(self, seed: bytes, attestation_json: str) -> str:
        """Sign an attestation JSON and return SignedAttestation JSON."""
        if len(seed) != 32:
            raise ValueError("Seed must be exactly 32 bytes")

        json_bytes = attestation_json.encode("utf-8")
        signature = self.sign(seed, json_bytes)
        signature_hex = signature.hex()

        payload = json.loads(attestation_json)
        return json.dumps({
            "payload": payload,
            "signature": signature_hex,
        }, separators=(",", ":"))


# Singleton instance
_ed25519_runtime: Optional[Ed25519Runtime] = None


def get_ed25519_runtime() -> Ed25519Runtime:
    """Get the singleton Ed25519 runtime instance."""
    global _ed25519_runtime
    if _ed25519_runtime is None:
        _ed25519_runtime = Ed25519Runtime()
    return _ed25519_runtime


# =============================================================================
# The exact bytes an offline receipt is signed over
# =============================================================================

OFFLINE_PAYLOAD_VERSION = 1


def offline_signed_payload(
    *,
    service_id: str,
    operation_type: str,
    evidence_hash: str,
    timestamp_ms: str,
    operation_id: str,
    operation_sequence: int,
    control_plane_results: Optional[dict[str, Any]] = None,
    supersedes: Optional[str] = None,
) -> bytes:
    """Build the exact byte string an offline attestation is signed over.

    This is the single definition of the signed payload. Both the signer
    (``Glacis._attest_offline``) and the verifier (``glacis.verify``) call it,
    so the two cannot drift apart — a payload change that broke verification
    would break signing in the same commit.

    ``control_plane_results`` and ``supersedes`` are included only when truthy,
    which is what the signer has always done: a receipt created without them is
    signed over a payload that does not carry the key at all.

    Note what is *not* here: ``id``, ``cpr_hash``, ``public_key``, ``is_offline``.
    Those ride alongside the signature, not inside it. See
    /verify/what-a-check-proves/ for the boundary written out in full.
    """
    body: dict[str, Any] = {
        "version": OFFLINE_PAYLOAD_VERSION,
        "service_id": service_id,
        "operation_type": operation_type,
        "evidence_hash": evidence_hash,
        "timestamp_ms": timestamp_ms,
        "operation_id": operation_id,
        "operation_sequence": operation_sequence,
        "mode": "offline",
    }
    if control_plane_results:
        body["control_plane_results"] = control_plane_results
    if supersedes:
        body["supersedes"] = supersedes

    return json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")


def offline_signed_payload_for(attestation: Any) -> bytes:
    """Rebuild the signed bytes from a reconstructed ``Attestation``.

    Args:
        attestation: an ``Attestation`` (duck-typed here so that this module
            stays free of model imports).

    Raises:
        ValueError: the receipt is missing a field the signed payload needs, so
            the bytes cannot be rebuilt at all. That is a structural failure,
            not a signature failure, and callers report it as such.
    """
    if attestation.timestamp is None:
        raise ValueError(
            "receipt has no timestamp; timestamp_ms is inside the signature, "
            "so the signed payload cannot be rebuilt without it"
        )

    return offline_signed_payload(
        service_id=attestation.service_id,
        operation_type=attestation.operation_type,
        evidence_hash=attestation.evidence_hash,
        timestamp_ms=str(attestation.timestamp),
        operation_id=attestation.operation_id,
        operation_sequence=attestation.operation_sequence,
        control_plane_results=attestation.control_plane_results,
        supersedes=attestation.supersedes,
    )
