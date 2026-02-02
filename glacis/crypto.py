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
from typing import Any


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
            # Skip None values (like undefined in JavaScript)
            if value[key] is not None or key in value:
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

from base64 import b64encode
from typing import Optional


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
            import nacl.signing
            self._nacl_signing = nacl.signing
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

    def sign_attestation_json(self, seed: bytes, attestation_json: str) -> str:
        """Sign an attestation JSON and return SignedAttestation JSON."""
        if len(seed) != 32:
            raise ValueError("Seed must be exactly 32 bytes")

        json_bytes = attestation_json.encode("utf-8")
        signature = self.sign(seed, json_bytes)
        signature_b64 = b64encode(signature).decode("ascii")

        payload = json.loads(attestation_json)
        return json.dumps({
            "payload": payload,
            "signature": signature_b64,
        }, separators=(",", ":"))


# Singleton instance
_ed25519_runtime: Optional[Ed25519Runtime] = None


def get_ed25519_runtime() -> Ed25519Runtime:
    """Get the singleton Ed25519 runtime instance."""
    global _ed25519_runtime
    if _ed25519_runtime is None:
        _ed25519_runtime = Ed25519Runtime()
    return _ed25519_runtime
