"""
WASM runtime for offline cryptographic operations.

Uses wasmtime-py to execute the s3p-core WASM module compiled for WASI target.
Falls back to PyNaCl if WASM runtime fails.

This provides Ed25519 signing and verification without reimplementing crypto in Python.
"""

import ctypes
import hashlib
import json
from base64 import b64encode
from pathlib import Path
from typing import Any, Optional


class WasmRuntimeError(Exception):
    """Error from WASM runtime operations."""

    pass


class PyNaClRuntime:
    """
    Pure Python Ed25519 runtime using PyNaCl (libsodium bindings).

    Used as a fallback when WASM runtime is unavailable or fails.
    """

    def __init__(self) -> None:
        try:
            import nacl.encoding
            import nacl.signing
            self._nacl_signing = nacl.signing
            self._nacl_encoding = nacl.encoding
        except ImportError:
            raise WasmRuntimeError(
                "PyNaCl not installed. Install with: pip install pynacl"
            )

    def derive_public_key(self, seed: bytes) -> bytes:
        """Derive Ed25519 public key from a 32-byte seed."""
        if len(seed) != 32:
            raise ValueError("Seed must be exactly 32 bytes")
        signing_key = self._nacl_signing.SigningKey(seed)
        return bytes(signing_key.verify_key)

    def get_public_key_hex(self, seed: bytes) -> str:
        """Get hex-encoded public key from a 32-byte seed."""
        return self.derive_public_key(seed).hex()

    def sha256(self, data: bytes) -> bytes:
        """Compute SHA-256 hash."""
        return hashlib.sha256(data).digest()

    def ed25519_sign(self, seed: bytes, message: bytes) -> bytes:
        """Sign a message with Ed25519."""
        if len(seed) != 32:
            raise ValueError("Seed must be exactly 32 bytes")
        signing_key = self._nacl_signing.SigningKey(seed)
        signed = signing_key.sign(message)
        return signed.signature  # Just the 64-byte signature

    def ed25519_verify(
        self, public_key: bytes, message: bytes, signature: bytes
    ) -> bool:
        """Verify an Ed25519 signature."""
        if len(public_key) != 32:
            raise ValueError("Public key must be exactly 32 bytes")
        if len(signature) != 64:
            raise ValueError("Signature must be exactly 64 bytes")
        try:
            verify_key = self._nacl_signing.VerifyKey(public_key)
            verify_key.verify(message, signature)
            return True
        except Exception:
            return False

    def sign_attestation_json(self, seed: bytes, attestation_json: str) -> str:
        """Sign an attestation JSON and return SignedAttestation JSON."""
        if len(seed) != 32:
            raise ValueError("Seed must be exactly 32 bytes")

        json_bytes = attestation_json.encode("utf-8")
        signature = self.ed25519_sign(seed, json_bytes)
        signature_b64 = b64encode(signature).decode("ascii")

        # Parse and re-serialize to ensure valid JSON
        payload = json.loads(attestation_json)
        return json.dumps({
            "payload": payload,
            "signature": signature_b64,
        }, separators=(",", ":"))


class WasmRuntime:
    """
    Singleton WASM runtime for Ed25519 cryptographic operations.

    Loads the s3p_core_wasi.wasm module and provides Python-friendly wrappers
    around the low-level C-ABI functions.
    """

    _instance: Optional["WasmRuntime"] = None

    def __init__(self) -> None:
        """Initialize the WASM runtime and load the s3p-core module."""
        try:
            import wasmtime
        except ImportError:
            raise WasmRuntimeError("wasmtime not installed. Install with: pip install wasmtime")

        # Create engine and linker
        self._engine = wasmtime.Engine()
        self._linker = wasmtime.Linker(self._engine)
        self._linker.define_wasi()

        # Load WASM module from package
        wasm_path = Path(__file__).parent / "wasm" / "s3p_core_wasi.wasm"
        if not wasm_path.exists():
            raise WasmRuntimeError(f"WASM module not found at {wasm_path}")

        self._module = wasmtime.Module.from_file(self._engine, str(wasm_path))

        # Create store with WASI config
        wasi_config = wasmtime.WasiConfig()
        wasi_config.inherit_stdout()
        wasi_config.inherit_stderr()

        self._store = wasmtime.Store(self._engine)
        self._store.set_wasi(wasi_config)

        # Instantiate the module
        self._wasm_instance = self._linker.instantiate(self._store, self._module)

        # Initialize WASI Reactor if present
        exports = self._wasm_instance.exports(self._store)
        if "_initialize" in exports:
            exports["_initialize"](self._store)

        # Get memory and exported functions
        self._memory = exports["memory"]
        self._alloc = exports["wasi_alloc"]
        self._dealloc = exports["wasi_dealloc"]
        self._derive_public_key = exports["wasi_derive_public_key"]
        self._sha256 = exports["wasi_sha256"]
        self._hash_canonical_json = exports["wasi_hash_canonical_json"]
        self._ed25519_sign = exports["wasi_ed25519_sign"]
        self._ed25519_verify = exports["wasi_ed25519_verify"]
        self._sign_attestation_json = exports["wasi_sign_attestation_json"]
        self._get_public_key_hex = exports["wasi_get_public_key_hex"]

    @classmethod
    def get_instance(cls, use_wasm: bool = False) -> "WasmRuntime":
        """
        Get or create the singleton runtime instance.

        Args:
            use_wasm: If True, try WASM runtime first. If False (default),
                     use PyNaCl directly for better compatibility.

        Falls back to PyNaCl if WASM runtime initialization fails.
        """
        if cls._instance is None:
            if use_wasm:
                try:
                    cls._instance = WasmRuntime()
                except Exception:
                    # WASM failed, use PyNaCl fallback
                    cls._instance = PyNaClRuntime()  # type: ignore[assignment]
            else:
                # Use PyNaCl by default for better compatibility
                cls._instance = PyNaClRuntime()  # type: ignore[assignment]
        assert cls._instance is not None
        return cls._instance

    def _write_bytes(self, data: bytes) -> int:
        """Allocate WASM memory and write bytes, returning the pointer."""
        size = len(data)
        ptr: int = self._alloc(self._store, size)
        if ptr == 0:
            raise WasmRuntimeError("Failed to allocate WASM memory")

        # Write data to WASM memory using ctypes
        mem_ptr = self._memory.data_ptr(self._store)
        base_addr = ctypes.addressof(mem_ptr.contents)
        dest_addr = base_addr + ptr

        ctypes.memmove(dest_addr, data, size)

        return ptr

    def _read_bytes(self, ptr: int, size: int) -> bytes:
        """Read bytes from WASM memory."""
        mem_ptr = self._memory.data_ptr(self._store)
        base_addr = ctypes.addressof(mem_ptr.contents)
        src_addr = base_addr + ptr

        return ctypes.string_at(src_addr, size)

    def _free(self, ptr: int, size: int) -> None:
        """Free WASM memory."""
        if ptr != 0 and size > 0:
            self._dealloc(self._store, ptr, size)

    def derive_public_key(self, seed: bytes) -> bytes:
        """
        Derive Ed25519 public key from a 32-byte seed.

        Args:
            seed: 32-byte Ed25519 signing seed

        Returns:
            32-byte public key

        Raises:
            WasmRuntimeError: If derivation fails
            ValueError: If seed is not 32 bytes
        """
        if len(seed) != 32:
            raise ValueError("Seed must be exactly 32 bytes")

        # Allocate memory for input and output
        seed_ptr = self._write_bytes(seed)
        out_ptr = self._alloc(self._store, 32)
        if out_ptr == 0:
            self._free(seed_ptr, 32)
            raise WasmRuntimeError("Failed to allocate output buffer")

        try:
            result = self._derive_public_key(self._store, seed_ptr, 32, out_ptr)
            if result != 0:
                raise WasmRuntimeError(f"derive_public_key failed with code {result}")

            return self._read_bytes(out_ptr, 32)
        finally:
            self._free(seed_ptr, 32)
            self._free(out_ptr, 32)

    def get_public_key_hex(self, seed: bytes) -> str:
        """
        Get hex-encoded public key from a 32-byte seed.

        Args:
            seed: 32-byte Ed25519 signing seed

        Returns:
            64-character hex string of public key

        Raises:
            WasmRuntimeError: If derivation fails
            ValueError: If seed is not 32 bytes
        """
        if len(seed) != 32:
            raise ValueError("Seed must be exactly 32 bytes")

        seed_ptr = self._write_bytes(seed)
        out_ptr = self._alloc(self._store, 64)
        if out_ptr == 0:
            self._free(seed_ptr, 32)
            raise WasmRuntimeError("Failed to allocate output buffer")

        try:
            result = self._get_public_key_hex(self._store, seed_ptr, 32, out_ptr)
            if result != 0:
                raise WasmRuntimeError(f"get_public_key_hex failed with code {result}")

            return self._read_bytes(out_ptr, 64).decode("ascii")
        finally:
            self._free(seed_ptr, 32)
            self._free(out_ptr, 64)

    def sha256(self, data: bytes) -> bytes:
        """
        Compute SHA-256 hash.

        Args:
            data: Input bytes to hash

        Returns:
            32-byte hash

        Raises:
            WasmRuntimeError: If hashing fails
        """
        data_ptr = self._write_bytes(data)
        out_ptr = self._alloc(self._store, 32)
        if out_ptr == 0:
            self._free(data_ptr, len(data))
            raise WasmRuntimeError("Failed to allocate output buffer")

        try:
            result = self._sha256(self._store, data_ptr, len(data), out_ptr)
            if result != 0:
                raise WasmRuntimeError(f"sha256 failed with code {result}")

            return self._read_bytes(out_ptr, 32)
        finally:
            self._free(data_ptr, len(data))
            self._free(out_ptr, 32)

    def hash_canonical_json(self, json_str: str) -> bytes:
        """
        Hash a JSON string using RFC 8785 canonical JSON + SHA-256.

        Args:
            json_str: JSON string to canonicalize and hash

        Returns:
            32-byte hash

        Raises:
            WasmRuntimeError: If hashing fails
        """
        json_bytes = json_str.encode("utf-8")
        json_ptr = self._write_bytes(json_bytes)
        out_ptr = self._alloc(self._store, 32)
        if out_ptr == 0:
            self._free(json_ptr, len(json_bytes))
            raise WasmRuntimeError("Failed to allocate output buffer")

        try:
            result = self._hash_canonical_json(
                self._store, json_ptr, len(json_bytes), out_ptr
            )
            if result != 0:
                raise WasmRuntimeError(
                    f"hash_canonical_json failed with code {result}"
                )

            return self._read_bytes(out_ptr, 32)
        finally:
            self._free(json_ptr, len(json_bytes))
            self._free(out_ptr, 32)

    def ed25519_sign(self, seed: bytes, message: bytes) -> bytes:
        """
        Sign a message with Ed25519.

        Args:
            seed: 32-byte signing seed
            message: Message bytes to sign

        Returns:
            64-byte signature

        Raises:
            WasmRuntimeError: If signing fails
            ValueError: If seed is not 32 bytes
        """
        if len(seed) != 32:
            raise ValueError("Seed must be exactly 32 bytes")

        seed_ptr = self._write_bytes(seed)
        msg_ptr = self._write_bytes(message)
        out_ptr = self._alloc(self._store, 64)
        if out_ptr == 0:
            self._free(seed_ptr, 32)
            self._free(msg_ptr, len(message))
            raise WasmRuntimeError("Failed to allocate output buffer")

        try:
            result = self._ed25519_sign(
                self._store, seed_ptr, 32, msg_ptr, len(message), out_ptr
            )
            if result != 0:
                raise WasmRuntimeError(f"ed25519_sign failed with code {result}")

            return self._read_bytes(out_ptr, 64)
        finally:
            self._free(seed_ptr, 32)
            self._free(msg_ptr, len(message))
            self._free(out_ptr, 64)

    def ed25519_verify(
        self, public_key: bytes, message: bytes, signature: bytes
    ) -> bool:
        """
        Verify an Ed25519 signature.

        Args:
            public_key: 32-byte public key
            message: Message bytes that were signed
            signature: 64-byte signature to verify

        Returns:
            True if signature is valid, False otherwise

        Raises:
            ValueError: If public_key or signature have wrong length
        """
        if len(public_key) != 32:
            raise ValueError("Public key must be exactly 32 bytes")
        if len(signature) != 64:
            raise ValueError("Signature must be exactly 64 bytes")

        pubkey_ptr = self._write_bytes(public_key)
        msg_ptr = self._write_bytes(message)
        sig_ptr = self._write_bytes(signature)

        try:
            result: int = self._ed25519_verify(
                self._store, pubkey_ptr, 32, msg_ptr, len(message), sig_ptr, 64
            )
            return result == 0
        finally:
            self._free(pubkey_ptr, 32)
            self._free(msg_ptr, len(message))
            self._free(sig_ptr, 64)

    def sign_attestation_json(self, seed: bytes, attestation_json: str) -> str:
        """
        Sign an attestation JSON and return SignedAttestation JSON.

        Args:
            seed: 32-byte signing seed
            attestation_json: JSON string of attestation payload

        Returns:
            JSON string of SignedAttestation with payload and signature

        Raises:
            WasmRuntimeError: If signing fails
            ValueError: If seed is not 32 bytes
        """
        if len(seed) != 32:
            raise ValueError("Seed must be exactly 32 bytes")

        json_bytes = attestation_json.encode("utf-8")
        seed_ptr = self._write_bytes(seed)
        json_ptr = self._write_bytes(json_bytes)

        # Allocate generous output buffer (input + base64 sig + JSON wrapper)
        out_cap = len(json_bytes) + 200
        out_ptr = self._alloc(self._store, out_cap)
        # size_t on wasm32 is 4 bytes, but use 8 for safety
        out_len_ptr = self._alloc(self._store, 8)

        if out_ptr == 0 or out_len_ptr == 0:
            self._free(seed_ptr, 32)
            self._free(json_ptr, len(json_bytes))
            self._free(out_ptr, out_cap)
            self._free(out_len_ptr, 8)
            raise WasmRuntimeError("Failed to allocate output buffer")

        try:
            result = self._sign_attestation_json(
                self._store,
                seed_ptr,
                32,
                json_ptr,
                len(json_bytes),
                out_ptr,
                out_cap,
                out_len_ptr,
            )
            if result != 0:
                raise WasmRuntimeError(
                    f"sign_attestation_json failed with code {result}"
                )

            # Read output length (4 bytes for wasm32 usize)
            out_len_bytes = self._read_bytes(out_len_ptr, 4)
            out_len = int.from_bytes(out_len_bytes, "little")

            # Read output JSON
            return self._read_bytes(out_ptr, out_len).decode("utf-8")
        finally:
            self._free(seed_ptr, 32)
            self._free(json_ptr, len(json_bytes))
            self._free(out_ptr, out_cap)
            self._free(out_len_ptr, 8)


def sign_offline_attestation(
    signing_seed: bytes,
    payload_hash: str,
    service_id: str,
    operation_type: str,
    timestamp_ms: int,
) -> dict[str, Any]:
    """
    Create and sign an offline attestation.

    This is a convenience function that builds the attestation structure
    and signs it using the WASM runtime.

    Args:
        signing_seed: 32-byte Ed25519 signing seed
        payload_hash: SHA-256 hash of input+output (64-char hex string)
        service_id: Service identifier
        operation_type: Type of operation (e.g., "completion")
        timestamp_ms: Timestamp in milliseconds

    Returns:
        Dict with 'payload' and 'signature' keys (SignedAttestation structure)

    Raises:
        WasmRuntimeError: If signing fails
        ValueError: If signing_seed is not 32 bytes
    """
    import json

    runtime = WasmRuntime.get_instance()

    # Build a minimal attestation structure for offline use
    # This is simpler than AttestationL0 since we don't need DPRF sampling
    attestation = {
        "version": 1,
        "serviceId": service_id,
        "operationType": operation_type,
        "payloadHash": payload_hash,
        "timestampMs": str(timestamp_ms),
        "mode": "offline",
    }

    attestation_json = json.dumps(attestation, separators=(",", ":"), sort_keys=True)
    signed_json = runtime.sign_attestation_json(signing_seed, attestation_json)

    result: dict[str, Any] = json.loads(signed_json)
    return result


def get_runtime() -> str:
    """
    Get the current runtime type name.

    Returns:
        String identifying the runtime: "WasmRuntime" or "PyNaClRuntime"
    """
    runtime = WasmRuntime.get_instance()
    return type(runtime).__name__
