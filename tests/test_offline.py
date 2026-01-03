"""
Acceptance tests for offline mode.

Tests the core offline functionality without needing OpenAI API access.
"""

import os
import tempfile
from pathlib import Path

import pytest


class TestOfflineMode:
    """Test offline attestation mode."""

    def test_offline_requires_signing_seed(self):
        """Offline mode requires a signing seed."""
        from glacis import Glacis

        with pytest.raises(ValueError, match="signing_seed is required"):
            Glacis(mode="offline")

    def test_offline_seed_must_be_32_bytes(self):
        """Signing seed must be exactly 32 bytes."""
        from glacis import Glacis

        with pytest.raises(ValueError, match="32 bytes"):
            Glacis(mode="offline", signing_seed=b"short")

    def test_offline_attest_creates_local_receipt(self):
        """Offline attestation creates a local receipt."""
        from glacis import Glacis

        seed = os.urandom(32)
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            glacis = Glacis(mode="offline", signing_seed=seed, db_path=db_path)

            receipt = glacis.attest(
                service_id="test-service",
                operation_type="inference",
                input={"prompt": "Hello"},
                output={"response": "Hi!"},
            )

            # Check receipt properties
            assert receipt.attestation_id.startswith("oatt_")
            assert receipt.witness_status == "UNVERIFIED"
            assert receipt.service_id == "test-service"
            assert receipt.operation_type == "inference"
            assert len(receipt.payload_hash) == 64  # SHA-256 hex
            assert len(receipt.signature) > 0
            assert len(receipt.public_key) == 64  # Ed25519 pubkey hex

            glacis.close()

    def test_offline_verify_returns_unverified(self):
        """Verify returns witness_status=UNVERIFIED for offline receipts."""
        from glacis import Glacis

        seed = os.urandom(32)
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            glacis = Glacis(mode="offline", signing_seed=seed, db_path=db_path)

            receipt = glacis.attest(
                service_id="test",
                operation_type="inference",
                input={"a": 1},
                output={"b": 2},
            )

            result = glacis.verify(receipt)

            assert result.valid is True
            assert result.witness_status == "UNVERIFIED"
            assert result.signature_valid is True

            glacis.close()

    def test_offline_receipts_persisted_to_sqlite(self):
        """Receipts are persisted to SQLite and can be retrieved."""
        from glacis import Glacis

        seed = os.urandom(32)
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Create first client and attest
            glacis1 = Glacis(mode="offline", signing_seed=seed, db_path=db_path)
            receipt = glacis1.attest(
                service_id="persist-test",
                operation_type="inference",
                input={"x": 1},
                output={"y": 2},
            )
            glacis1.close()

            # Create new client and retrieve
            glacis2 = Glacis(mode="offline", signing_seed=seed, db_path=db_path)
            stored = glacis2._storage.get_receipt(receipt.attestation_id)

            assert stored is not None
            assert stored.payload_hash == receipt.payload_hash
            assert stored.signature == receipt.signature
            assert stored.public_key == receipt.public_key

            glacis2.close()

    def test_get_last_receipt(self):
        """get_last_receipt returns the most recent receipt."""
        from glacis import Glacis

        seed = os.urandom(32)
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            glacis = Glacis(mode="offline", signing_seed=seed, db_path=db_path)

            receipt1 = glacis.attest(
                service_id="test",
                operation_type="inference",
                input={"n": 1},
                output={"r": 1},
            )
            receipt2 = glacis.attest(
                service_id="test",
                operation_type="inference",
                input={"n": 2},
                output={"r": 2},
            )

            last = glacis.get_last_receipt()
            assert last is not None
            assert last.attestation_id == receipt2.attestation_id

            glacis.close()

    def test_wasm_runtime_loads(self):
        """WASM runtime loads and functions correctly."""
        from glacis.wasm_runtime import WasmRuntime

        runtime = WasmRuntime.get_instance()

        # Test SHA-256
        hash_result = runtime.sha256(b"hello world")
        expected = bytes.fromhex(
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        )
        assert hash_result == expected

        # Test public key derivation
        seed = bytes(32)  # All zeros
        pubkey = runtime.derive_public_key(seed)
        assert len(pubkey) == 32

        pubkey_hex = runtime.get_public_key_hex(seed)
        assert len(pubkey_hex) == 64
        assert pubkey.hex() == pubkey_hex

    def test_wasm_signing_and_verification(self):
        """WASM Ed25519 signing and verification works."""
        from glacis.wasm_runtime import WasmRuntime

        runtime = WasmRuntime.get_instance()

        seed = os.urandom(32)
        message = b"test message"

        # Sign
        signature = runtime.ed25519_sign(seed, message)
        assert len(signature) == 64

        # Verify
        pubkey = runtime.derive_public_key(seed)
        assert runtime.ed25519_verify(pubkey, message, signature) is True

        # Verify with wrong message should fail
        assert runtime.ed25519_verify(pubkey, b"wrong", signature) is False


class TestOpenAIOfflineIntegration:
    """Test OpenAI integration with offline mode (no actual API calls)."""

    def test_attested_openai_requires_seed_for_offline(self):
        """attested_openai with offline=True requires signing_seed."""
        pytest.importorskip("openai")
        from glacis.integrations.openai import attested_openai

        with pytest.raises(ValueError, match="signing_seed is required"):
            attested_openai(offline=True)

    def test_attested_openai_offline_creates_client(self):
        """attested_openai with offline mode creates a wrapped client."""
        pytest.importorskip("openai")
        from glacis.integrations.openai import attested_openai

        seed = os.urandom(32)
        # This should work without OpenAI API key for client creation
        # (will fail on actual API call, but client creation should succeed)
        client = attested_openai(
            openai_api_key="sk-fake-key",
            offline=True,
            signing_seed=seed,
        )

        # Check the client was created and wrapped
        assert hasattr(client.chat.completions, "create")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
