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
            assert receipt.id.startswith("oatt_")
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
            stored = glacis2._storage.get_receipt(receipt.id)

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
            assert last.id == receipt2.id

            glacis.close()

    def test_ed25519_runtime_loads(self):
        """Ed25519 runtime loads and functions correctly."""
        from glacis.crypto import get_ed25519_runtime

        runtime = get_ed25519_runtime()

        # Test public key derivation
        seed = bytes(32)  # All zeros
        pubkey_hex = runtime.get_public_key_hex(seed)
        assert len(pubkey_hex) == 64

    def test_ed25519_signing(self):
        """Ed25519 signing works."""
        from glacis.crypto import get_ed25519_runtime

        runtime = get_ed25519_runtime()

        seed = os.urandom(32)
        message = b"test message"

        # Sign
        signature = runtime.sign(seed, message)
        assert len(signature) == 64


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


class TestGeminiOfflineIntegration:
    """Test Gemini integration with offline mode (no actual API calls)."""

    def test_attested_gemini_requires_seed_for_offline(self):
        """attested_gemini with offline=True requires signing_seed."""
        pytest.importorskip("google.genai")
        from glacis.integrations.gemini import attested_gemini

        with pytest.raises(ValueError, match="signing_seed is required"):
            attested_gemini(offline=True)

    def test_attested_gemini_offline_creates_client(self):
        """attested_gemini with offline mode creates a wrapped client."""
        pytest.importorskip("google.genai")
        from glacis.integrations.gemini import attested_gemini

        seed = os.urandom(32)
        client = attested_gemini(
            gemini_api_key="fake-key",
            offline=True,
            signing_seed=seed,
        )

        # Check the client was created and wrapped
        assert hasattr(client.models, "generate_content")

    def test_attested_gemini_requires_key_online(self):
        """Online mode requires glacis_api_key."""
        pytest.importorskip("google.genai")
        from glacis.integrations.gemini import attested_gemini

        with pytest.raises(ValueError, match="api_key|glacis_api_key"):
            attested_gemini(
                gemini_api_key="fake-key",
                offline=False,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
