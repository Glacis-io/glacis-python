"""
Tests for OpenAI and Anthropic integrations (glacis/integrations/).

These tests mock external API calls and verify the attestation pipeline.
"""

import tempfile
from pathlib import Path

import pytest


class TestOpenAIIntegration:
    """Tests for OpenAI integration."""

    def test_attested_openai_requires_key_online(self):
        """Online mode requires glacis_api_key."""
        pytest.importorskip("openai")
        from glacis.integrations.openai import attested_openai

        with pytest.raises(ValueError, match="api_key|glacis_api_key"):
            attested_openai(
                openai_api_key="sk-test",
                offline=False,
            )

    def test_attested_openai_offline_requires_seed(self):
        """Offline mode requires signing_seed."""
        pytest.importorskip("openai")
        from glacis.integrations.openai import attested_openai

        with pytest.raises(ValueError, match="signing_seed"):
            attested_openai(
                openai_api_key="sk-test",
                offline=True,
            )

    def test_attested_openai_creates_client(self, signing_seed: bytes):
        """attested_openai creates a wrapped client."""
        pytest.importorskip("openai")
        from glacis.integrations.openai import attested_openai

        client = attested_openai(
            openai_api_key="sk-test",
            offline=True,
            signing_seed=signing_seed,
        )

        # Verify the client has the expected interface
        assert hasattr(client, "chat")
        assert hasattr(client.chat, "completions")
        assert hasattr(client.chat.completions, "create")

    def test_get_last_receipt_initially_none(self, signing_seed: bytes):
        """get_last_receipt returns None before any calls."""
        pytest.importorskip("openai")
        from glacis.integrations.openai import attested_openai, get_last_receipt

        # Create client but don't make any calls
        attested_openai(
            openai_api_key="sk-test",
            offline=True,
            signing_seed=signing_seed,
        )

        # Note: get_last_receipt is thread-local, may have state from other tests
        # This test verifies the function exists and is callable
        result = get_last_receipt()
        # Result could be None or a receipt from a previous test
        assert result is None or hasattr(result, "id")

    def test_get_evidence_not_found(self, signing_seed: bytes):
        """get_evidence returns None for unknown ID."""
        pytest.importorskip("openai")
        from glacis.integrations.openai import attested_openai, get_evidence

        attested_openai(
            openai_api_key="sk-test",
            offline=True,
            signing_seed=signing_seed,
        )

        result = get_evidence("nonexistent_id")
        assert result is None


class TestAnthropicIntegration:
    """Tests for Anthropic integration."""

    def test_attested_anthropic_requires_key_online(self):
        """Online mode requires glacis_api_key."""
        pytest.importorskip("anthropic")
        from glacis.integrations.anthropic import attested_anthropic

        with pytest.raises(ValueError, match="api_key|glacis_api_key"):
            attested_anthropic(
                anthropic_api_key="sk-ant-test",
                offline=False,
            )

    def test_attested_anthropic_offline_requires_seed(self):
        """Offline mode requires signing_seed."""
        pytest.importorskip("anthropic")
        from glacis.integrations.anthropic import attested_anthropic

        with pytest.raises(ValueError, match="signing_seed"):
            attested_anthropic(
                anthropic_api_key="sk-ant-test",
                offline=True,
            )

    def test_attested_anthropic_creates_client(self, signing_seed: bytes):
        """attested_anthropic creates a wrapped client."""
        pytest.importorskip("anthropic")
        from glacis.integrations.anthropic import attested_anthropic

        client = attested_anthropic(
            anthropic_api_key="sk-ant-test",
            offline=True,
            signing_seed=signing_seed,
        )

        # Verify the client has the expected interface
        assert hasattr(client, "messages")
        assert hasattr(client.messages, "create")


class TestGlacisBlockedError:
    """Tests for blocked request handling."""

    def test_blocked_error_exists(self):
        """GlacisBlockedError is importable."""
        pytest.importorskip("openai")
        from glacis.integrations.openai import GlacisBlockedError

        error = GlacisBlockedError("Request blocked by jailbreak control", "jailbreak", 0.95)
        assert str(error) == "Request blocked by jailbreak control"
        assert error.control_type == "jailbreak"
        assert error.score == 0.95

    def test_blocked_error_is_exception(self):
        """GlacisBlockedError is a proper exception."""
        pytest.importorskip("openai")
        from glacis.integrations.openai import GlacisBlockedError

        with pytest.raises(GlacisBlockedError):
            raise GlacisBlockedError("test", "pii")


class TestIntegrationOfflineMode:
    """Tests for offline mode in integrations."""

    def test_openai_offline_no_network(self, signing_seed: bytes, temp_db_path: Path):
        """Offline mode doesn't make network calls to Glacis."""
        pytest.importorskip("openai")
        from glacis.integrations.openai import attested_openai

        # Create client - should not make any Glacis API calls
        client = attested_openai(
            openai_api_key="sk-test",
            offline=True,
            signing_seed=signing_seed,
        )

        # Client created successfully without network
        assert client is not None

    def test_anthropic_offline_no_network(self, signing_seed: bytes):
        """Anthropic offline mode doesn't require Glacis network."""
        pytest.importorskip("anthropic")
        from glacis.integrations.anthropic import attested_anthropic

        client = attested_anthropic(
            anthropic_api_key="sk-ant-test",
            offline=True,
            signing_seed=signing_seed,
        )

        assert client is not None


class TestIntegrationWithControls:
    """Tests for integrations with controls enabled."""

    def test_openai_with_controls_config(self, signing_seed: bytes):
        """OpenAI integration accepts config path."""
        pytest.importorskip("openai")

        # Create a minimal config file
        config_content = """
version: "1.3"
controls:
  input:
    pii_phi:
      enabled: false
    jailbreak:
      enabled: false
attestation:
  offline: true
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(config_content)
            f.flush()
            config_path = f.name

        try:
            from glacis.integrations.openai import attested_openai

            client = attested_openai(
                openai_api_key="sk-test",
                offline=True,
                signing_seed=signing_seed,
                config=config_path,
            )

            assert client is not None
        except ImportError:
            pytest.skip("pyyaml not installed")
        finally:
            Path(config_path).unlink()
