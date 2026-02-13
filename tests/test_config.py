"""
Tests for configuration loading (glacis/config.py).
"""

import tempfile
from pathlib import Path

import pytest


class TestConfigLoading:
    """Tests for glacis.yaml configuration loading."""

    def test_load_config_default(self):
        """Loading config without file returns defaults."""
        try:
            from glacis.config import load_config

            # Load from nonexistent path
            config = load_config("/nonexistent/path/glacis.yaml")

            assert config.version == "1.0"
            assert config.controls.pii_phi.enabled is False
            assert config.controls.jailbreak.enabled is False
            assert config.attestation.offline is True
        except ImportError:
            pytest.skip("pyyaml not installed")

    def test_load_config_from_file(self):
        """Loading config from YAML file."""
        try:
            import yaml  # noqa: F401

            from glacis.config import load_config

            config_content = """
version: "1.0"

controls:
  pii_phi:
    enabled: true
    backend: presidio
    mode: fast

  jailbreak:
    enabled: true
    backend: prompt_guard_22m
    threshold: 0.7
    action: block

attestation:
  offline: false
  service_id: custom-service
"""
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write(config_content)
                f.flush()
                temp_path = f.name

            try:
                config = load_config(temp_path)

                assert config.controls.pii_phi.enabled is True
                assert config.controls.pii_phi.mode == "fast"
                assert config.controls.jailbreak.enabled is True
                assert config.controls.jailbreak.threshold == 0.7
                assert config.controls.jailbreak.action == "block"
                assert config.attestation.offline is False
                assert config.attestation.service_id == "custom-service"
            finally:
                Path(temp_path).unlink()

        except ImportError:
            pytest.skip("pyyaml not installed")

    def test_config_pii_settings(self):
        """PII control configuration."""
        try:
            from glacis.config import PiiPhiConfig

            # Default settings
            config = PiiPhiConfig()
            assert config.enabled is False
            assert config.backend == "presidio"
            assert config.mode == "fast"

            # Custom settings
            config = PiiPhiConfig(enabled=True, mode="full")
            assert config.enabled is True
            assert config.mode == "full"

        except ImportError:
            pytest.skip("pyyaml not installed")

    def test_config_jailbreak_settings(self):
        """Jailbreak control configuration."""
        try:
            from glacis.config import JailbreakConfig

            # Default settings
            config = JailbreakConfig()
            assert config.enabled is False
            assert config.backend == "prompt_guard_22m"
            assert config.threshold == 0.5
            assert config.action == "flag"

            # Custom settings
            config = JailbreakConfig(
                enabled=True,
                backend="prompt_guard_86m",
                threshold=0.8,
                action="block",
            )
            assert config.enabled is True
            assert config.backend == "prompt_guard_86m"
            assert config.threshold == 0.8
            assert config.action == "block"

        except ImportError:
            pytest.skip("pyyaml not installed")

    def test_config_partial_override(self):
        """Config with partial values uses defaults for missing."""
        try:
            import yaml  # noqa: F401

            from glacis.config import load_config

            # Only override one setting
            config_content = """
controls:
  pii_phi:
    enabled: true
"""
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write(config_content)
                f.flush()
                temp_path = f.name

            try:
                config = load_config(temp_path)

                # Overridden
                assert config.controls.pii_phi.enabled is True
                # Defaults preserved
                assert config.controls.pii_phi.backend == "presidio"
                assert config.controls.jailbreak.enabled is False
                assert config.attestation.offline is True
            finally:
                Path(temp_path).unlink()

        except ImportError:
            pytest.skip("pyyaml not installed")

    def test_config_invalid_yaml(self):
        """Handle invalid YAML gracefully."""
        try:
            from glacis.config import load_config

            config_content = "{ invalid yaml: ["

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write(config_content)
                f.flush()
                temp_path = f.name

            try:
                # Should handle error and return defaults or raise
                with pytest.raises(Exception):
                    load_config(temp_path)
            finally:
                Path(temp_path).unlink()

        except ImportError:
            pytest.skip("pyyaml not installed")
