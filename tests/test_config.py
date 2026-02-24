"""
Tests for configuration loading (glacis/config.py).

Tests the v1.3 nested config schema with input/output stages,
per-control configs, and output_block_action.
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

            config = load_config("/nonexistent/path/glacis.yaml")

            assert config.version == "1.3"
            assert config.controls.input.pii_phi.enabled is False
            assert config.controls.input.jailbreak.enabled is False
            assert config.controls.input.word_filter.enabled is False
            assert config.controls.output.pii_phi.enabled is False
            assert config.controls.output_block_action == "block"
            assert config.sampling.l1_rate == 1.0
            assert config.sampling.l2_rate == 0.0
            assert config.attestation.offline is True
        except ImportError:
            pytest.skip("pyyaml not installed")

    def test_load_config_v13_nested(self):
        """Loading v1.3 config with nested input/output stages."""
        try:
            import yaml  # noqa: F401

            from glacis.config import load_config

            config_content = """
version: "1.3"
policy:
  id: "hipaa-safe-harbor"
  environment: "production"
  tags: ["healthcare", "hipaa"]
controls:
  output_block_action: "forward"
  input:
    pii_phi:
      enabled: true
      model: presidio
      mode: fast
      entities: ["US_SSN", "EMAIL_ADDRESS"]
      if_detected: "flag"
    word_filter:
      enabled: true
      entities: ["confidential", "proprietary"]
      if_detected: "flag"
    jailbreak:
      enabled: true
      model: prompt_guard_22m
      threshold: 0.5
      if_detected: block
  output:
    pii_phi:
      enabled: true
    word_filter:
      enabled: true
      entities: ["system prompt", "secret"]
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

                # Input stage
                assert config.controls.input.pii_phi.enabled is True
                assert config.controls.input.pii_phi.model == "presidio"
                assert config.controls.input.pii_phi.mode == "fast"
                assert config.controls.input.pii_phi.entities == ["US_SSN", "EMAIL_ADDRESS"]
                assert config.controls.input.pii_phi.if_detected == "flag"

                assert config.controls.input.word_filter.enabled is True
                assert config.controls.input.word_filter.entities == ["confidential", "proprietary"]

                assert config.controls.input.jailbreak.enabled is True
                assert config.controls.input.jailbreak.model == "prompt_guard_22m"
                assert config.controls.input.jailbreak.threshold == 0.5
                assert config.controls.input.jailbreak.if_detected == "block"

                # Output stage
                assert config.controls.output.pii_phi.enabled is True

                assert config.controls.output.word_filter.enabled is True
                assert config.controls.output.word_filter.entities == ["system prompt", "secret"]

                # Top-level
                assert config.controls.output_block_action == "forward"
                assert config.attestation.offline is False
                assert config.attestation.service_id == "custom-service"
                assert config.policy.id == "hipaa-safe-harbor"
                assert config.policy.environment == "production"
                assert config.policy.tags == ["healthcare", "hipaa"]
            finally:
                Path(temp_path).unlink()

        except ImportError:
            pytest.skip("pyyaml not installed")


class TestPiiPhiControlConfig:
    """Tests for PiiPhiControlConfig model."""

    def test_defaults(self):
        from glacis.config import PiiPhiControlConfig

        config = PiiPhiControlConfig()
        assert config.enabled is False
        assert config.model == "presidio"
        assert config.mode == "fast"
        assert config.entities == []
        assert config.if_detected == "flag"

    def test_custom_settings(self):
        from glacis.config import PiiPhiControlConfig

        config = PiiPhiControlConfig(
            enabled=True,
            model="presidio",
            mode="full",
            entities=["US_SSN"],
            if_detected="block",
        )
        assert config.enabled is True
        assert config.mode == "full"
        assert config.entities == ["US_SSN"]
        assert config.if_detected == "block"


class TestWordFilterControlConfig:
    """Tests for WordFilterControlConfig model."""

    def test_defaults(self):
        from glacis.config import WordFilterControlConfig

        config = WordFilterControlConfig()
        assert config.enabled is False
        assert config.entities == []
        assert config.if_detected == "flag"

    def test_custom_settings(self):
        from glacis.config import WordFilterControlConfig

        config = WordFilterControlConfig(
            enabled=True,
            entities=["secret", "confidential"],
            if_detected="block",
        )
        assert config.entities == ["secret", "confidential"]
        assert config.if_detected == "block"


class TestJailbreakControlConfig:
    """Tests for JailbreakControlConfig model."""

    def test_defaults(self):
        from glacis.config import JailbreakControlConfig

        config = JailbreakControlConfig()
        assert config.enabled is False
        assert config.model == "prompt_guard_22m"
        assert config.threshold == 0.5
        assert config.if_detected == "flag"

    def test_custom_settings(self):
        from glacis.config import JailbreakControlConfig

        config = JailbreakControlConfig(
            enabled=True,
            model="prompt_guard_86m",
            threshold=0.8,
            if_detected="block",
        )
        assert config.model == "prompt_guard_86m"
        assert config.threshold == 0.8
        assert config.if_detected == "block"

    def test_threshold_bounds(self):
        from pydantic import ValidationError

        from glacis.config import JailbreakControlConfig

        with pytest.raises(ValidationError):
            JailbreakControlConfig(threshold=1.5)

        with pytest.raises(ValidationError):
            JailbreakControlConfig(threshold=-0.1)


class TestSamplingConfig:
    """Tests for SamplingConfig model."""

    def test_defaults(self):
        from glacis.config import SamplingConfig

        config = SamplingConfig()
        assert config.l1_rate == 1.0
        assert config.l2_rate == 0.0

    def test_custom_rates(self):
        from glacis.config import SamplingConfig

        config = SamplingConfig(l1_rate=0.3, l2_rate=0.05)
        assert config.l1_rate == 0.3
        assert config.l2_rate == 0.05

    def test_bounds_validation(self):
        from pydantic import ValidationError

        from glacis.config import SamplingConfig

        with pytest.raises(ValidationError):
            SamplingConfig(l1_rate=1.5)

        with pytest.raises(ValidationError):
            SamplingConfig(l1_rate=-0.1)

        with pytest.raises(ValidationError):
            SamplingConfig(l2_rate=2.0)

    def test_glacis_config_has_sampling(self):
        from glacis.config import GlacisConfig

        config = GlacisConfig()
        assert config.sampling.l1_rate == 1.0
        assert config.sampling.l2_rate == 0.0

    def test_sampling_from_yaml(self):
        try:
            import yaml  # noqa: F401

            from glacis.config import load_config

            config_content = """
sampling:
  l1_rate: 0.3
  l2_rate: 0.05
"""
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write(config_content)
                f.flush()
                temp_path = f.name

            try:
                config = load_config(temp_path)
                assert config.sampling.l1_rate == 0.3
                assert config.sampling.l2_rate == 0.05
            finally:
                Path(temp_path).unlink()

        except ImportError:
            pytest.skip("pyyaml not installed")


class TestControlsConfig:
    """Tests for top-level ControlsConfig."""

    def test_output_block_action_default(self):
        from glacis.config import ControlsConfig

        config = ControlsConfig()
        assert config.output_block_action == "block"

    def test_output_block_action_forward(self):
        from glacis.config import ControlsConfig

        config = ControlsConfig(output_block_action="forward")
        assert config.output_block_action == "forward"

    def test_partial_input_only(self):
        """Config with only input stage, output defaults."""
        try:
            import yaml  # noqa: F401

            from glacis.config import load_config

            config_content = """
controls:
  input:
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

                assert config.controls.input.pii_phi.enabled is True
                # Defaults preserved
                assert config.controls.input.pii_phi.model == "presidio"
                assert config.controls.input.jailbreak.enabled is False
                assert config.controls.output.pii_phi.enabled is False
            finally:
                Path(temp_path).unlink()

        except ImportError:
            pytest.skip("pyyaml not installed")

    def test_invalid_yaml(self):
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
                with pytest.raises(Exception):
                    load_config(temp_path)
            finally:
                Path(temp_path).unlink()

        except ImportError:
            pytest.skip("pyyaml not installed")
