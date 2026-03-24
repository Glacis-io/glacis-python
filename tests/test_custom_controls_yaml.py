"""
Tests for YAML-based custom control registration.

Tests the full flow: glacis.yaml → env var substitution → dynamic import →
control instantiation → pipeline integration → attestation.
"""

import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from glacis.config import (
    CustomControlEntry,
    GlacisConfig,
    _substitute_env_vars,
    load_config,
)
from glacis.controls import _load_custom_control
from glacis.controls.base import BaseControl, ControlResult


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_receipt_state():
    """Reset context-var receipt state between tests."""
    from glacis.integrations.base import _last_receipt_var

    token = _last_receipt_var.set(None)
    yield
    _last_receipt_var.reset(token)


@pytest.fixture(autouse=True)
def _temp_home(tmp_path, monkeypatch):
    """Redirect evidence storage to temp directory."""
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture(autouse=True)
def _clean_dynamic_modules():
    """Remove dynamically imported test modules from sys.modules between tests.

    Without this, importlib caches the first tmp_path's module and silently
    reuses it even when a different tmp_path writes a different file.
    """
    yield
    for mod_name in list(sys.modules):
        if mod_name in ("my_control", "strict_ctrl", "broken_ctrl", "env_ctrl"):
            del sys.modules[mod_name]


def _write_control_module(directory: Path, filename: str = "my_control.py") -> None:
    """Write a simple custom control module to the given directory."""
    code = textwrap.dedent("""\
        from glacis.controls.base import BaseControl, ControlResult

        class SimpleControl(BaseControl):
            control_type = "custom"

            def __init__(self, if_detected="flag", threshold=0.5, **kwargs):
                self._action = if_detected
                self._threshold = threshold
                self._extra = kwargs

            def check(self, text):
                detected = "bad" in text.lower()
                return ControlResult(
                    control_type=self.control_type,
                    detected=detected,
                    action=self._action if detected else "forward",
                    score=1.0 if detected else 0.0,
                    latency_ms=1,
                    metadata={"threshold": self._threshold},
                )

        class GroundingControl(BaseControl):
            control_type = "grounding"

            def __init__(self, if_detected="flag", **kwargs):
                self._action = if_detected

            def check(self, text):
                return ControlResult(
                    control_type=self.control_type,
                    detected=False,
                    action="forward",
                    latency_ms=1,
                )

        class NotAControl:
            pass
    """)
    (directory / filename).write_text(code)


def _write_yaml(directory: Path, yaml_content: str) -> Path:
    """Write a glacis.yaml to the given directory and return its path."""
    yaml_path = directory / "glacis.yaml"
    yaml_path.write_text(textwrap.dedent(yaml_content))
    return yaml_path


# ─── Environment Variable Substitution ───────────────────────────────────────


class TestEnvVarSubstitution:
    """Tests for _substitute_env_vars()."""

    def test_simple_substitution(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret123")
        assert _substitute_env_vars("${MY_KEY}") == "secret123"

    def test_substitution_in_string(self, monkeypatch):
        monkeypatch.setenv("HOST", "example.com")
        assert _substitute_env_vars("https://${HOST}/api") == "https://example.com/api"

    def test_nested_dict(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "sk-test")
        data = {"args": {"api_key": "${API_KEY}", "threshold": 0.5}}
        result = _substitute_env_vars(data)
        assert result["args"]["api_key"] == "sk-test"
        assert result["args"]["threshold"] == 0.5

    def test_list_substitution(self, monkeypatch):
        monkeypatch.setenv("VAL", "replaced")
        data = ["${VAL}", "literal", 42]
        result = _substitute_env_vars(data)
        assert result == ["replaced", "literal", 42]

    def test_missing_env_var_raises(self):
        with pytest.raises(ValueError, match="Environment variable 'NONEXISTENT_VAR' is not set"):
            _substitute_env_vars("${NONEXISTENT_VAR}")

    def test_non_string_passthrough(self):
        assert _substitute_env_vars(42) == 42
        assert _substitute_env_vars(True) is True
        assert _substitute_env_vars(None) is None
        assert _substitute_env_vars(3.14) == 3.14


# ─── Dynamic Control Loading ────────────────────────────────────────────────


class TestLoadCustomControl:
    """Tests for _load_custom_control()."""

    def test_load_from_path(self, tmp_path):
        """Loads a control class from a module in config_dir."""
        _write_control_module(tmp_path)
        entry = CustomControlEntry(path="my_control.SimpleControl")
        ctrl = _load_custom_control(entry, config_dir=str(tmp_path))
        assert ctrl.control_type == "custom"
        assert isinstance(ctrl, BaseControl)

    def test_args_passed_to_constructor(self, tmp_path):
        """Constructor kwargs from YAML args are passed through."""
        _write_control_module(tmp_path)
        entry = CustomControlEntry(
            path="my_control.SimpleControl",
            args={"threshold": 0.9},
        )
        ctrl = _load_custom_control(entry, config_dir=str(tmp_path))
        assert ctrl._threshold == 0.9

    def test_if_detected_passed_to_constructor(self, tmp_path):
        """if_detected from YAML entry is available as a kwarg."""
        _write_control_module(tmp_path)
        entry = CustomControlEntry(
            path="my_control.SimpleControl",
            if_detected="block",
        )
        ctrl = _load_custom_control(entry, config_dir=str(tmp_path))
        assert ctrl._action == "block"

    def test_control_type_preserved(self, tmp_path):
        """Control's own control_type attribute is used, not forced to 'custom'."""
        _write_control_module(tmp_path)
        entry = CustomControlEntry(path="my_control.GroundingControl")
        ctrl = _load_custom_control(entry, config_dir=str(tmp_path))
        assert ctrl.control_type == "grounding"

    def test_invalid_path_format_raises(self, tmp_path):
        """Path without a dot raises ImportError."""
        entry = CustomControlEntry(path="NoDotsHere")
        with pytest.raises(ImportError, match="Invalid control path"):
            _load_custom_control(entry, config_dir=str(tmp_path))

    def test_module_not_found_raises(self, tmp_path):
        """Non-existent module raises ImportError with hint."""
        entry = CustomControlEntry(path="nonexistent_module.MyControl")
        with pytest.raises(ImportError, match="Cannot import module 'nonexistent_module'"):
            _load_custom_control(entry, config_dir=str(tmp_path))

    def test_class_not_found_raises(self, tmp_path):
        """Wrong class name raises AttributeError listing available controls."""
        _write_control_module(tmp_path)
        entry = CustomControlEntry(path="my_control.DoesNotExist")
        with pytest.raises(AttributeError, match="has no class 'DoesNotExist'"):
            _load_custom_control(entry, config_dir=str(tmp_path))

    def test_class_not_found_shows_available(self, tmp_path):
        """Error message lists available BaseControl subclasses."""
        _write_control_module(tmp_path)
        entry = CustomControlEntry(path="my_control.DoesNotExist")
        with pytest.raises(AttributeError, match="SimpleControl"):
            _load_custom_control(entry, config_dir=str(tmp_path))

    def test_not_base_control_raises(self, tmp_path):
        """Class that doesn't extend BaseControl raises TypeError."""
        _write_control_module(tmp_path)
        entry = CustomControlEntry(path="my_control.NotAControl")
        with pytest.raises(TypeError, match="is not a BaseControl subclass"):
            _load_custom_control(entry, config_dir=str(tmp_path))

    def test_bad_constructor_args_raises(self, tmp_path):
        """Wrong constructor args raise TypeError with details."""
        code = textwrap.dedent("""\
            from glacis.controls.base import BaseControl, ControlResult

            class StrictControl(BaseControl):
                control_type = "custom"
                def __init__(self, required_arg):
                    self._val = required_arg
                def check(self, text):
                    return ControlResult(control_type=self.control_type, detected=False)
        """)
        (tmp_path / "strict_ctrl.py").write_text(code)
        entry = CustomControlEntry(
            path="strict_ctrl.StrictControl",
            args={},  # Missing required_arg
        )
        with pytest.raises(TypeError, match="Failed to instantiate"):
            _load_custom_control(entry, config_dir=str(tmp_path))

    def test_config_dir_added_to_sys_path(self, tmp_path):
        """YAML file's directory is added to sys.path for import resolution."""
        import sys

        _write_control_module(tmp_path)
        config_dir = str(tmp_path)

        # Remove it if it happens to be there
        if config_dir in sys.path:
            sys.path.remove(config_dir)

        entry = CustomControlEntry(path="my_control.SimpleControl")
        _load_custom_control(entry, config_dir=config_dir)
        assert config_dir in sys.path


# ─── Config Loading with Custom Controls ─────────────────────────────────────


class TestConfigLoading:
    """Tests for load_config() with custom control entries."""

    def test_yaml_with_custom_controls(self, tmp_path, monkeypatch):
        """Custom controls are parsed from YAML into CustomControlEntry models."""
        _write_yaml(tmp_path, """\
            version: "1.3"
            controls:
              output:
                custom:
                  - path: "my_control.SimpleControl"
                    enabled: true
                    if_detected: "flag"
                    args:
                      threshold: 0.8
        """)
        cfg = load_config(str(tmp_path / "glacis.yaml"))
        assert len(cfg.controls.output.custom) == 1
        entry = cfg.controls.output.custom[0]
        assert entry.path == "my_control.SimpleControl"
        assert entry.enabled is True
        assert entry.if_detected == "flag"
        assert entry.args["threshold"] == 0.8

    def test_yaml_env_var_substitution(self, tmp_path, monkeypatch):
        """${ENV_VAR} in YAML is replaced before Pydantic validation."""
        monkeypatch.setenv("TEST_API_KEY", "sk-test-123")
        _write_yaml(tmp_path, """\
            version: "1.3"
            controls:
              output:
                custom:
                  - path: "my_control.SimpleControl"
                    args:
                      api_key: "${TEST_API_KEY}"
        """)
        cfg = load_config(str(tmp_path / "glacis.yaml"))
        assert cfg.controls.output.custom[0].args["api_key"] == "sk-test-123"

    def test_config_dir_set(self, tmp_path):
        """load_config sets _config_dir to the YAML file's parent directory."""
        _write_yaml(tmp_path, "version: '1.3'\n")
        cfg = load_config(str(tmp_path / "glacis.yaml"))
        assert cfg._config_dir == str(tmp_path.resolve())

    def test_disabled_custom_controls_in_yaml(self, tmp_path):
        """Disabled entries are parsed but not loaded by ControlsRunner."""
        _write_yaml(tmp_path, """\
            version: "1.3"
            controls:
              output:
                custom:
                  - path: "my_control.SimpleControl"
                    enabled: false
        """)
        cfg = load_config(str(tmp_path / "glacis.yaml"))
        assert len(cfg.controls.output.custom) == 1
        assert cfg.controls.output.custom[0].enabled is False

    def test_multiple_custom_controls(self, tmp_path):
        """Multiple custom controls can be listed."""
        _write_yaml(tmp_path, """\
            version: "1.3"
            controls:
              input:
                custom:
                  - path: "mod_a.CtrlA"
                    args:
                      x: 1
                  - path: "mod_b.CtrlB"
                    enabled: false
              output:
                custom:
                  - path: "mod_c.CtrlC"
                    if_detected: "block"
        """)
        cfg = load_config(str(tmp_path / "glacis.yaml"))
        assert len(cfg.controls.input.custom) == 2
        assert len(cfg.controls.output.custom) == 1


# ─── ControlsRunner Integration ──────────────────────────────────────────────


class TestControlsRunnerYAML:
    """Tests for ControlsRunner loading YAML custom controls."""

    def test_runner_loads_enabled_custom_controls(self, tmp_path):
        """Enabled YAML custom controls are loaded into the runner."""
        _write_control_module(tmp_path)
        _write_yaml(tmp_path, """\
            version: "1.3"
            controls:
              output:
                custom:
                  - path: "my_control.SimpleControl"
                    enabled: true
                    if_detected: "flag"
        """)
        cfg = load_config(str(tmp_path / "glacis.yaml"))

        from glacis.controls import ControlsRunner

        runner = ControlsRunner(
            output_config=cfg.controls.output,
            config_dir=cfg._config_dir,
        )
        assert runner.has_output_controls
        result = runner.run_output("this is bad content")
        assert len(result.results) == 1
        assert result.results[0].detected is True

    def test_runner_skips_disabled_custom_controls(self, tmp_path):
        """Disabled YAML custom controls are not loaded."""
        _write_control_module(tmp_path)
        _write_yaml(tmp_path, """\
            version: "1.3"
            controls:
              output:
                custom:
                  - path: "my_control.SimpleControl"
                    enabled: false
        """)
        cfg = load_config(str(tmp_path / "glacis.yaml"))

        from glacis.controls import ControlsRunner

        runner = ControlsRunner(
            output_config=cfg.controls.output,
            config_dir=cfg._config_dir,
        )
        assert not runner.has_output_controls

    def test_runner_mixes_builtin_and_custom(self, tmp_path):
        """Built-in and custom controls coexist in the same stage."""
        _write_control_module(tmp_path)
        _write_yaml(tmp_path, """\
            version: "1.3"
            controls:
              output:
                word_filter:
                  enabled: true
                  entities: ["forbidden"]
                  if_detected: "flag"
                custom:
                  - path: "my_control.SimpleControl"
                    enabled: true
        """)
        cfg = load_config(str(tmp_path / "glacis.yaml"))

        from glacis.controls import ControlsRunner

        runner = ControlsRunner(
            output_config=cfg.controls.output,
            config_dir=cfg._config_dir,
        )
        assert runner.has_output_controls
        result = runner.run_output("this is clean text")
        # Both controls should have run
        assert len(result.results) == 2


# ─── End-to-End Pipeline Integration ─────────────────────────────────────────


class TestE2EYAMLCustomControl:
    """Full pipeline: YAML config → custom control → attestation."""

    def test_e2e_yaml_custom_control_in_attestation(self, tmp_path, signing_seed):
        """Custom control from YAML runs and results appear in attestation evidence."""
        pytest.importorskip("openai")

        _write_control_module(tmp_path)
        _write_yaml(tmp_path, """\
            version: "1.3"
            controls:
              output:
                custom:
                  - path: "my_control.SimpleControl"
                    enabled: true
                    if_detected: "flag"
                    args:
                      threshold: 0.75
        """)

        # Build mock OpenAI response
        msg = MagicMock()
        msg.role = "assistant"
        msg.content = "This is a bad response"  # triggers SimpleControl

        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "stop"

        usage = MagicMock()
        usage.prompt_tokens = 9
        usage.completion_tokens = 12
        usage.total_tokens = 21

        resp = MagicMock()
        resp.model = "gpt-4"
        resp.choices = [choice]
        resp.usage = usage

        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = resp
            mock_cls.return_value = mock_client

            from glacis.integrations.openai import attested_openai

            client = attested_openai(
                openai_api_key="sk-test",
                offline=True,
                signing_seed=signing_seed,
                config=str(tmp_path / "glacis.yaml"),
            )
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        from glacis.integrations.base import get_evidence, get_last_receipt

        receipt = get_last_receipt()
        assert receipt is not None

        evidence = get_evidence(receipt.id)
        assert evidence is not None

        cp = evidence.get("control_plane_results", {})
        assert cp is not None
        # Controls are in cp["controls"] as ControlExecution dicts
        controls = cp.get("controls", [])
        # Should have our custom control result with type="custom" and stage="output"
        custom_results = [
            c for c in controls
            if c.get("type") == "custom" and c.get("stage") == "output"
        ]
        assert len(custom_results) >= 1
        assert custom_results[0]["status"] == "flag"

    def test_e2e_env_var_substitution_reaches_control(self, tmp_path, signing_seed, monkeypatch):
        """${ENV_VAR} in YAML args is substituted and reaches the control constructor."""
        pytest.importorskip("openai")

        monkeypatch.setenv("CUSTOM_THRESHOLD", "0.42")

        # Write a control that stores threshold so we can verify
        code = textwrap.dedent("""\
            from glacis.controls.base import BaseControl, ControlResult

            class EnvVarControl(BaseControl):
                control_type = "custom"
                received_threshold = None  # class-level for inspection

                def __init__(self, if_detected="flag", threshold="default", **kwargs):
                    self._action = if_detected
                    EnvVarControl.received_threshold = threshold

                def check(self, text):
                    return ControlResult(
                        control_type=self.control_type,
                        detected=False,
                        action="forward",
                        latency_ms=1,
                        metadata={"threshold": EnvVarControl.received_threshold},
                    )
        """)
        (tmp_path / "env_ctrl.py").write_text(code)

        _write_yaml(tmp_path, """\
            version: "1.3"
            controls:
              output:
                custom:
                  - path: "env_ctrl.EnvVarControl"
                    enabled: true
                    args:
                      threshold: "${CUSTOM_THRESHOLD}"
        """)

        msg = MagicMock()
        msg.role = "assistant"
        msg.content = "clean response"
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "stop"
        usage = MagicMock()
        usage.prompt_tokens = 5
        usage.completion_tokens = 5
        usage.total_tokens = 10
        resp = MagicMock()
        resp.model = "gpt-4"
        resp.choices = [choice]
        resp.usage = usage

        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = resp
            mock_cls.return_value = mock_client

            from glacis.integrations.openai import attested_openai

            client = attested_openai(
                openai_api_key="sk-test",
                offline=True,
                signing_seed=signing_seed,
                config=str(tmp_path / "glacis.yaml"),
            )
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # Verify the env var was substituted and reached the constructor
        import importlib
        env_ctrl = importlib.import_module("env_ctrl")
        assert env_ctrl.EnvVarControl.received_threshold == "0.42"

        # Also verify it shows up in the attestation evidence
        from glacis.integrations.base import get_evidence, get_last_receipt

        receipt = get_last_receipt()
        evidence = get_evidence(receipt.id)
        cp = evidence.get("control_plane_results", {})
        controls = cp.get("controls", [])
        custom_results = [
            c for c in controls
            if c.get("type") == "custom" and c.get("stage") == "output"
        ]
        assert len(custom_results) >= 1
        assert custom_results[0]["status"] == "forward"

    def test_e2e_input_stage_custom_control(self, tmp_path, signing_seed):
        """Custom control on the input stage runs and appears in attestation."""
        pytest.importorskip("openai")

        _write_control_module(tmp_path)
        _write_yaml(tmp_path, """\
            version: "1.3"
            controls:
              input:
                custom:
                  - path: "my_control.SimpleControl"
                    enabled: true
                    if_detected: "flag"
        """)

        msg = MagicMock()
        msg.role = "assistant"
        msg.content = "clean response"
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "stop"
        usage = MagicMock()
        usage.prompt_tokens = 5
        usage.completion_tokens = 5
        usage.total_tokens = 10
        resp = MagicMock()
        resp.model = "gpt-4"
        resp.choices = [choice]
        resp.usage = usage

        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = resp
            mock_cls.return_value = mock_client

            from glacis.integrations.openai import attested_openai

            client = attested_openai(
                openai_api_key="sk-test",
                offline=True,
                signing_seed=signing_seed,
                config=str(tmp_path / "glacis.yaml"),
            )
            # "bad" in the user message triggers SimpleControl on input stage
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "This is bad input"}],
            )

        from glacis.integrations.base import get_evidence, get_last_receipt

        receipt = get_last_receipt()
        evidence = get_evidence(receipt.id)
        cp = evidence.get("control_plane_results", {})
        controls = cp.get("controls", [])
        input_customs = [
            c for c in controls
            if c.get("type") == "custom" and c.get("stage") == "input"
        ]
        assert len(input_customs) >= 1
        assert input_customs[0]["status"] == "flag"

    def test_e2e_multiple_custom_controls_same_stage(self, tmp_path, signing_seed):
        """Multiple enabled custom controls in the same stage all run."""
        pytest.importorskip("openai")

        _write_control_module(tmp_path)  # has SimpleControl + GroundingControl
        _write_yaml(tmp_path, """\
            version: "1.3"
            controls:
              output:
                custom:
                  - path: "my_control.SimpleControl"
                    enabled: true
                    if_detected: "flag"
                  - path: "my_control.GroundingControl"
                    enabled: true
                    if_detected: "flag"
        """)

        msg = MagicMock()
        msg.role = "assistant"
        msg.content = "This is a bad response"
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "stop"
        usage = MagicMock()
        usage.prompt_tokens = 5
        usage.completion_tokens = 5
        usage.total_tokens = 10
        resp = MagicMock()
        resp.model = "gpt-4"
        resp.choices = [choice]
        resp.usage = usage

        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = resp
            mock_cls.return_value = mock_client

            from glacis.integrations.openai import attested_openai

            client = attested_openai(
                openai_api_key="sk-test",
                offline=True,
                signing_seed=signing_seed,
                config=str(tmp_path / "glacis.yaml"),
            )
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        from glacis.integrations.base import get_evidence, get_last_receipt

        receipt = get_last_receipt()
        evidence = get_evidence(receipt.id)
        cp = evidence.get("control_plane_results", {})
        controls = cp.get("controls", [])
        output_customs = [
            c for c in controls if c.get("stage") == "output"
        ]
        # Should have both SimpleControl (type="custom") and GroundingControl (type="grounding")
        types = {c.get("type") for c in output_customs}
        assert "custom" in types
        assert "grounding" in types
        assert len(output_customs) >= 2


# ─── Additional Coverage ─────────────────────────────────────────────────────


class TestControlsRunnerErrorHandling:
    """Tests for runner behavior when custom controls raise exceptions."""

    def test_check_exception_returns_error_result(self, tmp_path):
        """A custom control whose check() raises returns action='error', not a crash."""
        code = textwrap.dedent("""\
            from glacis.controls.base import BaseControl, ControlResult

            class BrokenControl(BaseControl):
                control_type = "custom"
                def __init__(self, if_detected="flag", **kwargs):
                    pass
                def check(self, text):
                    raise RuntimeError("something went wrong")
        """)
        (tmp_path / "broken_ctrl.py").write_text(code)
        _write_yaml(tmp_path, """\
            version: "1.3"
            controls:
              output:
                custom:
                  - path: "broken_ctrl.BrokenControl"
                    enabled: true
        """)
        cfg = load_config(str(tmp_path / "glacis.yaml"))

        from glacis.controls import ControlsRunner

        runner = ControlsRunner(
            output_config=cfg.controls.output,
            config_dir=cfg._config_dir,
        )
        result = runner.run_output("any text")
        assert len(result.results) == 1
        assert result.results[0].action == "error"
        assert "something went wrong" in result.results[0].metadata.get("error", "")

    def test_runner_wraps_load_failure_with_clear_message(self, tmp_path):
        """ControlsRunner wraps load failures in RuntimeError with path context."""
        _write_yaml(tmp_path, """\
            version: "1.3"
            controls:
              output:
                custom:
                  - path: "nonexistent_module.FakeControl"
                    enabled: true
        """)
        cfg = load_config(str(tmp_path / "glacis.yaml"))

        from glacis.controls import ControlsRunner

        with pytest.raises(RuntimeError, match="nonexistent_module.FakeControl"):
            ControlsRunner(
                output_config=cfg.controls.output,
                config_dir=cfg._config_dir,
            )

    def test_if_detected_from_yaml_wins_over_args(self, tmp_path):
        """Top-level if_detected always overrides if_detected in args."""
        _write_control_module(tmp_path)
        entry = CustomControlEntry(
            path="my_control.SimpleControl",
            if_detected="block",
            args={"if_detected": "flag"},  # should be overridden
        )
        ctrl = _load_custom_control(entry, config_dir=str(tmp_path))
        assert ctrl._action == "block"

    def test_control_with_no_args_in_yaml(self, tmp_path):
        """Custom control with no args key in YAML still receives if_detected."""
        _write_control_module(tmp_path)
        _write_yaml(tmp_path, """\
            version: "1.3"
            controls:
              output:
                custom:
                  - path: "my_control.SimpleControl"
                    if_detected: "block"
        """)
        cfg = load_config(str(tmp_path / "glacis.yaml"))
        entry = cfg.controls.output.custom[0]
        # args should default to empty dict, if_detected should be "block"
        assert entry.args == {}
        assert entry.if_detected == "block"

        ctrl = _load_custom_control(entry, config_dir=cfg._config_dir)
        assert ctrl._action == "block"

    def test_multiple_env_vars_in_single_string(self, monkeypatch):
        """Multiple ${VAR} references in one string are all substituted."""
        monkeypatch.setenv("HOST", "api.example.com")
        monkeypatch.setenv("PORT", "8443")
        result = _substitute_env_vars("https://${HOST}:${PORT}/v1")
        assert result == "https://api.example.com:8443/v1"
