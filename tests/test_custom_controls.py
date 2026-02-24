"""
Tests for custom control interface (glacis/controls/base.py).

Verifies that custom controls can be written, injected into the pipeline,
and work correctly with the ControlsRunner.
"""

import pytest

try:
    from glacis.config import InputControlsConfig, OutputControlsConfig
    from glacis.controls import ControlsRunner, StageResult
    from glacis.controls.base import BaseControl, ControlResult

    CONTROLS_AVAILABLE = True
except ImportError:
    CONTROLS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CONTROLS_AVAILABLE, reason="controls extra not installed")


# --- Test custom control implementations ---

class SimpleScanningControl(BaseControl):
    """Minimal scanning control (3-line check)."""
    control_type = "content_safety"

    def check(self, text: str) -> ControlResult:
        detected = "bad" in text.lower()
        return ControlResult(
            control_type=self.control_type,
            detected=detected,
            action="flag" if detected else "forward",
            categories=["toxicity"] if detected else [],
        )


class ErrorControl(BaseControl):
    """Control that always raises an exception."""
    control_type = "custom"

    def check(self, text: str) -> ControlResult:
        raise RuntimeError("Model loading failed")


class BlockingControl(BaseControl):
    """Control that always blocks."""
    control_type = "content_safety"

    def check(self, text: str) -> ControlResult:
        return ControlResult(
            control_type=self.control_type,
            detected=True,
            action="block",
            score=0.99,
            categories=["toxicity"],
        )


# --- Tests ---

class TestCustomScanningControl:
    """Tests for custom scanning controls."""

    def test_minimal_scanning_control(self):
        """3-line custom scanning control works."""
        control = SimpleScanningControl()

        result = control.check("This has bad content")
        assert result.detected is True
        assert result.action == "flag"
        assert "toxicity" in result.categories

        result = control.check("This is fine")
        assert result.detected is False
        assert result.action == "forward"

    def test_scanning_control_in_pipeline(self):
        """Custom scanning control works in ControlsRunner."""
        runner = ControlsRunner(input_controls=[SimpleScanningControl()])

        result = runner.run_input("This has bad words")
        assert any(r.control_type == "content_safety" for r in result.results)
        assert any(r.detected for r in result.results)

        runner.close()


class TestCustomControlErrors:
    """Tests for error handling in custom controls."""

    def test_error_control_returns_error_result(self):
        """Control that throws produces action='error' result."""
        runner = ControlsRunner(input_controls=[ErrorControl()])

        result = runner.run_input("test text")
        assert len(result.results) == 1
        assert result.results[0].action == "error"
        assert "Model loading failed" in result.results[0].metadata.get("error", "")

    def test_error_does_not_crash_pipeline(self):
        """Error in one control doesn't prevent others from running."""
        runner = ControlsRunner(
            input_controls=[ErrorControl(), SimpleScanningControl()],
        )

        result = runner.run_input("bad content")
        assert len(result.results) == 2

        # One error, one detection
        actions = {r.action for r in result.results}
        assert "error" in actions

        runner.close()


class TestCustomControlBlocking:
    """Tests for blocking behavior with custom controls."""

    def test_blocking_control_sets_should_block(self):
        """Custom blocking control sets should_block=True."""
        runner = ControlsRunner(input_controls=[BlockingControl()])

        result = runner.run_input("any content")
        assert result.should_block is True

        runner.close()

    def test_blocking_control_in_output_stage(self):
        """Custom blocking control works in output stage."""
        runner = ControlsRunner(output_controls=[BlockingControl()])

        result = runner.run_output("response text")
        assert result.should_block is True

        runner.close()


class TestCustomControlPerStage:
    """Tests for per-stage custom control registration."""

    def test_input_only_controls(self):
        """Controls registered for input only don't affect output."""
        scanner = SimpleScanningControl()
        runner = ControlsRunner(input_controls=[scanner])

        assert runner.has_input_controls is True
        assert runner.has_output_controls is False

        # Input stage has the control
        input_result = runner.run_input("bad content")
        assert len(input_result.results) > 0

        # Output stage has no controls
        output_result = runner.run_output("bad content")
        assert len(output_result.results) == 0

        runner.close()

    def test_output_only_controls(self):
        """Controls registered for output only don't affect input."""
        scanner = SimpleScanningControl()
        runner = ControlsRunner(output_controls=[scanner])

        assert runner.has_input_controls is False
        assert runner.has_output_controls is True

        input_result = runner.run_input("bad content")
        assert len(input_result.results) == 0

        output_result = runner.run_output("bad content")
        assert len(output_result.results) > 0

        runner.close()

    def test_same_control_both_stages(self):
        """Same control in both stages works."""
        scanner = SimpleScanningControl()
        runner = ControlsRunner(
            input_controls=[scanner],
            output_controls=[scanner],
        )

        assert runner.has_input_controls is True
        assert runner.has_output_controls is True

        input_result = runner.run_input("bad")
        output_result = runner.run_output("bad")

        assert len(input_result.results) > 0
        assert len(output_result.results) > 0

        runner.close()


class TestCustomControlTypeMapping:
    """Tests for control type mapping in the accumulator."""

    def test_known_type_passes_through(self):
        """Known control types (e.g., 'content_safety') pass through."""
        from glacis.integrations.base import _map_control_type

        assert _map_control_type("pii") == "pii"
        assert _map_control_type("jailbreak") == "jailbreak"
        assert _map_control_type("content_safety") == "content_safety"
        assert _map_control_type("word_filter") == "word_filter"
        assert _map_control_type("custom") == "custom"

    def test_unknown_type_maps_to_custom(self):
        """Unknown control types map to 'custom'."""
        from glacis.integrations.base import _map_control_type

        assert _map_control_type("toxicity_guard") == "custom"
        assert _map_control_type("my_special_control") == "custom"
