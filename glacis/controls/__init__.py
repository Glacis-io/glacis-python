"""
GLACIS Controls Module.

Provides modular controls for PII/PHI detection, jailbreak detection,
word filtering, and custom controls. Controls are organized into a staged
pipeline (input/output) where all controls run in parallel via
``ThreadPoolExecutor``.

Pipeline stages:
- **Input stage**: All controls run in parallel before the LLM call
- **Output stage**: All controls run in parallel after the LLM call

Example (using individual controls):
    >>> from glacis.controls import PIIControl, JailbreakControl, WordFilterControl
    >>> from glacis.config import PiiPhiControlConfig, JailbreakControlConfig
    >>>
    >>> # PII detection
    >>> pii = PIIControl(PiiPhiControlConfig(enabled=True, mode="fast"))
    >>> result = pii.check("SSN: 123-45-6789")
    >>> result.detected
    True

Example (using ControlsRunner):
    >>> from glacis.controls import ControlsRunner
    >>> from glacis.config import load_config
    >>>
    >>> cfg = load_config()  # Loads glacis.yaml
    >>> runner = ControlsRunner(input_config=cfg.controls.input,
    ...                         output_config=cfg.controls.output)
    >>> stage_result = runner.run_input("Patient SSN: 123-45-6789")
    >>> stage_result.should_block    # True if any control wants to block
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from glacis.controls.base import BaseControl, ControlAction, ControlResult
from glacis.controls.jailbreak import JailbreakControl
from glacis.controls.pii import PIIControl
from glacis.controls.word_filter import WordFilterControl

if TYPE_CHECKING:
    from glacis.config import InputControlsConfig, OutputControlsConfig


@dataclass
class StageResult:
    """Result from running a pipeline stage (input or output).

    Attributes:
        results: All control results from this stage.
        effective_text: The text (always equals the original input).
        should_block: True if any control in this stage returned action="block".
    """

    results: list[ControlResult] = field(default_factory=list)
    effective_text: str = ""
    should_block: bool = False


class ControlsRunner:
    """
    Orchestrates a staged control pipeline.

    All controls run in parallel via ``ThreadPoolExecutor``.
    Single control = direct call (no thread overhead).

    Args:
        input_config: Config for input stage controls.
        output_config: Config for output stage controls.
        input_controls: Custom controls to add to input stage.
        output_controls: Custom controls to add to output stage.
        debug: Enable debug logging.
    """

    def __init__(
        self,
        input_config: Optional["InputControlsConfig"] = None,
        output_config: Optional["OutputControlsConfig"] = None,
        input_controls: Optional[list[BaseControl]] = None,
        output_controls: Optional[list[BaseControl]] = None,
        debug: bool = False,
    ) -> None:
        self._debug = debug

        # Build control lists per stage
        self._input_controls: list[BaseControl] = []
        self._output_controls: list[BaseControl] = []

        # --- Input stage built-in controls ---
        if input_config:
            if input_config.pii_phi.enabled:
                self._input_controls.append(PIIControl(input_config.pii_phi))
                if debug:
                    pii = input_config.pii_phi
                    print(
                        f"[glacis] Input PIIControl initialized "
                        f"(model={pii.model}, mode={pii.mode})"
                    )

            if input_config.word_filter.enabled:
                self._input_controls.append(WordFilterControl(input_config.word_filter))
                if debug:
                    n = len(input_config.word_filter.entities)
                    print(
                        f"[glacis] Input WordFilterControl "
                        f"initialized ({n} entities)"
                    )

            if input_config.jailbreak.enabled:
                self._input_controls.append(JailbreakControl(input_config.jailbreak))
                if debug:
                    print(
                        f"[glacis] Input JailbreakControl initialized "
                        f"(model={input_config.jailbreak.model})"
                    )

        # --- Output stage built-in controls ---
        if output_config:
            if output_config.pii_phi.enabled:
                self._output_controls.append(PIIControl(output_config.pii_phi))
                if debug:
                    pii = output_config.pii_phi
                    print(
                        f"[glacis] Output PIIControl initialized "
                        f"(model={pii.model}, mode={pii.mode})"
                    )

            if output_config.word_filter.enabled:
                self._output_controls.append(WordFilterControl(output_config.word_filter))
                if debug:
                    n = len(output_config.word_filter.entities)
                    print(
                        f"[glacis] Output WordFilterControl "
                        f"initialized ({n} entities)"
                    )

            if output_config.jailbreak.enabled:
                self._output_controls.append(JailbreakControl(output_config.jailbreak))
                if debug:
                    print(
                        f"[glacis] Output JailbreakControl initialized "
                        f"(model={output_config.jailbreak.model})"
                    )

        # --- Custom controls ---
        for ctrl in (input_controls or []):
            self._input_controls.append(ctrl)
            if debug:
                print(f"[glacis] Input custom control '{ctrl.control_type}'")

        for ctrl in (output_controls or []):
            self._output_controls.append(ctrl)
            if debug:
                print(f"[glacis] Output custom control '{ctrl.control_type}'")

    @property
    def has_input_controls(self) -> bool:
        """Whether any input controls are configured."""
        return bool(self._input_controls)

    @property
    def has_output_controls(self) -> bool:
        """Whether any output controls are configured."""
        return bool(self._output_controls)

    def run_input(self, text: str) -> StageResult:
        """Run input pipeline: all controls in parallel.

        Args:
            text: Original input text.

        Returns:
            StageResult with all results and block decision.
        """
        return self._run_stage(text, self._input_controls)

    def run_output(self, text: str) -> StageResult:
        """Run output pipeline: all controls in parallel.

        Args:
            text: LLM response text.

        Returns:
            StageResult with all results and block decision.
        """
        return self._run_stage(text, self._output_controls)

    def _run_stage(
        self, text: str, controls: list[BaseControl],
    ) -> StageResult:
        """Run all controls in parallel and aggregate results."""
        results = self._run_controls(text, controls)
        should_block = any(r.action == "block" for r in results)

        return StageResult(
            results=results,
            effective_text=text,
            should_block=should_block,
        )

    def _run_controls(
        self, text: str, controls: list[BaseControl],
    ) -> list[ControlResult]:
        """Run controls in parallel."""
        if not controls:
            return []

        # Single control — direct call, no thread overhead
        if len(controls) == 1:
            try:
                result = controls[0].check(text)
                if self._debug:
                    print(
                        f"[glacis] {result.control_type}: "
                        f"detected={result.detected}, action={result.action} "
                        f"({result.latency_ms}ms)"
                    )
                return [result]
            except Exception as e:
                if self._debug:
                    print(f"[glacis] {controls[0].control_type}: ERROR - {e}")
                return [ControlResult(
                    control_type=controls[0].control_type,
                    detected=False,
                    action="error",
                    metadata={"error": str(e)},
                )]

        # Multiple controls — parallel execution
        results: list[ControlResult] = []
        with ThreadPoolExecutor(max_workers=len(controls)) as executor:
            future_to_control = {
                executor.submit(control.check, text): control
                for control in controls
            }
            for future in as_completed(future_to_control):
                control = future_to_control[future]
                try:
                    result = future.result()
                    results.append(result)
                    if self._debug:
                        print(
                            f"[glacis] {result.control_type}: "
                            f"detected={result.detected}, action={result.action} "
                            f"({result.latency_ms}ms)"
                        )
                except Exception as e:
                    results.append(ControlResult(
                        control_type=control.control_type,
                        detected=False,
                        action="error",
                        metadata={"error": str(e)},
                    ))
                    if self._debug:
                        print(f"[glacis] {control.control_type}: ERROR - {e}")

        return results

    def close(self) -> None:
        """Release resources for all controls."""
        for ctrl in self._input_controls + self._output_controls:
            ctrl.close()


# Public exports
__all__ = [
    "BaseControl",
    "ControlAction",
    "ControlResult",
    "ControlsRunner",
    "JailbreakControl",
    "PIIControl",
    "StageResult",
    "WordFilterControl",
]
