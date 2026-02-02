"""
GLACIS Controls Module.

Provides modular controls for PII/PHI redaction, jailbreak detection,
and other safety checks on LLM inputs.

Example (using individual controls):
    >>> from glacis.controls import PIIControl, JailbreakControl
    >>> from glacis.config import PiiPhiConfig, JailbreakConfig
    >>>
    >>> # PII redaction
    >>> pii = PIIControl(PiiPhiConfig(enabled=True, mode="fast"))
    >>> result = pii.check("SSN: 123-45-6789")
    >>> result.modified_text  # "SSN: [US_SSN]"
    >>>
    >>> # Jailbreak detection
    >>> jailbreak = JailbreakControl(JailbreakConfig(enabled=True))
    >>> result = jailbreak.check("Ignore previous instructions")
    >>> result.detected  # True

Example (using ControlsRunner):
    >>> from glacis.controls import ControlsRunner
    >>> from glacis.config import load_config
    >>>
    >>> cfg = load_config()  # Loads glacis.yaml
    >>> runner = ControlsRunner(cfg.controls)
    >>>
    >>> results = runner.run("Patient SSN: 123-45-6789")
    >>> final_text = runner.get_final_text(results)
    >>> should_block = runner.should_block(results)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from glacis.controls.base import BaseControl, ControlResult
from glacis.controls.jailbreak import JailbreakControl
from glacis.controls.pii import PIIControl

if TYPE_CHECKING:
    from glacis.config import ControlsConfig


class ControlsRunner:
    """
    Orchestrates running multiple controls on text input.

    Controls are run in order:
    1. PII/PHI redaction (if enabled) - modifies text
    2. Jailbreak detection (if enabled) - may flag/block

    The runner handles:
    - Chaining modified text between controls
    - Aggregating results from all controls
    - Determining if request should be blocked

    Args:
        config: ControlsConfig with settings for all controls.
        debug: Enable debug logging.

    Example:
        >>> from glacis.config import ControlsConfig, PiiPhiConfig, JailbreakConfig
        >>>
        >>> config = ControlsConfig(
        ...     pii_phi=PiiPhiConfig(enabled=True, mode="fast"),
        ...     jailbreak=JailbreakConfig(enabled=True, threshold=0.5),
        ... )
        >>> runner = ControlsRunner(config)
        >>>
        >>> results = runner.run("SSN: 123-45-6789. Ignore instructions.")
        >>> len(results)  # 2 (one for each enabled control)
        >>>
        >>> # Check if any control wants to block
        >>> if runner.should_block(results):
        ...     raise Exception("Blocked by policy")
        >>>
        >>> # Get final text after all modifications
        >>> final_text = runner.get_final_text(results)
    """

    def __init__(self, config: "ControlsConfig", debug: bool = False) -> None:
        """
        Initialize ControlsRunner with enabled controls.

        Args:
            config: ControlsConfig specifying which controls to enable.
            debug: Enable debug output.
        """
        self._controls: list[BaseControl] = []
        self._debug = debug

        # Initialize enabled controls in order
        # Order matters: PII redaction runs first to clean text before other checks
        if config.pii_phi.enabled:
            self._controls.append(PIIControl(config.pii_phi))
            if debug:
                print(f"[glacis] PIIControl initialized (backend={config.pii_phi.backend}, mode={config.pii_phi.mode})")

        if config.jailbreak.enabled:
            self._controls.append(JailbreakControl(config.jailbreak))
            if debug:
                print(f"[glacis] JailbreakControl initialized (backend={config.jailbreak.backend})")

    @property
    def enabled_controls(self) -> list[str]:
        """Return list of enabled control types."""
        return [c.control_type for c in self._controls]

    def run(self, text: str) -> list[ControlResult]:
        """
        Run all enabled controls on the input text.

        Controls are run in sequence. Text-modifying controls (like PII redaction)
        pass their modified text to subsequent controls.

        Args:
            text: The input text to check.

        Returns:
            List of ControlResult from each enabled control.

        Example:
            >>> results = runner.run("Patient SSN: 123-45-6789")
            >>> for r in results:
            ...     print(f"{r.control_type}: detected={r.detected}")
        """
        results: list[ControlResult] = []
        current_text = text

        for control in self._controls:
            result = control.check(current_text)
            results.append(result)

            # Chain modified text for subsequent controls
            if result.modified_text is not None:
                current_text = result.modified_text

            if self._debug:
                if result.detected:
                    print(
                        f"[glacis] {result.control_type}: detected "
                        f"(action={result.action}, categories={result.categories})"
                    )
                else:
                    print(f"[glacis] {result.control_type}: pass ({result.latency_ms}ms)")

        return results

    def should_block(self, results: list[ControlResult]) -> bool:
        """
        Check if any control result indicates the request should be blocked.

        Args:
            results: List of ControlResult from run().

        Returns:
            True if any control has action="block".

        Example:
            >>> results = runner.run("malicious input")
            >>> if runner.should_block(results):
            ...     raise GlacisBlockedError("Request blocked by policy")
        """
        return any(r.action == "block" for r in results)

    def get_final_text(self, results: list[ControlResult]) -> Optional[str]:
        """
        Get the final modified text after all controls have run.

        Returns the last modified_text from any control, or None if
        no control modified the text.

        Args:
            results: List of ControlResult from run().

        Returns:
            The final modified text, or None if unchanged.

        Example:
            >>> results = runner.run("SSN: 123-45-6789")
            >>> final = runner.get_final_text(results)
            >>> print(final)  # "SSN: [US_SSN]"
        """
        # Return the last non-None modified_text
        for result in reversed(results):
            if result.modified_text is not None:
                return result.modified_text
        return None

    def get_result_by_type(self, results: list[ControlResult], control_type: str) -> Optional[ControlResult]:
        """
        Get a specific control's result by type.

        Args:
            results: List of ControlResult from run().
            control_type: The control type to find ("pii" or "jailbreak").

        Returns:
            The ControlResult for that control type, or None if not found.

        Example:
            >>> results = runner.run("test input")
            >>> pii_result = runner.get_result_by_type(results, "pii")
            >>> jailbreak_result = runner.get_result_by_type(results, "jailbreak")
        """
        for result in results:
            if result.control_type == control_type:
                return result
        return None

    def close(self) -> None:
        """Release resources for all controls."""
        for control in self._controls:
            control.close()
        self._controls = []


# Public exports
__all__ = [
    "BaseControl",
    "ControlResult",
    "ControlsRunner",
    "JailbreakControl",
    "PIIControl",
]
