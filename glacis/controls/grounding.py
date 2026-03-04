"""
GLACIS Grounding Control (Stub).

Placeholder for grounding / hallucination detection. The ``check(text)``
interface only receives a single text string, but effective grounding
validation requires comparing the LLM output against reference material.

This built-in implementation always passes (detected=False). To implement
real grounding validation, create a custom control that accepts
``reference_text`` in its constructor::

    class MyGroundingControl(BaseControl):
        control_type = "grounding"

        def __init__(self, reference_text: str, threshold: float = 0.5, **kwargs):
            self.reference_text = reference_text
            self.threshold = threshold

        def check(self, text: str) -> ControlResult:
            # Compare text against self.reference_text
            ...

Then register it in ``glacis.yaml``::

    controls:
      output:
        custom:
          - path: "my_grounding.MyGroundingControl"
            enabled: true
            args:
              reference_text: "The source material to ground against..."
              threshold: 0.7
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from glacis.controls.base import BaseControl, ControlResult

if TYPE_CHECKING:
    from glacis.config import GroundingControlConfig


class GroundingControl(BaseControl):
    """
    Grounding control stub.

    Always returns ``detected=False``. Establishes the ``"grounding"``
    control type for attestation classification. Extend via custom controls
    for real grounding validation with reference text.

    Args:
        config: GroundingControlConfig with threshold and if_detected settings.
    """

    control_type = "grounding"

    def __init__(self, config: "GroundingControlConfig") -> None:
        self._config = config

    def check(self, text: str) -> ControlResult:
        """
        Stub check — always passes.

        Args:
            text: The text to analyze (unused in stub).

        Returns:
            ControlResult with detected=False and stub metadata.
        """
        return ControlResult(
            control_type=self.control_type,
            detected=False,
            action="forward",
            latency_ms=0,
            metadata={
                "stub": True,
                "note": "Built-in stub. Use custom control with reference_text for real grounding.",
            },
        )
