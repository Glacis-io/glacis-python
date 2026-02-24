"""
GLACIS Word Filter Control.

Literal string matching control for detecting configured terms.
Uses ``re.escape()`` + ``re.IGNORECASE`` for safe, case-insensitive
matching without ReDoS risk.

Example:
    >>> from glacis.config import WordFilterControlConfig
    >>> config = WordFilterControlConfig(
    ...     enabled=True,
    ...     entities=["confidential", "proprietary"],
    ... )
    >>> control = WordFilterControl(config)
    >>> result = control.check("This is confidential data")
    >>> result.detected
    True
    >>> result.categories
    ['confidential']
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from glacis.controls.base import BaseControl, ControlResult

if TYPE_CHECKING:
    from glacis.config import WordFilterControlConfig

# Safety limits to prevent abuse
MAX_ENTITIES = 500
MAX_ENTITY_LENGTH = 256


class WordFilterControl(BaseControl):
    """
    Literal keyword matching control.

    Matches configured terms case-insensitively using escaped regex patterns.
    Only literal matching is supported (no wildcards or regex from config).

    Args:
        config: WordFilterControlConfig with enabled, entities,
                and if_detected settings.

    Raises:
        ValueError: If entities exceed safety limits.
    """

    control_type = "word_filter"

    def __init__(self, config: "WordFilterControlConfig") -> None:
        self._config = config

        # Validate safety limits
        if len(config.entities) > MAX_ENTITIES:
            raise ValueError(
                f"Word filter supports at most {MAX_ENTITIES} entities, "
                f"got {len(config.entities)}"
            )
        for term in config.entities:
            if len(term) > MAX_ENTITY_LENGTH:
                raise ValueError(
                    f"Word filter entity exceeds {MAX_ENTITY_LENGTH} chars: "
                    f"{term[:50]}..."
                )

        # Pre-compile the combined pattern for all terms
        # re.escape ensures no regex injection / ReDoS risk
        self._pattern: re.Pattern[str] | None = None
        if config.entities:
            escaped = [re.escape(term) for term in config.entities]
            self._pattern = re.compile("|".join(escaped), re.IGNORECASE)

    def check(self, text: str) -> ControlResult:
        """
        Check text for configured terms.

        Args:
            text: Input text to check.

        Returns:
            ControlResult with matched terms in categories.
        """
        start_time = time.perf_counter()

        if not text or not self._pattern:
            return ControlResult(
                control_type=self.control_type,
                detected=False,
                action="forward",
                latency_ms=0,
            )

        matches = list(self._pattern.finditer(text))
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        if not matches:
            return ControlResult(
                control_type=self.control_type,
                detected=False,
                action="forward",
                latency_ms=latency_ms,
            )

        # Collect unique matched terms (lowercased for de-dup)
        categories = sorted({m.group().lower() for m in matches})

        return ControlResult(
            control_type=self.control_type,
            detected=True,
            action=self._config.if_detected,
            categories=categories,
            latency_ms=latency_ms,
            metadata={"count": len(matches)},
        )
