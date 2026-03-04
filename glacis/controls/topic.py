"""
GLACIS Topic Control.

Keyword-based topic enforcement control that supports two modes:
- **Blocklist**: Flag text containing blocked topic keywords
- **Allowlist**: Flag text that doesn't match any allowed topic (off-topic)

Uses ``re.escape()`` + ``re.IGNORECASE`` for safe, case-insensitive matching.

Example:
    >>> from glacis.controls.topic import TopicControl
    >>> from glacis.config import TopicControlConfig
    >>>
    >>> # Blocklist mode: flag text about competitors
    >>> config = TopicControlConfig(
    ...     enabled=True,
    ...     blocked_topics=["competitor", "rival product"],
    ...     if_detected="flag",
    ... )
    >>> control = TopicControl(config)
    >>> result = control.check("Our competitor has a similar feature")
    >>> result.detected  # True
    >>> result.categories  # ['competitor']
    >>>
    >>> # Allowlist mode: only allow healthcare topics
    >>> config = TopicControlConfig(
    ...     enabled=True,
    ...     allowed_topics=["healthcare", "medical", "patient"],
    ...     if_detected="block",
    ... )
    >>> control = TopicControl(config)
    >>> result = control.check("Tell me about sports")
    >>> result.detected  # True (off-topic)
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Optional

from glacis.controls.base import BaseControl, ControlResult

if TYPE_CHECKING:
    from glacis.config import TopicControlConfig

# Safety limits
MAX_TOPICS = 500
MAX_TOPIC_LENGTH = 256


class TopicControl(BaseControl):
    """
    Topic enforcement control using keyword matching.

    Supports two complementary modes:
    - **blocked_topics**: Flag when any blocked keyword is found in the text
    - **allowed_topics**: Flag when none of the allowed keywords are found (off-topic)

    When both are configured, blocked topics are checked first. If a blocked
    topic is found, the text is flagged regardless of allowed topics.

    Args:
        config: TopicControlConfig with allowed_topics, blocked_topics,
                and if_detected settings.

    Raises:
        ValueError: If topic lists exceed safety limits.
    """

    control_type = "topic"

    def __init__(self, config: "TopicControlConfig") -> None:
        self._config = config

        # Validate safety limits
        all_topics = config.allowed_topics + config.blocked_topics
        if len(all_topics) > MAX_TOPICS:
            raise ValueError(
                f"Topic control supports at most {MAX_TOPICS} topics total, "
                f"got {len(all_topics)}"
            )
        for topic in all_topics:
            if len(topic) > MAX_TOPIC_LENGTH:
                raise ValueError(
                    f"Topic keyword exceeds {MAX_TOPIC_LENGTH} chars: "
                    f"{topic[:50]}..."
                )

        # Pre-compile patterns
        self._blocked_pattern: Optional[re.Pattern[str]] = None
        if config.blocked_topics:
            escaped = [re.escape(t) for t in config.blocked_topics]
            self._blocked_pattern = re.compile("|".join(escaped), re.IGNORECASE)

        self._allowed_pattern: Optional[re.Pattern[str]] = None
        if config.allowed_topics:
            escaped = [re.escape(t) for t in config.allowed_topics]
            self._allowed_pattern = re.compile("|".join(escaped), re.IGNORECASE)

    def check(self, text: str) -> ControlResult:
        """
        Check text for topic violations.

        Logic:
        1. If blocked_topics configured and any match → detected (blocked topic)
        2. If allowed_topics configured and none match → detected (off-topic)
        3. Otherwise → not detected

        Args:
            text: Input text to check.

        Returns:
            ControlResult with matched topics in categories.
        """
        start_time = time.perf_counter()

        if not text:
            return ControlResult(
                control_type=self.control_type,
                detected=False,
                action="forward",
                latency_ms=0,
            )

        # Check blocked topics first
        if self._blocked_pattern:
            matches = list(self._blocked_pattern.finditer(text))
            if matches:
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                categories = sorted({m.group().lower() for m in matches})
                return ControlResult(
                    control_type=self.control_type,
                    detected=True,
                    action=self._config.if_detected,
                    categories=categories,
                    latency_ms=latency_ms,
                    metadata={"reason": "blocked_topic", "count": len(matches)},
                )

        # Check allowed topics (if configured, text must match at least one)
        if self._allowed_pattern:
            matches = list(self._allowed_pattern.finditer(text))
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            if not matches:
                return ControlResult(
                    control_type=self.control_type,
                    detected=True,
                    action=self._config.if_detected,
                    categories=["off_topic"],
                    latency_ms=latency_ms,
                    metadata={"reason": "off_topic"},
                )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return ControlResult(
            control_type=self.control_type,
            detected=False,
            action="forward",
            latency_ms=latency_ms,
        )
