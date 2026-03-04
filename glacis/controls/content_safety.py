"""
GLACIS Content Safety Control.

Detects toxic, harmful, or policy-violating content using HuggingFace
toxicity classification models.

Supported models:
- toxic-bert: unitary/toxic-bert (multi-label toxicity, DeBERTa-based)
  Categories: toxic, severe_toxic, obscene, threat, insult, identity_hate

Example:
    >>> from glacis.controls.content_safety import ContentSafetyControl
    >>> from glacis.config import ContentSafetyControlConfig
    >>>
    >>> control = ContentSafetyControl(
    ...     ContentSafetyControlConfig(enabled=True, threshold=0.5)
    ... )
    >>> result = control.check("You are terrible and I hate you")
    >>> result.detected  # True
    >>> result.categories  # ['insult', 'toxic']
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from glacis.controls.base import BaseControl, ControlResult

if TYPE_CHECKING:
    from glacis.config import ContentSafetyControlConfig


class ContentSafetyControl(BaseControl):
    """
    Content safety / toxicity detection control.

    Uses HuggingFace toxicity classification models to detect harmful
    content across multiple categories. The model is lazy-loaded on first
    ``check()`` call to avoid startup overhead.

    Supported models:
    - toxic-bert: unitary/toxic-bert
      - Multi-label classifier (can detect multiple categories simultaneously)
      - Categories: toxic, severe_toxic, obscene, threat, insult, identity_hate
      - ~110M parameters (BERT-base)

    Args:
        config: ContentSafetyControlConfig with model, threshold, categories,
                and if_detected settings.

    Example:
        >>> config = ContentSafetyControlConfig(
        ...     enabled=True,
        ...     model="toxic-bert",
        ...     threshold=0.5,
        ...     categories=["threat", "insult"],
        ...     if_detected="flag",
        ... )
        >>> control = ContentSafetyControl(config)
        >>> result = control.check("I will hurt you")
        >>> if result.detected:
        ...     print(f"Unsafe content: {result.categories}")
    """

    control_type = "content_safety"

    MODEL_REGISTRY = {
        "toxic-bert": "unitary/toxic-bert",
    }

    def __init__(self, config: "ContentSafetyControlConfig") -> None:
        self._config = config
        self._classifier: Optional[Any] = None  # Lazy init

        # Validate model
        if config.model not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown content safety model: {config.model}. "
                f"Available models: {list(self.MODEL_REGISTRY.keys())}"
            )

        # Normalize category filter to lowercase for comparison
        self._category_filter = {c.lower() for c in config.categories} if config.categories else set()

    def _ensure_initialized(self) -> None:
        """Lazy-initialize the classifier on first use."""
        if self._classifier is not None:
            return

        import os

        try:
            from transformers import logging as hf_logging
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "Content safety detection requires the 'transformers' package. "
                "Install with: pip install glacis[content-safety]"
            )

        # Suppress HuggingFace verbosity
        hf_logging.set_verbosity_error()  # type: ignore[no-untyped-call]

        # Disable HuggingFace Hub telemetry
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        model_name = self.MODEL_REGISTRY[self._config.model]

        # toxic-bert is a multi-label classifier — use text-classification
        # with top_k=None to get scores for all labels
        self._classifier = pipeline(
            "text-classification",
            model=model_name,
            device="cpu",
            top_k=None,  # Return all label scores
        )

    def check(self, text: str) -> ControlResult:
        """
        Check text for toxic or harmful content.

        Args:
            text: The text to analyze.

        Returns:
            ControlResult with detection results:
            - detected: True if any category exceeds threshold
            - action: Configured action if detected, "forward" otherwise
            - score: Maximum score across checked categories
            - categories: List of categories that exceeded threshold
            - metadata: Raw scores and model info
        """
        self._ensure_initialized()
        assert self._classifier is not None

        start_time = time.perf_counter()

        if not text:
            return ControlResult(
                control_type=self.control_type,
                detected=False,
                action="forward",
                latency_ms=0,
            )

        # Run classification — returns list of {label, score} dicts
        # Truncate to model's max length (512 tokens for BERT)
        results = self._classifier(text, truncation=True, max_length=512)

        # top_k=None returns [[{label, score}, ...]] — unwrap outer list
        if results and isinstance(results[0], list):
            results = results[0]

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Filter by configured categories (if any)
        flagged_categories = []
        max_score = 0.0
        all_scores: dict[str, float] = {}

        for item in results:
            label = item["label"].lower()
            score = item["score"]
            all_scores[label] = score

            # Skip if category filter is active and this label isn't in it
            if self._category_filter and label not in self._category_filter:
                continue

            if score >= self._config.threshold:
                flagged_categories.append(label)

            max_score = max(max_score, score)

        detected = bool(flagged_categories)

        return ControlResult(
            control_type=self.control_type,
            detected=detected,
            action=self._config.if_detected if detected else "forward",
            score=max_score,
            categories=sorted(flagged_categories),
            latency_ms=latency_ms,
            metadata={
                "all_scores": all_scores,
                "model": self._config.model,
                "threshold": self._config.threshold,
            },
        )

    def close(self) -> None:
        """Release classifier resources."""
        if self._classifier is not None:
            del self._classifier
            self._classifier = None
