"""
GLACIS Judge Framework — Abstract base class and data models for LLM judges.

Judges evaluate items against reference data and produce scored verdicts.
Multiple judges can run in parallel, and their scores are aggregated into
a ReviewResult. The framework is use-case agnostic — it works for
fact-checking (L1, 0-3 scale) and conformity review (L2, 0-1 scale).

Thresholds are configurable via JudgesConfig. Without explicit config,
defaults match the L1 fact-checking scale (max_score=3, uphold>=2, etc.).

The ReviewResult is then attested via glacis.attest() to create an auditable
review record linked to the original attestation by operation_id.

Example:
    >>> from glacis.judges import BaseJudge, JudgeRunner, JudgeVerdict
    >>>
    >>> class MyJudge(BaseJudge):
    ...     judge_id = "my-judge"
    ...     def evaluate(self, item, reference=None, rubric=None):
    ...         return JudgeVerdict(judge_id=self.judge_id, score=2.5, rationale="Good")
    >>>
    >>> runner = JudgeRunner(judges=[MyJudge()])
    >>> result = runner.run({"question": "...", "answer": "..."}, reference="source doc")
    >>> result.final_score
    2.5
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from glacis.judges.config import JudgesConfig

logger = logging.getLogger(__name__)


class JudgeVerdict(BaseModel):
    """Result from a single judge evaluating a single item.

    Attributes:
        judge_id: Identifier for the judge (e.g., model ID).
        score: Numeric score (0 or above). Scale is defined by JudgesConfig.max_score.
        rationale: Judge's explanation for the score.
        latency_ms: Processing time in milliseconds.
        metadata: Judge-specific metadata for audit trail.
    """

    judge_id: str
    score: float = Field(ge=0.0)
    rationale: str
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewResult(BaseModel):
    """Aggregated result from running all judges on one item.

    Attributes:
        verdicts: Individual judge results.
        final_score: Average of all judge scores.
        max_score: Maximum possible score (from config).
        consensus: True if judges agree within the threshold.
        recommendation: Derived action — uphold, borderline, or escalate.
    """

    verdicts: list[JudgeVerdict]
    final_score: float
    max_score: float = 3.0
    consensus: bool
    recommendation: Literal["uphold", "borderline", "escalate"]


class BaseJudge(ABC):
    """Abstract base class for LLM judges.

    Subclasses must set `judge_id` and implement `evaluate()`.
    The evaluate() method receives a generic item dict — the structure
    depends on the use case (e.g., QA pair for fact-checking, control
    plane results for conformity review).
    """

    judge_id: str

    @abstractmethod
    def evaluate(
        self,
        item: dict[str, Any],
        reference: Optional[str] = None,
        rubric: Optional[str] = None,
    ) -> JudgeVerdict:
        """Evaluate a single item against reference data.

        Args:
            item: The item to evaluate. Structure is use-case specific.
            reference: Optional reference data for evaluation context.
            rubric: Optional scoring rubric override (prompt text).

        Returns:
            JudgeVerdict with score and rationale.
        """
        pass

    def close(self) -> None:
        """Release resources held by this judge."""
        pass

    def __enter__(self) -> BaseJudge:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class JudgeRunner:
    """Runs multiple judges on an item and aggregates their scores.

    All thresholds are driven by JudgesConfig. Without explicit config,
    defaults match the L1 fact-checking scale (max=3, uphold>=2, borderline>=1).

    Args:
        judges: List of BaseJudge instances to run.
        consensus_threshold: Maximum score difference before flagging
            disagreement. Ignored if config is provided.
        config: Optional JudgesConfig with all thresholds. If None,
            uses defaults (backward compatible).

    Example:
        >>> runner = JudgeRunner(judges=[judge_a, judge_b])
        >>> result = runner.run(item, reference=source_doc)
        >>> result.recommendation  # "uphold", "borderline", or "escalate"
        >>>
        >>> # With custom config
        >>> cfg = JudgesConfig(uphold_threshold=2.5, consensus_threshold=0.5)
        >>> runner = JudgeRunner(judges=[judge_a], config=cfg)
    """

    def __init__(
        self,
        judges: list[BaseJudge],
        consensus_threshold: float = 1.0,
        config: Optional[JudgesConfig] = None,
    ) -> None:
        self._judges = judges
        self._config = config or JudgesConfig(consensus_threshold=consensus_threshold)

    @property
    def config(self) -> JudgesConfig:
        """Current judge configuration."""
        return self._config

    def run(
        self,
        item: dict[str, Any],
        reference: Optional[str] = None,
        rubric: Optional[str] = None,
    ) -> ReviewResult:
        """Run all judges on the item and aggregate results.

        If a judge raises an exception, it is caught and recorded as a
        verdict with score=0 and the error message as rationale.

        Args:
            item: The item to evaluate.
            reference: Optional reference data for evaluation context.
            rubric: Optional scoring rubric override.

        Returns:
            ReviewResult with aggregated scores and recommendation.
        """
        verdicts: list[JudgeVerdict] = []

        for judge in self._judges:
            try:
                verdict = judge.evaluate(item, reference, rubric)
                verdicts.append(verdict)
            except Exception as e:
                logger.warning("Judge %s failed: %s", judge.judge_id, e)
                verdicts.append(
                    JudgeVerdict(
                        judge_id=judge.judge_id,
                        score=0.0,
                        rationale=f"Judge error: {e}",
                        metadata={"error": str(e)},
                    )
                )

        scores = [v.score for v in verdicts]
        final_score = sum(scores) / len(scores) if scores else 0.0
        consensus = (
            (max(scores) - min(scores)) <= self._config.consensus_threshold
            if len(scores) >= 2
            else True
        )

        recommendation: Literal["uphold", "borderline", "escalate"]
        if final_score >= self._config.uphold_threshold:
            recommendation = "uphold"
        elif final_score >= self._config.borderline_threshold:
            recommendation = "borderline"
        else:
            recommendation = "escalate"

        return ReviewResult(
            verdicts=verdicts,
            final_score=round(final_score, self._config.score_precision),
            max_score=self._config.max_score,
            consensus=consensus,
            recommendation=recommendation,
        )

    def close(self) -> None:
        """Release resources for all judges."""
        for judge in self._judges:
            judge.close()
