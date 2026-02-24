"""Configuration model for the GLACIS Judge Framework.

Defines tunable thresholds for score aggregation and recommendation logic.
Works for any scored evaluation — fact-checking (0-3 scale),
conformity review (0-1 scale), or custom scales. Higher score always
means better.

Usage:
    >>> from glacis.judges import JudgesConfig, JudgeRunner
    >>>
    >>> # L1 fact-checking (defaults)
    >>> runner = JudgeRunner(judges=[...])
    >>>
    >>> # Custom thresholds
    >>> cfg = JudgesConfig(uphold_threshold=2.5, consensus_threshold=0.5)
    >>> runner = JudgeRunner(judges=[...], config=cfg)
    >>>
    >>> # L2 conformity review (future)
    >>> cfg = JudgesConfig(
    ...     max_score=1.0,
    ...     uphold_threshold=0.7,
    ...     borderline_threshold=0.4,
    ...     consensus_threshold=0.2,
    ... )
"""

from pydantic import BaseModel, Field


class JudgesConfig(BaseModel):
    """Tunable thresholds for the judge pipeline.

    Attributes:
        max_score: Maximum score on the rubric scale. Default 3.0 (0-3 rubric).
        consensus_threshold: Maximum score spread between judges before
            flagging disagreement. Default 1.0.
        uphold_threshold: Minimum average score for "uphold" recommendation.
            Default 2.0.
        borderline_threshold: Minimum average score for "borderline"
            recommendation. Scores below this → "escalate". Default 1.0.
        score_precision: Decimal places for rounding final_score. Default 4.
    """

    max_score: float = Field(
        default=3.0,
        ge=0.0,
        description="Maximum score on the rubric scale",
    )
    consensus_threshold: float = Field(
        default=1.0,
        ge=0.0,
        description="Max score spread between judges before flagging disagreement",
    )
    uphold_threshold: float = Field(
        default=2.0,
        ge=0.0,
        description="Minimum avg score for 'uphold' recommendation",
    )
    borderline_threshold: float = Field(
        default=1.0,
        ge=0.0,
        description="Minimum avg score for 'borderline' (below → 'escalate')",
    )
    score_precision: int = Field(
        default=4,
        ge=0,
        le=10,
        description="Decimal places for rounding final_score",
    )
