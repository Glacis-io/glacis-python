"""
Base interface for GLACIS controls.

All controls (PII, jailbreak, word_filter, custom, etc.) implement the
BaseControl interface. This enables a pluggable, modular control architecture
where all controls run in parallel via ThreadPoolExecutor.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# Canonical action type — used in both ControlResult (SDK) and
# ControlExecution (wire format).  No translation layer needed.
ControlAction = Literal["forward", "flag", "block", "error"]


class ControlResult(BaseModel):
    """
    Result from running a control.

    All controls return this standardized result, enabling consistent
    handling in the pipeline and attestation.

    Attributes:
        control_type: The type of control (e.g., "pii", "jailbreak", "word_filter")
        detected: Whether a threat/issue was detected
        action: Action taken — "forward", "flag", "block", or "error"
        score: Confidence score from ML-based controls (0-1)
        categories: List of detected categories (e.g., ["US_SSN", "PERSON"])
        latency_ms: Processing time in milliseconds
        metadata: Control-specific metadata (for audit trail)
        modified_text: Reserved for future rewrite mode (not currently used)
    """

    control_type: str = Field(description="Control type identifier")
    detected: bool = Field(default=False, description="Whether threat was detected")
    action: ControlAction = Field(
        default="forward", description="Action taken or recommended"
    )
    score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score (0-1)"
    )
    categories: list[str] = Field(
        default_factory=list, description="Detected categories"
    )
    latency_ms: int = Field(default=0, description="Processing time in ms")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Control-specific metadata for audit"
    )
    modified_text: Optional[str] = Field(
        default=None, description="Reserved for future rewrite mode (not currently used)"
    )


class BaseControl(ABC):
    """
    Abstract base class for all GLACIS controls.

    Controls are responsible for:
    1. Checking text for threats/issues
    2. Returning standardized ControlResult with detection info

    Built-in implementations:
    - PIIControl: PII/PHI detection
    - JailbreakControl: Jailbreak/prompt injection detection
    - WordFilterControl: Literal keyword matching

    Custom controls implement this interface and are injected via
    ``input_controls`` / ``output_controls`` on the integration wrappers.

    Example:
        >>> class MyGuard(BaseControl):
        ...     control_type = "custom"
        ...     def check(self, text):
        ...         return ControlResult(control_type="custom", detected=False)
    """

    control_type: str  # Class attribute — override in subclass

    @abstractmethod
    def check(self, text: str) -> ControlResult:
        """
        Check text against this control.

        Args:
            text: Input text to check

        Returns:
            ControlResult with detection info
        """
        pass

    def close(self) -> None:
        """
        Release resources held by this control.

        Override in subclasses that hold expensive resources
        (e.g., ML models, database connections).
        """
        pass

    def __enter__(self) -> "BaseControl":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit — release resources."""
        self.close()
