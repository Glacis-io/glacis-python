"""
Base interface for GLACIS controls.

All controls (PII, jailbreak, toxicity, etc.) implement the BaseControl interface.
This enables a pluggable, modular control architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ControlResult(BaseModel):
    """
    Result from running a control.

    All controls return this standardized result, enabling consistent
    handling in the ControlsRunner and attestation pipeline.

    Attributes:
        control_type: The type of control (e.g., "pii", "jailbreak")
        detected: Whether a threat/issue was detected
        action: Action taken/recommended ("pass", "flag", "block", "redact")
        score: Confidence score from ML-based controls (0-1)
        categories: List of detected categories (e.g., ["US_SSN", "PERSON"])
        latency_ms: Processing time in milliseconds
        modified_text: Text after transformation (for redaction controls)
        metadata: Control-specific metadata (for audit trail)
    """

    control_type: str = Field(description="Control type identifier")
    detected: bool = Field(default=False, description="Whether threat was detected")
    action: Literal["pass", "flag", "block", "redact"] = Field(
        default="pass", description="Action taken or recommended"
    )
    score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score (0-1)"
    )
    categories: list[str] = Field(
        default_factory=list, description="Detected categories"
    )
    latency_ms: int = Field(default=0, description="Processing time in ms")
    modified_text: Optional[str] = Field(
        default=None, description="Text after transformation (if applicable)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Control-specific metadata for audit"
    )


class BaseControl(ABC):
    """
    Abstract base class for all GLACIS controls.

    Controls are responsible for:
    1. Checking text for threats/issues
    2. Optionally transforming text (e.g., redaction)
    3. Returning standardized ControlResult

    Implementations:
    - PIIControl: PII/PHI detection and redaction
    - JailbreakControl: Jailbreak/prompt injection detection

    Example:
        >>> control = PIIControl(config)
        >>> result = control.check("SSN: 123-45-6789")
        >>> result.detected
        True
        >>> result.modified_text
        "SSN: [US_SSN]"
    """

    control_type: str  # Class attribute - override in subclass

    @abstractmethod
    def check(self, text: str) -> ControlResult:
        """
        Check text against this control.

        Args:
            text: Input text to check

        Returns:
            ControlResult with detection info and optional transformed text
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
        """Context manager exit - release resources."""
        self.close()
