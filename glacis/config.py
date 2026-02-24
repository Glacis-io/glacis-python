"""
GLACIS Configuration Module

Provides configuration models and loading for glacis.yaml config files.
Controls are configured per-stage (input/output) with per-control toggles.

Example glacis.yaml:
    version: "1.3"
    policy:
      id: "hipaa-safe-harbor"
      environment: "production"
      tags: ["healthcare", "hipaa"]
    controls:
      output_block_action: "block"
      input:
        pii_phi:
          enabled: true
          model: "presidio"
          mode: "fast"
          entities: ["US_SSN", "EMAIL_ADDRESS"]
          if_detected: "flag"
        word_filter:
          enabled: true
          entities: ["confidential", "proprietary"]
          if_detected: "flag"
        jailbreak:
          enabled: true
          model: "prompt_guard_22m"
          threshold: 0.5
          if_detected: "block"
      output:
        pii_phi:
          enabled: true
        word_filter:
          enabled: true
          entities: ["system prompt", "secret"]
    sampling:
      l1_rate: 1.0
      l2_rate: 0.0
    attestation:
      offline: true
      service_id: "my-service"
    evidence_storage:
      backend: "sqlite"
      path: "~/.glacis"

Usage:
    from glacis.config import load_config

    config = load_config()                      # Auto-load ./glacis.yaml
    config = load_config("path/to/glacis.yaml") # Explicit path
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from glacis.judges.config import JudgesConfig

# ---------------------------------------------------------------------------
# Per-control config models
# ---------------------------------------------------------------------------

class PiiPhiControlConfig(BaseModel):
    """PII/PHI control configuration (used in both input and output stages)."""

    enabled: bool = Field(default=False, description="Enable PII/PHI scanning")
    entities: list[str] = Field(
        default_factory=list,
        description=(
            "Entity types to scan for (e.g. 'US_SSN', 'EMAIL_ADDRESS'). "
            "Empty = all HIPAA entities."
        ),
    )
    model: str = Field(
        default="presidio",
        description="Detection model/engine identifier",
    )
    mode: Literal["fast", "full"] = Field(
        default="fast",
        description="Scanning mode: 'fast' (regex-only) or 'full' (regex + NER)",
    )
    if_detected: Literal["forward", "flag", "block"] = Field(
        default="flag",
        description=(
            "Behavior when PII/PHI is detected: "
            "forward (observe), flag (log + continue), or block (halt)"
        ),
    )


class WordFilterControlConfig(BaseModel):
    """Word filter control configuration."""

    enabled: bool = Field(default=False, description="Enable word filter")
    entities: list[str] = Field(
        default_factory=list,
        description="Literal terms to match (case-insensitive)",
    )
    if_detected: Literal["forward", "flag", "block"] = Field(
        default="flag",
        description=(
            "Behavior when term matched: "
            "forward (observe), flag (log + continue), or block (halt)"
        ),
    )


class JailbreakControlConfig(BaseModel):
    """Jailbreak/prompt injection detection configuration."""

    enabled: bool = Field(default=False, description="Enable jailbreak detection")
    model: str = Field(
        default="prompt_guard_22m",
        description="Detection model: 'prompt_guard_22m' or 'prompt_guard_86m'",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold (0-1)",
    )
    if_detected: Literal["forward", "flag", "block"] = Field(
        default="flag",
        description=(
            "Behavior when jailbreak detected: "
            "forward (observe), flag (log + continue), or block (halt)"
        ),
    )


# ---------------------------------------------------------------------------
# Stage configs
# ---------------------------------------------------------------------------

class InputControlsConfig(BaseModel):
    """Controls applied to input text before LLM call."""

    pii_phi: PiiPhiControlConfig = Field(default_factory=PiiPhiControlConfig)
    word_filter: WordFilterControlConfig = Field(default_factory=WordFilterControlConfig)
    jailbreak: JailbreakControlConfig = Field(default_factory=JailbreakControlConfig)


class OutputControlsConfig(BaseModel):
    """Controls applied to output text after LLM call."""

    pii_phi: PiiPhiControlConfig = Field(default_factory=PiiPhiControlConfig)
    word_filter: WordFilterControlConfig = Field(default_factory=WordFilterControlConfig)
    jailbreak: JailbreakControlConfig = Field(default_factory=JailbreakControlConfig)


class ControlsConfig(BaseModel):
    """Top-level controls configuration with input/output stages."""

    output_block_action: Literal["block", "forward"] = Field(
        default="block",
        description=(
            "Behavior when an output control blocks: "
            "'block' = raise GlacisBlockedError (response withheld), "
            "'forward' = return response but mark determination as blocked."
        ),
    )
    input: InputControlsConfig = Field(default_factory=InputControlsConfig)
    output: OutputControlsConfig = Field(default_factory=OutputControlsConfig)


# ---------------------------------------------------------------------------
# Non-control config models (unchanged)
# ---------------------------------------------------------------------------

class PolicyConfig(BaseModel):
    """Policy metadata included in attestations."""

    id: str = Field(default="default", description="Policy identifier")
    version: str = Field(default="1.0", description="Policy version")
    environment: str = Field(
        default="development", description="Environment (e.g., 'production', 'staging')",
    )
    tags: list[str] = Field(default_factory=list, description="Custom tags")


class AttestationConfig(BaseModel):
    """Attestation settings."""

    offline: bool = Field(default=True, description="Use offline mode (local signing)")
    service_id: str = Field(default="openai", description="Service identifier")


class SamplingConfig(BaseModel):
    """Sampling configuration for attestation tiers.

    Controls the probability of promoting attestations to higher tiers:
    - L0: Control plane results only (always collected)
    - L1: Evidence collection (input/output payloads, judge review)
    - L2: Deep inspection (extended analysis)

    Sampling is deterministic and auditor-reproducible via HMAC-SHA256.
    """

    l1_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Probability of L1 sampling (evidence collection / judge review). "
            "1.0 = review all."
        ),
    )
    l2_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of L2 sampling (deep inspection, implies L1). 0.0 = disabled.",
    )

    @model_validator(mode="after")
    def _l2_within_l1(self) -> "SamplingConfig":
        if self.l2_rate > self.l1_rate:
            raise ValueError(
                f"l2_rate ({self.l2_rate}) must be <= l1_rate ({self.l1_rate}): "
                f"L2 sampling is a subset of L1"
            )
        return self


class EvidenceStorageConfig(BaseModel):
    """Evidence storage backend configuration."""

    backend: Literal["sqlite", "json"] = Field(
        default="sqlite",
        description=(
            "Evidence storage backend: 'sqlite' (default) "
            "or 'json' (JSONL append-only log)"
        ),
    )
    path: Optional[str] = Field(
        default=None,
        description=(
            "Storage path. For sqlite: .db file path. "
            "For json: directory containing .jsonl files. "
            "Default: ~/.glacis"
        ),
    )


class GlacisConfig(BaseModel):
    """Root configuration model for GLACIS."""

    version: str = Field(default="1.3", description="Config schema version")
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    controls: ControlsConfig = Field(default_factory=ControlsConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    judges: JudgesConfig = Field(default_factory=JudgesConfig)
    attestation: AttestationConfig = Field(default_factory=AttestationConfig)
    evidence_storage: EvidenceStorageConfig = Field(default_factory=EvidenceStorageConfig)

    @property
    def storage(self) -> EvidenceStorageConfig:
        """Alias for evidence_storage."""
        return self.evidence_storage


def load_config(path: Optional[str] = None) -> GlacisConfig:
    """
    Load configuration from a YAML file.

    Args:
        path: Explicit path to config file. If None, looks for ./glacis.yaml

    Returns:
        GlacisConfig instance (defaults if no config file found)
    """
    if path:
        config_path = Path(path)
    else:
        config_path = Path("glacis.yaml")

    if config_path.exists():
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "pyyaml is required for config file support. "
                "Install with: pip install pyyaml"
            )

        with open(config_path) as f:
            data = yaml.safe_load(f)

        if data:
            return GlacisConfig(**data)

    return GlacisConfig()
