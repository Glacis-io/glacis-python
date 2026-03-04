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
      path: "~/.glacis/glacis.db"

Usage:
    from glacis.config import load_config

    config = load_config()                      # Auto-load ./glacis.yaml
    config = load_config("path/to/glacis.yaml") # Explicit path
"""

import os
import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr, model_validator

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


class ContentSafetyControlConfig(BaseModel):
    """Content safety / toxicity detection configuration.

    Uses HuggingFace toxicity classification models to detect harmful content
    including toxic language, threats, insults, and identity-based hate speech.

    Supported models:
    - toxic-bert: unitary/toxic-bert (multi-label toxicity classifier)

    Categories (when using toxic-bert): toxic, severe_toxic, obscene, threat,
    insult, identity_hate. Empty categories list = check all.
    """

    enabled: bool = Field(default=False, description="Enable content safety scanning")
    model: str = Field(
        default="toxic-bert",
        description="Detection model identifier (e.g., 'toxic-bert')",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold (0-1). Scores above this are flagged.",
    )
    categories: list[str] = Field(
        default_factory=list,
        description=(
            "Toxicity categories to check (e.g., 'toxic', 'threat', 'insult'). "
            "Empty = all categories."
        ),
    )
    if_detected: Literal["forward", "flag", "block"] = Field(
        default="flag",
        description=(
            "Behavior when harmful content detected: "
            "forward (observe), flag (log + continue), or block (halt)"
        ),
    )


class TopicControlConfig(BaseModel):
    """Topic enforcement / off-topic detection configuration.

    Uses keyword matching to enforce topic boundaries. Supports two modes:
    - Blocklist: Flag text that matches any blocked topic
    - Allowlist: Flag text that doesn't match any allowed topic (off-topic)

    When both are configured, blocked topics are checked first.
    """

    enabled: bool = Field(default=False, description="Enable topic enforcement")
    allowed_topics: list[str] = Field(
        default_factory=list,
        description=(
            "Allowed topic keywords (case-insensitive). "
            "If non-empty, text must match at least one to pass."
        ),
    )
    blocked_topics: list[str] = Field(
        default_factory=list,
        description=(
            "Blocked topic keywords (case-insensitive). "
            "If any match, the text is flagged."
        ),
    )
    if_detected: Literal["forward", "flag", "block"] = Field(
        default="flag",
        description=(
            "Behavior when topic violation detected: "
            "forward (observe), flag (log + continue), or block (halt)"
        ),
    )


class PromptSecurityControlConfig(BaseModel):
    """Prompt extraction / system prompt leakage detection configuration.

    Detects attempts to extract system prompts, override instructions, or
    manipulate LLM behavior through prompt injection patterns. Ships with
    built-in patterns; additional custom patterns can be added.
    """

    enabled: bool = Field(default=False, description="Enable prompt security scanning")
    patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Additional regex patterns to detect (case-insensitive). "
            "These are added to the built-in pattern set."
        ),
    )
    if_detected: Literal["forward", "flag", "block"] = Field(
        default="block",
        description=(
            "Behavior when prompt extraction detected: "
            "forward (observe), flag (log + continue), or block (halt). "
            "Defaults to 'block' for security."
        ),
    )


class GroundingControlConfig(BaseModel):
    """Grounding / hallucination detection configuration.

    Note: The built-in implementation is a pass-through stub because the
    ``check(text)`` interface doesn't receive reference text for comparison.
    For real grounding validation, use the ``custom`` section with a
    ``BaseControl`` subclass that accepts ``reference_text`` in its constructor.
    """

    enabled: bool = Field(default=False, description="Enable grounding control")
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Grounding score threshold (for custom implementations)",
    )
    if_detected: Literal["forward", "flag", "block"] = Field(
        default="flag",
        description=(
            "Behavior when grounding issue detected: "
            "forward (observe), flag (log + continue), or block (halt)"
        ),
    )


class CustomControlEntry(BaseModel):
    """Configuration for a single custom control loaded from a Python module.

    The control class must be a ``BaseControl`` subclass. It is imported at
    startup and instantiated with the ``args`` dict as keyword arguments.

    Example YAML::

        custom:
          - path: "my_controls.ToxicityControl"
            enabled: true
            if_detected: "flag"
            args:
              api_key: "${MY_API_KEY}"
              threshold: 0.8
    """

    path: str = Field(
        description=(
            "Dot-separated import path: 'module.ClassName'. "
            "The module is resolved relative to the YAML file's directory."
        ),
    )
    enabled: bool = Field(default=True, description="Enable this control")
    if_detected: Literal["forward", "flag", "block"] = Field(
        default="flag",
        description="Action when the control detects an issue",
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Constructor kwargs passed to the control class",
    )


# ---------------------------------------------------------------------------
# Stage configs
# ---------------------------------------------------------------------------

class InputControlsConfig(BaseModel):
    """Controls applied to input text before LLM call."""

    pii_phi: PiiPhiControlConfig = Field(default_factory=PiiPhiControlConfig)
    word_filter: WordFilterControlConfig = Field(default_factory=WordFilterControlConfig)
    jailbreak: JailbreakControlConfig = Field(default_factory=JailbreakControlConfig)
    content_safety: ContentSafetyControlConfig = Field(default_factory=ContentSafetyControlConfig)
    topic: TopicControlConfig = Field(default_factory=TopicControlConfig)
    prompt_security: PromptSecurityControlConfig = Field(
        default_factory=PromptSecurityControlConfig
    )
    grounding: GroundingControlConfig = Field(default_factory=GroundingControlConfig)
    custom: list[CustomControlEntry] = Field(
        default_factory=list,
        description="Custom controls loaded from Python modules",
    )


class OutputControlsConfig(BaseModel):
    """Controls applied to output text after LLM call."""

    pii_phi: PiiPhiControlConfig = Field(default_factory=PiiPhiControlConfig)
    word_filter: WordFilterControlConfig = Field(default_factory=WordFilterControlConfig)
    jailbreak: JailbreakControlConfig = Field(default_factory=JailbreakControlConfig)
    content_safety: ContentSafetyControlConfig = Field(default_factory=ContentSafetyControlConfig)
    topic: TopicControlConfig = Field(default_factory=TopicControlConfig)
    prompt_security: PromptSecurityControlConfig = Field(
        default_factory=PromptSecurityControlConfig
    )
    grounding: GroundingControlConfig = Field(default_factory=GroundingControlConfig)
    custom: list[CustomControlEntry] = Field(
        default_factory=list,
        description="Custom controls loaded from Python modules",
    )


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
    - L1: Evidence collection (input/output payloads retained locally)
    - L2: Deep inspection (flagged for judge evaluation, implies L1)

    Sampling is deterministic and auditor-reproducible via HMAC-SHA256.
    """

    l1_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Probability of L1 sampling (evidence collection). "
            "1.0 = collect all."
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
            "Storage path. For sqlite: full .db file path "
            "(default: ~/.glacis/glacis.db). "
            "For json: directory containing .jsonl files "
            "(default: ~/.glacis)."
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

    # Set by load_config() — directory of the YAML file, used for module resolution
    _config_dir: Optional[str] = PrivateAttr(default=None)

    @property
    def storage(self) -> EvidenceStorageConfig:
        """Alias for evidence_storage."""
        return self.evidence_storage


# ---------------------------------------------------------------------------
# Environment variable substitution
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively replace ``${VAR}`` placeholders with environment variable values.

    Supports strings, dicts, and lists. Non-string/collection values pass through
    unchanged. Raises ``ValueError`` if a referenced variable is not set.
    """
    if isinstance(obj, str):
        def _replace(m: "re.Match[str]") -> str:
            var = m.group(1)
            val = os.environ.get(var)
            if val is None:
                raise ValueError(
                    f"Environment variable '{var}' is not set. "
                    f"Referenced in glacis.yaml via ${{{var}}}. "
                    f"Set the variable or remove the ${{{var}}} reference."
                )
            return val
        return _ENV_PATTERN.sub(_replace, obj)
    elif isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: Optional[str] = None) -> GlacisConfig:
    """
    Load configuration from a YAML file.

    Supports ``${ENV_VAR}`` substitution in all string values — useful for
    passing API keys and secrets to custom controls without hard-coding them.

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
            data = _substitute_env_vars(data)
            cfg = GlacisConfig(**data)
            cfg._config_dir = str(config_path.parent.resolve())
            return cfg

    return GlacisConfig()
