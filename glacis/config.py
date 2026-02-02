"""
GLACIS Configuration Module

Provides configuration models and loading for glacis.yaml config files.
The config file allows toggling controls (PII/PHI redaction, etc.) on/off.

Example glacis.yaml:
    version: "1.0"
    policy:
      id: "hipaa-safe-harbor"
      tenant_id: "my-org"
    controls:
      pii_phi:
        enabled: true
        backend: "presidio"
        mode: "fast"
      jailbreak:
        enabled: true
        backend: "prompt_guard_22m"
        threshold: 0.5
        action: "flag"
    attestation:
      offline: true
      service_id: "my-service"

Usage:
    from glacis.config import load_config

    # Auto-load from ./glacis.yaml
    config = load_config()

    # Or explicit path
    config = load_config("path/to/glacis.yaml")
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PiiPhiConfig(BaseModel):
    """PII/PHI redaction control configuration."""

    enabled: bool = Field(default=False, description="Enable PII/PHI redaction")
    backend: str = Field(
        default="presidio",
        description="Backend model identifier (e.g., 'presidio')",
    )
    mode: Literal["fast", "full"] = Field(
        default="fast",
        description="Redaction mode: 'fast' (regex-only) or 'full' (regex + NER)",
    )


class JailbreakConfig(BaseModel):
    """Jailbreak/prompt injection detection configuration."""

    enabled: bool = Field(default=False, description="Enable jailbreak detection")
    backend: str = Field(
        default="prompt_guard_22m",
        description="Backend model: 'prompt_guard_22m' or 'prompt_guard_86m'",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold (0-1)",
    )
    action: Literal["block", "flag", "log"] = Field(
        default="flag",
        description="Action when jailbreak detected: 'block', 'flag', or 'log'",
    )


class ControlsConfig(BaseModel):
    """Configuration for all controls."""

    pii_phi: PiiPhiConfig = Field(default_factory=PiiPhiConfig)
    jailbreak: JailbreakConfig = Field(default_factory=JailbreakConfig)


class PolicyConfig(BaseModel):
    """Policy metadata included in attestations."""

    id: str = Field(default="default", description="Policy identifier")
    version: str = Field(default="1.0", description="Policy version")
    tenant_id: str = Field(default="default", description="Tenant identifier")


class AttestationConfig(BaseModel):
    """Attestation settings."""

    offline: bool = Field(default=True, description="Use offline mode (local signing)")
    service_id: str = Field(default="openai", description="Service identifier")


class GlacisConfig(BaseModel):
    """Root configuration model for GLACIS."""

    version: str = Field(default="1.0", description="Config schema version")
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    controls: ControlsConfig = Field(default_factory=ControlsConfig)
    attestation: AttestationConfig = Field(default_factory=AttestationConfig)


def load_config(path: Optional[str] = None) -> GlacisConfig:
    """
    Load configuration from a YAML file.

    Args:
        path: Explicit path to config file. If None, looks for ./glacis.yaml

    Returns:
        GlacisConfig instance (defaults if no config file found)

    Example:
        >>> config = load_config()  # Auto-load ./glacis.yaml
        >>> config = load_config("configs/production.yaml")  # Explicit path
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
