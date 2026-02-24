"""
GLACIS integration base module.

Provides shared functionality for all provider integrations:
- GlacisBlockedError exception
- Thread-local receipt storage
- Evidence retrieval
- Logger suppression
- Config and client initialization helpers
- Generic ControlResultsAccumulator for staged pipeline
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

if TYPE_CHECKING:
    from glacis import Glacis
    from glacis.config import GlacisConfig, SamplingConfig
    from glacis.controls import ControlsRunner, StageResult
    from glacis.controls.base import BaseControl
    from glacis.models import (
        Attestation,
        ControlExecution,
        ControlPlaneResults,
        ControlType,
    )


# SDK version used in ControlExecution records
SDK_VERSION = "0.4.0"

# Known control types that map directly to ControlType enum
_KNOWN_CONTROL_TYPES = frozenset({
    "content_safety", "pii", "jailbreak", "topic",
    "prompt_security", "grounding", "word_filter", "custom",
})

# Thread-local storage for the last receipt
_thread_local = threading.local()


class GlacisBlockedError(Exception):
    """Raised when a control blocks the request."""

    def __init__(self, message: str, control_type: str, score: Optional[float] = None):
        super().__init__(message)
        self.control_type = control_type
        self.score = score


def get_last_receipt() -> Optional["Attestation"]:
    """
    Get the last attestation from the current thread.

    Returns:
        The last Attestation, or None if no attestation has been made in this thread.
    """
    return getattr(_thread_local, "last_receipt", None)


def set_last_receipt(receipt: "Attestation") -> None:
    """Store the last receipt in thread-local storage."""
    _thread_local.last_receipt = receipt


def get_evidence(
    attestation_id: str,
    storage_backend: Optional[str] = None,
    storage_path: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Get the full evidence for an attestation by ID.

    Evidence includes the full input, output, and control_plane_results that
    were attested. This data is stored locally and never sent to GLACIS servers.

    Args:
        attestation_id: The attestation ID (att_xxx or oatt_xxx)
        storage_backend: Backend type override ("sqlite" or "json")
        storage_path: Storage path override

    Returns:
        Dict with input, output, control_plane_results, and metadata,
        or None if not found
    """
    from glacis.storage import create_storage

    storage = create_storage(
        backend=storage_backend or "sqlite",
        path=Path(storage_path) if storage_path else None,
    )
    return storage.get_evidence(attestation_id)


# Loggers to suppress for clean customer experience
NOISY_LOGGERS = [
    "glacis",
    "presidio-analyzer",
    "presidio_analyzer",
    "spacy",
    "httpx",
    "httpcore",
    "httpcore.http11",
    "httpcore.connection",
    "transformers",
    "urllib3",
    "urllib3.connectionpool",
    "huggingface_hub",
    "filelock",
]


def suppress_noisy_loggers(provider_loggers: list[str] | None = None) -> None:
    """
    Suppress noisy third-party loggers for clean customer experience.

    Args:
        provider_loggers: Additional provider-specific loggers to suppress
    """
    loggers = NOISY_LOGGERS.copy()
    if provider_loggers:
        loggers.extend(provider_loggers)

    for logger_name in loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def initialize_config(
    config_path: Optional[str],
    offline: Optional[bool],
    glacis_api_key: Optional[str],
    default_service_id: str,
    service_id: str,
) -> tuple["GlacisConfig", bool, str]:
    """
    Initialize and configure Glacis settings.

    Args:
        config_path: Path to glacis.yaml config file
        offline: Offline mode override
        glacis_api_key: Glacis API key (implies online mode if provided)
        default_service_id: Default service ID for this provider
        service_id: User-provided service ID

    Returns:
        Tuple of (config, effective_offline, effective_service_id)
    """
    from glacis.config import load_config

    cfg: GlacisConfig = load_config(config_path)

    # Determine offline mode
    if offline is not None:
        effective_offline = offline
    elif glacis_api_key:
        effective_offline = False
    else:
        effective_offline = cfg.attestation.offline

    # Determine service ID
    effective_service_id = (
        service_id if service_id != default_service_id else cfg.attestation.service_id
    )

    return cfg, effective_offline, effective_service_id


def create_glacis_client(
    offline: bool,
    signing_seed: Optional[bytes],
    glacis_api_key: Optional[str],
    glacis_base_url: str,
    debug: bool,
    storage_backend: Optional[str] = None,
    storage_path: Optional[str] = None,
    sampling_config: Optional["SamplingConfig"] = None,
) -> "Glacis":
    """
    Create a Glacis client (online or offline).

    Args:
        offline: Whether to use offline mode
        signing_seed: 32-byte signing seed (required for offline)
        glacis_api_key: API key (required for online)
        glacis_base_url: Base URL for Glacis API
        debug: Enable debug logging
        storage_backend: Storage backend type ("sqlite" or "json")
        storage_path: Storage path override
        sampling_config: Sampling configuration (l1_rate, l2_rate)

    Returns:
        Configured Glacis client

    Raises:
        ValueError: If required parameters are missing
    """
    from glacis import Glacis

    extra_kwargs: dict[str, Any] = {}
    if storage_backend:
        extra_kwargs["storage_backend"] = storage_backend
    if storage_path:
        extra_kwargs["storage_path"] = Path(storage_path)
    if sampling_config:
        extra_kwargs["sampling_config"] = sampling_config

    if offline:
        if not signing_seed:
            raise ValueError("signing_seed is required for offline mode")
        return Glacis(
            mode="offline",
            signing_seed=signing_seed,
            debug=debug,
            **extra_kwargs,
        )
    else:
        if not glacis_api_key:
            raise ValueError("glacis_api_key is required for online mode")
        return Glacis(
            api_key=glacis_api_key,
            base_url=glacis_base_url,
            debug=debug,
            **extra_kwargs,
        )


def create_controls_runner(
    cfg: "GlacisConfig",
    debug: bool,
    input_controls: Optional[list["BaseControl"]] = None,
    output_controls: Optional[list["BaseControl"]] = None,
) -> Optional["ControlsRunner"]:
    """
    Create controls runner if any control is enabled.

    Args:
        cfg: Glacis configuration
        debug: Enable debug logging
        input_controls: Custom controls for input stage
        output_controls: Custom controls for output stage

    Returns:
        ControlsRunner if any control is enabled, None otherwise
    """
    input_cfg = cfg.controls.input
    output_cfg = cfg.controls.output

    has_builtin = (
        input_cfg.pii_phi.enabled
        or input_cfg.word_filter.enabled
        or input_cfg.jailbreak.enabled
        or output_cfg.pii_phi.enabled
        or output_cfg.word_filter.enabled
        or output_cfg.jailbreak.enabled
    )
    has_custom = bool(input_controls or output_controls)

    if not has_builtin and not has_custom:
        return None

    from glacis.controls import ControlsRunner

    return ControlsRunner(
        input_config=input_cfg,
        output_config=output_cfg,
        input_controls=input_controls,
        output_controls=output_controls,
        debug=debug,
    )


def store_evidence(
    receipt: "Attestation",
    service_id: str,
    operation_type: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    control_plane_results: Any = None,
    metadata: Optional[dict[str, Any]] = None,
    debug: bool = False,
    storage_backend: Optional[str] = None,
    storage_path: Optional[str] = None,
) -> None:
    """
    Store attestation evidence locally for audit trail.

    Args:
        receipt: Attestation
        service_id: Service identifier
        operation_type: Type of operation
        input_data: Input payload (original, pre-controls)
        output_data: Output payload (effective, post-controls)
        control_plane_results: Control plane results (dict or typed model)
        metadata: Additional metadata
        debug: Enable debug logging
        storage_backend: Backend type override ("sqlite" or "json")
        storage_path: Storage path override
    """
    from glacis.storage import create_storage

    storage = create_storage(
        backend=storage_backend or "sqlite",
        path=Path(storage_path) if storage_path else None,
    )

    sampling_level = "L0"
    if receipt.sampling_decision:
        sampling_level = receipt.sampling_decision.level

    storage.store_evidence(
        attestation_id=receipt.id,
        attestation_hash=receipt.evidence_hash,
        mode="offline" if receipt.is_offline else "online",
        service_id=service_id,
        operation_type=operation_type,
        timestamp=receipt.timestamp or 0,
        input_data=input_data,
        output_data=output_data,
        control_plane_results=control_plane_results,
        metadata=metadata,
        sampling_level=sampling_level,
    )
    if debug:
        print(f"[glacis] Attestation created: {receipt.id}")


# ---------------------------------------------------------------------------
# Generic Control Results Accumulator
# ---------------------------------------------------------------------------

def _map_control_type(control_type: str) -> str:
    """Map a control_type string to a ControlType enum value.

    Known types pass through; unknown types map to "custom".
    """
    return control_type if control_type in _KNOWN_CONTROL_TYPES else "custom"


class ControlResultsAccumulator:
    """Accumulates results from staged control pipeline runs.

    Fully generic — no hardcoded control type handling. Works with any
    control type (built-in or custom).
    """

    def __init__(self) -> None:
        self.control_executions: list["ControlExecution"] = []
        self.should_block: bool = False
        self.effective_input_text: Optional[str] = None
        self.effective_output_text: Optional[str] = None

    def update_from_stage(
        self, stage_result: "StageResult", stage: Literal["input", "output"],
    ) -> None:
        """Update accumulator from a StageResult.

        Creates ControlExecution entries for each ControlResult in the stage.
        Direct mapping — result.action becomes ControlExecution.status.

        Args:
            stage_result: Result from ControlsRunner.run_input/run_output.
            stage: Which pipeline stage this came from.
        """
        from glacis.models import ControlExecution

        for result in stage_result.results:
            self.control_executions.append(
                ControlExecution(
                    id=f"glacis-{result.control_type}",
                    type=cast("ControlType", _map_control_type(result.control_type)),
                    version=SDK_VERSION,
                    provider=result.metadata.get("provider", "glacis"),
                    latency_ms=result.latency_ms,
                    status=result.action,
                    score=result.score,
                    stage=stage,
                )
            )

        if stage_result.should_block:
            self.should_block = True

        # Track effective text per stage
        if stage == "input":
            self.effective_input_text = stage_result.effective_text
        else:
            self.effective_output_text = stage_result.effective_text

    def get_blocking_control(self) -> Optional["ControlExecution"]:
        """Find the first control result that caused a block.

        Returns the ControlResult that triggered blocking, or None.
        Used for error reporting.
        """
        # Check the original stage results stored in control_executions
        for ce in self.control_executions:
            if ce.status == "block":
                return ce
        return None


# ---------------------------------------------------------------------------
# Pipeline helper functions for provider integrations
# ---------------------------------------------------------------------------

def run_input_controls(
    runner: "ControlsRunner",
    text: str,
    accumulator: ControlResultsAccumulator,
) -> str:
    """Run input controls and update accumulator.

    Args:
        runner: ControlsRunner with configured controls.
        text: Original input text.
        accumulator: Accumulator to update with results.

    Returns:
        The original text (controls are scan-only).
    """
    stage_result = runner.run_input(text)
    accumulator.update_from_stage(stage_result, "input")
    return stage_result.effective_text


def run_output_controls(
    runner: "ControlsRunner",
    text: str,
    accumulator: ControlResultsAccumulator,
) -> str:
    """Run output controls and update accumulator.

    Args:
        runner: ControlsRunner with configured controls.
        text: LLM response text.
        accumulator: Accumulator to update with results.

    Returns:
        The original text (controls are scan-only).
    """
    stage_result = runner.run_output(text)
    accumulator.update_from_stage(stage_result, "output")
    return stage_result.effective_text


def create_control_plane_results(
    accumulator: ControlResultsAccumulator,
    cfg: "GlacisConfig",
    model: str,
    provider: str,
    system_prompt_hash: Optional[str] = None,
    temperature: Optional[float] = None,
) -> "ControlPlaneResults":
    """Create ControlPlaneResults from accumulated results."""
    from glacis.models import (
        ControlPlaneResults,
        Determination,
        ModelInfo,
        PolicyContext,
    )

    action: Literal["forwarded", "blocked"]
    action = "blocked" if accumulator.should_block else "forwarded"

    return ControlPlaneResults(
        policy=PolicyContext(
            id=cfg.policy.id,
            version=cfg.policy.version,
            model=ModelInfo(
                model_id=model,
                provider=provider,
                system_prompt_hash=system_prompt_hash,
                temperature=temperature,
            ),
            environment=cfg.policy.environment,
            tags=cfg.policy.tags,
        ),
        determination=Determination(action=action),
        controls=accumulator.control_executions,
    )


def handle_blocked_request(
    glacis_client: "Glacis",
    service_id: str,
    input_data: dict[str, Any],
    control_plane_results: Any,
    provider: str,
    model: str,
    blocking_control_type: str,
    blocking_score: Optional[float],
    debug: bool,
    storage_backend: Optional[str] = None,
    storage_path: Optional[str] = None,
) -> None:
    """Attest a blocked request and raise GlacisBlockedError.

    Args:
        glacis_client: Glacis client for attestation.
        service_id: Service identifier.
        input_data: Original input data.
        control_plane_results: Control plane results to include.
        provider: LLM provider name.
        model: LLM model name.
        blocking_control_type: Type of control that caused the block.
        blocking_score: Score from the blocking control (if applicable).
        debug: Enable debug logging.
        storage_backend: Evidence storage backend override.
        storage_path: Evidence storage path override.
    """
    output_data = {"blocked": True, "reason": f"{blocking_control_type}_detected"}
    metadata = {"provider": provider, "model": model, "blocked": str(True)}

    try:
        receipt = glacis_client.attest(
            service_id=service_id,
            operation_type="completion",
            input=input_data,
            output=output_data,
            metadata=metadata,
            control_plane_results=control_plane_results,
        )
        set_last_receipt(receipt)
        store_evidence(
            receipt=receipt,
            service_id=service_id,
            operation_type="completion",
            input_data=input_data,
            output_data=output_data,
            control_plane_results=control_plane_results,
            metadata=metadata,
            debug=debug,
            storage_backend=storage_backend,
            storage_path=storage_path,
        )
    except Exception as e:
        if debug:
            print(f"[glacis] Attestation failed: {e}")

    score_str = f" (score={blocking_score:.2f})" if blocking_score is not None else ""
    raise GlacisBlockedError(
        f"Blocked by {blocking_control_type}{score_str}",
        control_type=blocking_control_type,
        score=blocking_score,
    )


__all__ = [
    "GlacisBlockedError",
    "get_last_receipt",
    "set_last_receipt",
    "get_evidence",
    "suppress_noisy_loggers",
    "initialize_config",
    "create_glacis_client",
    "create_controls_runner",
    "store_evidence",
    "NOISY_LOGGERS",
    "ControlResultsAccumulator",
    "run_input_controls",
    "run_output_controls",
    "create_control_plane_results",
    "handle_blocked_request",
]
