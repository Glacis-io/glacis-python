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
from dataclasses import dataclass
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
SDK_VERSION = "0.5.0"

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
    policy_key: Optional[bytes] = None,
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
        policy_key: 32-byte HMAC key for sampling (falls back to signing_seed)

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
    if policy_key:
        extra_kwargs["policy_key"] = policy_key

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
                    id=f"glacis-{stage}-{result.control_type}",
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


def build_metadata(
    provider: str,
    model: str,
    custom_metadata: Optional[dict[str, str]] = None,
    **extra: str,
) -> dict[str, Any]:
    """Build metadata dict with provider defaults and optional custom fields.

    Provider-managed keys ('provider', 'model') cannot be overridden by
    custom_metadata.

    Args:
        provider: LLM provider name (e.g. "openai", "anthropic", "gemini").
        model: LLM model name.
        custom_metadata: Optional customer-provided metadata to merge in.
        **extra: Additional provider-managed keys (e.g. blocked="True").

    Returns:
        Merged metadata dict.

    Raises:
        ValueError: If custom_metadata tries to override 'provider' or 'model'.
    """
    base: dict[str, Any] = {"provider": provider, "model": model, **extra}
    if custom_metadata:
        reserved = {"provider", "model"}
        for key in reserved:
            if key in custom_metadata:
                raise ValueError(f"Cannot override reserved metadata key: '{key}'")
        base.update(custom_metadata)
    return base


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
    custom_metadata: Optional[dict[str, str]] = None,
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
        custom_metadata: Optional customer-provided metadata to merge in.
    """
    output_data = {"blocked": True, "reason": f"{blocking_control_type}_detected"}
    metadata = build_metadata(provider, model, custom_metadata, blocked=str(True))

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


# ---------------------------------------------------------------------------
# Consolidated integration helpers
# ---------------------------------------------------------------------------


@dataclass
class IntegrationContext:
    """Shared state for an attested integration closure.

    Bundles the ~10 variables that every provider integration closure needs,
    so they can be passed as a single object to the pipeline helpers.
    """

    glacis: "Glacis"
    cfg: "GlacisConfig"
    controls_runner: Optional["ControlsRunner"]
    effective_service_id: str
    storage_backend: Optional[str]
    storage_path: Optional[str]
    output_block_action: str
    custom_metadata: Optional[dict[str, str]]
    debug: bool


def setup_integration(
    config: Optional[str],
    offline: Optional[bool],
    glacis_api_key: Optional[str],
    glacis_base_url: str,
    default_service_id: str,
    service_id: str,
    debug: bool,
    signing_seed: Optional[bytes],
    policy_key: Optional[bytes],
    input_controls: Optional[list["BaseControl"]],
    output_controls: Optional[list["BaseControl"]],
    metadata: Optional[dict[str, str]],
) -> IntegrationContext:
    """Initialize config, controls, and Glacis client for an integration.

    Replaces the identical ~30-line setup block in each factory function.

    Returns:
        IntegrationContext with all closure state needed by the
        attested wrapper.
    """
    cfg, effective_offline, effective_service_id = initialize_config(
        config_path=config,
        offline=offline,
        glacis_api_key=glacis_api_key,
        default_service_id=default_service_id,
        service_id=service_id,
    )
    controls_runner = create_controls_runner(
        cfg, debug,
        input_controls=input_controls,
        output_controls=output_controls,
    )
    glacis_client = create_glacis_client(
        offline=effective_offline,
        signing_seed=signing_seed,
        glacis_api_key=glacis_api_key,
        glacis_base_url=glacis_base_url,
        debug=debug,
        storage_backend=cfg.evidence_storage.backend,
        storage_path=cfg.evidence_storage.path,
        sampling_config=cfg.sampling,
        policy_key=policy_key,
    )
    return IntegrationContext(
        glacis=glacis_client,
        cfg=cfg,
        controls_runner=controls_runner,
        effective_service_id=effective_service_id,
        storage_backend=cfg.evidence_storage.backend,
        storage_path=cfg.evidence_storage.path,
        output_block_action=cfg.controls.output_block_action,
        custom_metadata=metadata,
        debug=debug,
    )


def check_input_block(
    ctx: IntegrationContext,
    accumulator: ControlResultsAccumulator,
    model: str,
    provider: str,
    input_data: dict[str, Any],
    system_prompt_hash: Optional[str] = None,
    temperature: Optional[float] = None,
) -> None:
    """Check if input controls want to block, and raise GlacisBlockedError if so.

    No-op if the accumulator has not flagged a block.
    """
    if not accumulator.should_block:
        return
    cpr = create_control_plane_results(
        accumulator, ctx.cfg, model, provider,
        system_prompt_hash=system_prompt_hash,
        temperature=temperature,
    )
    blocking = accumulator.get_blocking_control()
    handle_blocked_request(
        glacis_client=ctx.glacis,
        service_id=ctx.effective_service_id,
        input_data=input_data,
        control_plane_results=cpr,
        provider=provider,
        model=model,
        blocking_control_type=blocking.type if blocking else "unknown",
        blocking_score=blocking.score if blocking else None,
        debug=ctx.debug,
        storage_backend=ctx.storage_backend,
        storage_path=ctx.storage_path,
        custom_metadata=ctx.custom_metadata,
    )


def check_output_block(
    ctx: IntegrationContext,
    accumulator: ControlResultsAccumulator,
    model: str,
    provider: str,
    input_data: dict[str, Any],
    control_plane_results: "ControlPlaneResults",
) -> None:
    """Check if output controls want to block, and raise GlacisBlockedError if so.

    Respects the ``output_block_action`` setting: only blocks when set to
    ``"block"``; ``"forward"`` lets the response through (determination will
    still show ``blocked`` in the attestation).
    """
    if not (
        accumulator.should_block
        and ctx.controls_runner
        and ctx.controls_runner.has_output_controls
    ):
        return
    if ctx.output_block_action != "block":
        return
    blocking = accumulator.get_blocking_control()
    handle_blocked_request(
        glacis_client=ctx.glacis,
        service_id=ctx.effective_service_id,
        input_data=input_data,
        control_plane_results=control_plane_results,
        provider=provider,
        model=model,
        blocking_control_type=blocking.type if blocking else "unknown",
        blocking_score=blocking.score if blocking else None,
        debug=ctx.debug,
        storage_backend=ctx.storage_backend,
        storage_path=ctx.storage_path,
        custom_metadata=ctx.custom_metadata,
    )


def attest_and_store(
    ctx: IntegrationContext,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    metadata: dict[str, Any],
    control_plane_results: "ControlPlaneResults",
) -> None:
    """Attest and store evidence, silently handling errors.

    Combines ``glacis.attest()`` + ``set_last_receipt()`` + ``store_evidence()``
    in a single call.  Errors are swallowed (printed in debug mode) so the LLM
    response is never lost due to attestation failures.
    """
    try:
        receipt = ctx.glacis.attest(
            service_id=ctx.effective_service_id,
            operation_type="completion",
            input=input_data,
            output=output_data,
            metadata=metadata,
            control_plane_results=control_plane_results,
        )
        set_last_receipt(receipt)
        store_evidence(
            receipt=receipt,
            service_id=ctx.effective_service_id,
            operation_type="completion",
            input_data=input_data,
            output_data=output_data,
            control_plane_results=control_plane_results,
            metadata=metadata,
            debug=ctx.debug,
            storage_backend=ctx.storage_backend,
            storage_path=ctx.storage_path,
        )
    except Exception as e:
        if ctx.debug:
            print(f"[glacis] Attestation failed: {e}")


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
    "build_metadata",
    "handle_blocked_request",
    "IntegrationContext",
    "setup_integration",
    "check_input_block",
    "check_output_block",
    "attest_and_store",
]
