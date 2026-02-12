"""
GLACIS integration base module.

Provides shared functionality for all provider integrations:
- GlacisBlockedError exception
- Thread-local receipt storage
- Evidence retrieval
- Logger suppression
- Config and client initialization helpers
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

if TYPE_CHECKING:
    from glacis import Glacis
    from glacis.config import GlacisConfig
    from glacis.controls import ControlsRunner
    from glacis.models import (
        AttestReceipt,
        ControlExecution,
        ControlPlaneResults,
        OfflineAttestReceipt,
    )


# Thread-local storage for the last receipt
_thread_local = threading.local()


class GlacisBlockedError(Exception):
    """Raised when a control blocks the request."""

    def __init__(self, message: str, control_type: str, score: Optional[float] = None):
        super().__init__(message)
        self.control_type = control_type
        self.score = score


def get_last_receipt() -> Optional[Union["AttestReceipt", "OfflineAttestReceipt"]]:
    """
    Get the last attestation receipt from the current thread.

    Returns:
        The last AttestReceipt or OfflineAttestReceipt, or None if no attestation
        has been made in this thread.
    """
    return getattr(_thread_local, "last_receipt", None)


def set_last_receipt(receipt: Union["AttestReceipt", "OfflineAttestReceipt"]) -> None:
    """Store the last receipt in thread-local storage."""
    _thread_local.last_receipt = receipt


def get_evidence(attestation_id: str) -> Optional[dict[str, Any]]:
    """
    Get the full evidence for an attestation by ID.

    Evidence includes the full input, output, and control_plane_results that
    were attested. This data is stored locally and never sent to GLACIS servers.

    Args:
        attestation_id: The attestation ID (att_xxx or oatt_xxx)

    Returns:
        Dict with input, output, control_plane_results, and metadata,
        or None if not found

    Example:
        >>> receipt = get_last_receipt()
        >>> evidence = get_evidence(receipt.id)
        >>> print(evidence["input"]["messages"])
        >>> print(evidence["control_plane_results"]["controls"])
    """
    from glacis.storage import ReceiptStorage

    storage = ReceiptStorage()
    return storage.get_evidence(attestation_id)


# Loggers to suppress for clean customer experience
NOISY_LOGGERS = [
    "glacis",
    "presidio-analyzer",
    "presidio-anonymizer",
    "presidio_analyzer",
    "presidio_anonymizer",
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
    redaction: Union[bool, Literal["fast", "full"], None],
    offline: Optional[bool],
    glacis_api_key: Optional[str],
    default_service_id: str,
    service_id: str,
) -> tuple["GlacisConfig", bool, str]:
    """
    Initialize and configure Glacis settings.

    Args:
        config_path: Path to glacis.yaml config file
        redaction: PII redaction mode override
        offline: Offline mode override
        glacis_api_key: Glacis API key (implies online mode if provided)
        default_service_id: Default service ID for this provider
        service_id: User-provided service ID

    Returns:
        Tuple of (config, effective_offline, effective_service_id)
    """
    from glacis.config import load_config

    cfg: GlacisConfig = load_config(config_path)

    # Handle backward-compatible redaction parameter
    if redaction is not None:
        if redaction is True:
            cfg.controls.pii_phi.enabled = True
            cfg.controls.pii_phi.mode = "fast"
        elif redaction is False:
            cfg.controls.pii_phi.enabled = False
        else:
            cfg.controls.pii_phi.enabled = True
            cfg.controls.pii_phi.mode = redaction

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
) -> "Glacis":
    """
    Create a Glacis client (online or offline).

    Args:
        offline: Whether to use offline mode
        signing_seed: 32-byte signing seed (required for offline)
        glacis_api_key: API key (required for online)
        glacis_base_url: Base URL for Glacis API
        debug: Enable debug logging

    Returns:
        Configured Glacis client

    Raises:
        ValueError: If required parameters are missing
    """
    from glacis import Glacis

    if offline:
        if not signing_seed:
            raise ValueError("signing_seed is required for offline mode")
        return Glacis(
            mode="offline",
            signing_seed=signing_seed,
            debug=debug,
        )
    else:
        if not glacis_api_key:
            raise ValueError("glacis_api_key is required for online mode")
        return Glacis(
            api_key=glacis_api_key,
            base_url=glacis_base_url,
            debug=debug,
        )


def create_controls_runner(
    cfg: "GlacisConfig",
    debug: bool,
) -> Optional["ControlsRunner"]:
    """
    Create controls runner if any control is enabled.

    Args:
        cfg: Glacis configuration
        debug: Enable debug logging

    Returns:
        ControlsRunner if any control is enabled, None otherwise
    """
    if cfg.controls.pii_phi.enabled or cfg.controls.jailbreak.enabled:
        from glacis.controls import ControlsRunner
        return ControlsRunner(cfg.controls, debug=debug)
    return None


def store_evidence(
    receipt: Union["AttestReceipt", "OfflineAttestReceipt"],
    service_id: str,
    operation_type: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    control_plane_results: Optional["ControlPlaneResults"],
    metadata: dict[str, Any],
    debug: bool,
) -> None:
    """
    Store attestation evidence locally for audit trail.

    Args:
        receipt: Attestation receipt
        service_id: Service identifier
        operation_type: Type of operation
        input_data: Input payload
        output_data: Output payload
        control_plane_results: Control plane results
        metadata: Additional metadata
        debug: Enable debug logging
    """
    from glacis.models import OfflineAttestReceipt
    from glacis.storage import ReceiptStorage

    storage = ReceiptStorage()
    # Both receipt types now use evidence_hash field (aligned)
    evidence_hash = receipt.evidence_hash
    # Both receipt types now use id field (aligned)
    attestation_id = receipt.id

    # Extract sampling level from receipt if available
    sampling_level = "L0"
    if hasattr(receipt, "sampling_decision") and receipt.sampling_decision:
        sampling_level = receipt.sampling_decision.level

    storage.store_evidence(
        attestation_id=attestation_id,
        attestation_hash=evidence_hash,
        mode="offline" if isinstance(receipt, OfflineAttestReceipt) else "online",
        service_id=service_id,
        operation_type=operation_type,
        timestamp=receipt.timestamp,
        input_data=input_data,
        output_data=output_data,
        control_plane_results=control_plane_results,
        metadata=metadata,
        sampling_level=sampling_level,
    )
    if debug:
        print(f"[glacis] Attestation created: {attestation_id}")


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
    "process_text_for_controls",
    "create_control_plane_results_from_accumulator",
    "create_control_plane_attestation_from_accumulator",  # backward compat alias
    "handle_blocked_request",
]


# --- Shared Control Execution Logic ---

class ControlResultsAccumulator:
    """Accumulates results from multiple control execution runs.

    Tracks internal state for determining action (forwarded/redacted/blocked).
    PII and jailbreak summaries are now captured in controls[] rather than
    separate summary objects.
    """

    def __init__(self) -> None:
        # Internal state for action determination
        self._pii_detected: bool = False
        self._pii_categories: list[str] = []
        self._jailbreak_detected: bool = False
        self._jailbreak_score: float = 0.0
        self._jailbreak_action: str = "pass"
        self.control_executions: list["ControlExecution"] = []
        self.should_block: bool = False

    def update(self, results: list[Any]) -> None:
        """Update accumulator with check results."""
        from glacis.models import ControlExecution

        for result in results:
            if result.control_type == "pii" and result.detected:
                self._pii_detected = True
                self._pii_categories = sorted(
                    set(self._pii_categories) | set(result.categories)
                )
                self.control_executions.append(
                    ControlExecution(
                        id="glacis-pii-redactor",
                        type="pii",
                        version="0.3.0",
                        provider="glacis",
                        latency_ms=result.latency_ms,
                        status="flag",
                    )
                )

            elif result.control_type == "jailbreak":
                if not self._jailbreak_detected or (result.score or 0) > self._jailbreak_score:
                    self._jailbreak_detected = result.detected
                    self._jailbreak_score = result.score or 0.0
                    self._jailbreak_action = result.action
                self.control_executions.append(
                    ControlExecution(
                        id="glacis-jailbreak-detector",
                        type="jailbreak",
                        version="0.3.0",
                        provider="glacis",
                        latency_ms=result.latency_ms,
                        status=result.action if result.detected else "pass",
                    )
                )
                if result.action == "block":
                    self.should_block = True


def process_text_for_controls(
    runner: "ControlsRunner",
    text: str,
    accumulator: ControlResultsAccumulator
) -> str:
    """Run controls on text, update accumulator, and return (potentially redacted) text."""
    results = runner.run(text)
    accumulator.update(results)
    final_text = runner.get_final_text(results) or text
    return final_text


def create_control_plane_results_from_accumulator(
    accumulator: ControlResultsAccumulator,
    cfg: "GlacisConfig",
    model: str,
    provider: str,
) -> "ControlPlaneResults":
    """Create ControlPlaneResults from accumulated results.

    Note: PII and jailbreak detection results are now captured in controls[]
    rather than separate summary objects. Sampling info moved to Evidence/Review.
    """
    from glacis.models import (
        ControlPlaneResults,
        Determination,
        ModelInfo,
        PolicyContext,
        PolicyScope,
        SafetyScores,
    )

    action: Literal["forwarded", "redacted", "blocked"]
    trigger: Optional[str]
    if accumulator._pii_detected:
        action, trigger = "redacted", "pii"
    elif accumulator._jailbreak_detected:
        action = "blocked" if accumulator._jailbreak_action == "block" else "forwarded"
        trigger = "jailbreak"
    else:
        action, trigger = "forwarded", None

    return ControlPlaneResults(
        policy=PolicyContext(
            id=cfg.policy.id,
            version=cfg.policy.version,
            model=ModelInfo(model_id=model, provider=provider),
            scope=PolicyScope(
                environment=cfg.policy.environment,
                tags=cfg.policy.tags,
            ),
        ),
        determination=Determination(action=action, trigger=trigger, confidence=1.0),
        controls=accumulator.control_executions,
        safety=SafetyScores(
            overall_risk=accumulator._jailbreak_score
        ),
    )


# Keep old name for backward compatibility
create_control_plane_attestation_from_accumulator = create_control_plane_results_from_accumulator


def handle_blocked_request(
    glacis_client: "Glacis",
    service_id: str,
    input_data: dict[str, Any],
    control_plane_results: Any,
    provider: str,
    model: str,
    jailbreak_score: float,
    debug: bool,
) -> None:
    """Attest a blocked request and raise GlacisBlockedError."""
    output_data = {"blocked": True, "reason": "jailbreak_detected"}
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
        )
    except Exception as e:
        if debug:
            print(f"[glacis] Attestation failed: {e}")

    raise GlacisBlockedError(
        f"Jailbreak detected (score={jailbreak_score:.2f})",
        control_type="jailbreak",
        score=jailbreak_score,
    )
