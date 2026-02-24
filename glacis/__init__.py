"""
GLACIS SDK for Python

AI Compliance Attestation - hash locally, prove globally.

Example (online):
    >>> from glacis import Glacis
    >>> glacis = Glacis(api_key="glsk_live_xxx")
    >>> receipt = glacis.attest(
    ...     service_id="my-ai-service",
    ...     operation_type="inference",
    ...     input={"prompt": "Hello, world!"},
    ...     output={"response": "Hi there!"},
    ... )
    >>> print(f"Attestation ID: {receipt.id}")

Example (offline):
    >>> from glacis import Glacis
    >>> import os
    >>> glacis = Glacis(mode="offline", signing_seed=os.urandom(32))
    >>> receipt = glacis.attest(...)  # Returns Attestation
    >>> result = glacis.verify(receipt)  # witness_status="UNVERIFIED"

Async Example:
    >>> from glacis import AsyncGlacis
    >>> glacis = AsyncGlacis(api_key="glsk_live_xxx")
    >>> receipt = await glacis.attest(...)

Controls Example:
    >>> from glacis.controls import ControlsRunner, PIIControl, WordFilterControl
    >>> from glacis.config import load_config
    >>> cfg = load_config()  # Loads glacis.yaml
    >>> runner = ControlsRunner(input_config=cfg.controls.input,
    ...                         output_config=cfg.controls.output)
    >>> result = runner.run_input("Patient SSN: 123-45-6789")
    >>> result.effective_text  # always equals original text (scan-only)
"""

from glacis.client import AsyncGlacis, Glacis, GlacisMode, OperationContext
from glacis.crypto import canonical_json, hash_payload
from glacis.models import (
    Attestation,
    AttestationMetadata,
    AttestInput,
    AttestReceipt,  # Deprecated alias for Attestation
    ControlExecution,
    ControlPlaneResults,
    ControlStatus,
    ControlType,
    DeepInspection,  # Deprecated alias for Review
    Determination,
    Evidence,
    GlacisApiError,
    GlacisConfig,
    InclusionProof,
    LogEntry,
    LogQueryParams,
    LogQueryResult,
    MerkleInclusionProof,  # Deprecated alias for InclusionProof
    ModelInfo,
    OfflineAttestReceipt,  # Deprecated alias for Attestation
    OfflineVerifyResult,
    PolicyContext,
    Receipt,
    Review,
    SamplingDecision,
    SignedTreeHead,
    VerifyResult,
)
from glacis.storage import (
    JsonStorageBackend,
    ReceiptStorage,
    StorageBackend,
    create_storage,
)

# Controls module (optional dependencies for individual controls)
try:
    from glacis.controls import (  # noqa: F401
        BaseControl,
        ControlAction,
        ControlResult,
        ControlsRunner,
        JailbreakControl,
        PIIControl,
        StageResult,
        WordFilterControl,
    )

    _CONTROLS_AVAILABLE = True
except ImportError:
    _CONTROLS_AVAILABLE = False

__version__ = "0.5.0"

__all__ = [
    # Main classes
    "Glacis",
    "AsyncGlacis",
    "GlacisMode",
    "OperationContext",
    # Exceptions
    "GlacisApiError",
    # Evidence Storage
    "StorageBackend",
    "ReceiptStorage",
    "JsonStorageBackend",
    "create_storage",
    # Core models (v1.2)
    "Attestation",
    "Receipt",
    "GlacisConfig",
    "AttestInput",
    "AttestationMetadata",
    "VerifyResult",
    "OfflineVerifyResult",
    "LogQueryParams",
    "LogQueryResult",
    "LogEntry",
    "InclusionProof",
    "MerkleInclusionProof",  # Deprecated alias
    "SignedTreeHead",
    # Control Plane (L0)
    "ControlPlaneResults",
    "PolicyContext",
    "ModelInfo",
    "Determination",
    "ControlExecution",
    "ControlType",
    "ControlStatus",
    "SamplingDecision",
    # L1/L2
    "Evidence",
    "Review",
    # Deprecated aliases (one release)
    "AttestReceipt",
    "OfflineAttestReceipt",
    "DeepInspection",
    # Crypto utilities
    "canonical_json",
    "hash_payload",
]

# Add controls exports if available
if _CONTROLS_AVAILABLE:
    __all__.extend([
        "BaseControl",
        "ControlAction",
        "ControlResult",
        "ControlsRunner",
        "PIIControl",
        "JailbreakControl",
        "WordFilterControl",
        "StageResult",
    ])
