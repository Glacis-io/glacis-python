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
    >>> receipt = glacis.attest(...)  # Returns OfflineAttestReceipt
    >>> result = glacis.verify(receipt)  # witness_status="UNVERIFIED"

Async Example:
    >>> from glacis import AsyncGlacis
    >>> glacis = AsyncGlacis(api_key="glsk_live_xxx")
    >>> receipt = await glacis.attest(...)

Streaming Example:
    >>> from glacis import Glacis
    >>> from glacis.streaming import StreamingSession
    >>> glacis = Glacis(api_key="glsk_live_xxx")
    >>> session = await StreamingSession.start(glacis, {
    ...     "service_id": "voice-assistant",
    ...     "operation_type": "completion",
    ...     "session_do_url": "https://session-do.glacis.io",
    ... })
    >>> await session.attest_chunk(input=audio_chunk, output=transcript)
    >>> receipt = await session.end(metadata={"duration": "00:05:23"})

Controls Example:
    >>> from glacis.controls import ControlsRunner, PIIControl, JailbreakControl
    >>> from glacis.config import load_config
    >>> cfg = load_config()  # Loads glacis.yaml
    >>> runner = ControlsRunner(cfg.controls)
    >>> results = runner.run("Patient SSN: 123-45-6789")
"""

from glacis.client import AsyncGlacis, Glacis, GlacisMode
from glacis.crypto import canonical_json, hash_payload
from glacis.models import (
    AttestationMetadata,
    AttestInput,
    AttestReceipt,
    ControlExecution,
    ControlPlaneResults,
    ControlStatus,
    ControlType,
    DeepInspection,
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
    OfflineAttestReceipt,
    OfflineVerifyResult,
    PolicyContext,
    PolicyScope,
    Review,
    SafetyScores,
    SamplingDecision,
    SignedTreeHead,
    VerifyResult,
)
from glacis.storage import ReceiptStorage
from glacis.streaming import SessionContext, SessionReceipt, StreamingSession

# Controls module (optional dependencies for individual controls)
try:
    from glacis.controls import (  # noqa: F401
        BaseControl,
        ControlResult,
        ControlsRunner,
        JailbreakControl,
        PIIControl,
    )

    _CONTROLS_AVAILABLE = True
except ImportError:
    _CONTROLS_AVAILABLE = False

__version__ = "0.3.0"

__all__ = [
    # Main classes
    "Glacis",
    "AsyncGlacis",
    "GlacisMode",
    # Exceptions
    "GlacisApiError",
    # Streaming
    "StreamingSession",
    "SessionContext",
    "SessionReceipt",
    # Storage (offline mode)
    "ReceiptStorage",
    # Models
    "GlacisConfig",
    "AttestInput",
    "AttestationMetadata",
    "AttestReceipt",
    "OfflineAttestReceipt",
    "VerifyResult",
    "OfflineVerifyResult",
    "LogQueryParams",
    "LogQueryResult",
    "LogEntry",
    "InclusionProof",
    "MerkleInclusionProof",  # Deprecated alias for InclusionProof
    "SignedTreeHead",
    # Control Plane (L0)
    "ControlPlaneResults",
    "PolicyContext",
    "PolicyScope",
    "ModelInfo",
    "Determination",
    "ControlExecution",
    "ControlType",
    "ControlStatus",
    "SafetyScores",
    "SamplingDecision",
    # L1/L2 Attestation
    "Evidence",
    "Review",
    "DeepInspection",
    # Crypto utilities
    "canonical_json",
    "hash_payload",
]

# Add controls exports if available
if _CONTROLS_AVAILABLE:
    __all__.extend([
        "BaseControl",
        "ControlResult",
        "ControlsRunner",
        "PIIControl",
        "JailbreakControl",
    ])
