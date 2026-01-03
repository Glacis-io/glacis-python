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
    >>> print(f"Verified at: {receipt.verify_url}")

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
"""

from glacis.client import AsyncGlacis, Glacis, GlacisMode
from glacis.crypto import canonical_json, hash_payload
from glacis.models import (
    AttestInput,
    AttestReceipt,
    GlacisConfig,
    LogEntry,
    LogQueryParams,
    LogQueryResult,
    MerkleInclusionProof,
    OfflineAttestReceipt,
    OfflineVerifyResult,
    SignedTreeHead,
    VerifyResult,
)
from glacis.storage import ReceiptStorage
from glacis.streaming import SessionContext, SessionReceipt, StreamingSession

__version__ = "0.2.0"

__all__ = [
    # Main classes
    "Glacis",
    "AsyncGlacis",
    "GlacisMode",
    # Streaming
    "StreamingSession",
    "SessionContext",
    "SessionReceipt",
    # Storage (offline mode)
    "ReceiptStorage",
    # Models
    "GlacisConfig",
    "AttestInput",
    "AttestReceipt",
    "OfflineAttestReceipt",
    "VerifyResult",
    "OfflineVerifyResult",
    "LogQueryParams",
    "LogQueryResult",
    "LogEntry",
    "MerkleInclusionProof",
    "SignedTreeHead",
    # Crypto utilities
    "canonical_json",
    "hash_payload",
]
