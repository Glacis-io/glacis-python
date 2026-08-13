"""
Pydantic models for the GLACIS API (v1.3 spec).

These models match the glacis-specification-v1.3 schemas.
Wire format is snake_case throughout.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class GlacisConfig(BaseModel):
    """Configuration for the Glacis client."""

    api_key: str = Field(..., description="API key (glsk_live_xxx or glsk_test_xxx)")
    base_url: str = Field(
        default="https://api.glacis.io", description="Base URL for the API"
    )
    debug: bool = Field(default=False, description="Enable debug logging")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    base_delay: float = Field(
        default=1.0, description="Base delay in seconds for exponential backoff"
    )
    max_delay: float = Field(
        default=30.0, description="Maximum delay in seconds for backoff"
    )


# ==============================================================================
# Transparency Proof Models (RFC 6962)
# ==============================================================================


# SignedTreeHead and InclusionProof keep camelCase aliases because they are
# embedded in API response DTOs (VerifyResult, LogQueryResult) where the
# Notary API sends camelCase JSON. The spec wire format is snake_case, but
# these models serve double duty as API deserialization targets.


class SignedTreeHead(BaseModel):
    """Cryptographic commitment to Merkle tree state."""

    model_config = ConfigDict(populate_by_name=True)

    tree_size: int = Field(alias="treeSize", description="Total number of leaves")
    timestamp: str = Field(description="ISO 8601 timestamp when signed")
    root_hash: str = Field(alias="rootHash", description="Root hash (hex-encoded)")
    public_key: Optional[str] = Field(
        alias="publicKey", default=None, description="Ed25519 public key (hex)"
    )
    signature: str = Field(description="Ed25519 signature (hex-encoded)")


class InclusionProof(BaseModel):
    """RFC 6962 Merkle inclusion proof."""

    model_config = ConfigDict(populate_by_name=True)

    leaf_index: int = Field(alias="leafIndex", description="Leaf index in tree (0-based)")
    tree_size: int = Field(alias="treeSize", description="Tree size when proof generated")
    hashes: list[str] = Field(description="Sibling hashes (hex-encoded)")
    root_hash: Optional[str] = Field(
        alias="rootHash", default=None, description="Root hash (hex-encoded)",
    )


# Backward-compatible alias (deprecated)
MerkleInclusionProof = InclusionProof

# STH is deprecated — use SignedTreeHead
STH = SignedTreeHead


class TransparencyProofs(BaseModel):
    """RFC 6962 transparency proof structure."""

    model_config = ConfigDict(populate_by_name=True)

    inclusion_proof: InclusionProof
    sth_curr: SignedTreeHead
    sth_prev: SignedTreeHead
    consistency_path: list[str] = Field(default_factory=list)


# ==============================================================================
# Attestation Metadata
# ==============================================================================


class AttestationMetadata(BaseModel):
    """Metadata for attestation requests (v1.2 spec)."""

    model_config = ConfigDict(populate_by_name=True)

    operation_id: Optional[str] = Field(
        default=None,
        description="UUID linking all attestations in an operation",
    )
    operation_sequence: Optional[int] = Field(
        default=None,
        description="Ordinal sequence within the operation",
    )
    supersedes: Optional[str] = Field(
        default=None, description="Attestation ID this replaces (revision chains)"
    )


class AttestInput(BaseModel):
    """Input for attestation."""

    model_config = ConfigDict(populate_by_name=True)

    service_id: str = Field(description="Service identifier")
    operation_type: str = Field(
        description="Type of operation (inference, embedding, completion, classification)",
    )
    input: Any = Field(description="Input data (hashed locally, never sent)")
    output: Any = Field(description="Output data (hashed locally, never sent)")
    metadata: Optional[AttestationMetadata] = Field(
        default=None, description="Optional metadata for correlation and revision chains"
    )


# ==============================================================================
# Sampling Decision
# ==============================================================================


class SamplingDecision(BaseModel):
    """Deterministic, auditor-reproducible sampling tier assignment (v1.2 spec)."""

    model_config = ConfigDict(populate_by_name=True)

    level: str = Field(description="Sampling tier: L0, L1, or L2")
    sample_value: int = Field(
        default=0,
        description="First 8 bytes of prf_tag, big-endian uint64",
    )
    prf_tag: list[int] = Field(
        default_factory=list,
        description="Full HMAC-SHA256 tag over the evidence hash",
    )


# ==============================================================================
# Evidence & Review (L1/L2)
# ==============================================================================


class Evidence(BaseModel):
    """L1 Attestation - Sampled evidence payload (v1.2 spec).

    Structure is application-defined. CPR integrity is attested
    independently via cpr_hash in the Merkle leaf.
    """

    model_config = ConfigDict(populate_by_name=True)

    sample_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability this evidence was sampled",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="The evidence payload",
    )


class Review(BaseModel):
    """L2 Attestation - Deep review record (v1.3 spec).

    Flattened from the previous Review + DeepInspection structure.
    """

    model_config = ConfigDict(populate_by_name=True)

    sample_probability: float = Field(ge=0.0, le=1.0)
    judge_ids: list[str] = Field(default_factory=list)
    conformity_score: float = Field(ge=0.0, le=1.0)
    recommendation: Literal["uphold", "borderline", "escalate"]
    rationale: str


# ==============================================================================
# Control Plane Models (SDK convenience — wire format is dict[str, Any])
# ==============================================================================

ControlType = Literal[
    "content_safety",
    "pii",
    "jailbreak",
    "topic",
    "prompt_security",
    "grounding",
    "word_filter",
    "custom",
]

ControlStatus = Literal["forward", "flag", "block", "error"]


class ModelInfo(BaseModel):
    """Model information for policy context."""

    model_config = ConfigDict(populate_by_name=True)

    model_id: str
    provider: str
    system_prompt_hash: Optional[str] = None
    temperature: Optional[float] = None


class PolicyContext(BaseModel):
    """Policy metadata for attestation."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    version: str
    model: Optional[ModelInfo] = None
    environment: str = "development"
    tags: list[str] = Field(default_factory=list)


class Determination(BaseModel):
    """Whether the request was forwarded or blocked."""

    model_config = ConfigDict(populate_by_name=True)

    action: Literal["forwarded", "blocked"]


class ControlExecution(BaseModel):
    """Record of a control execution."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    type: ControlType
    version: str
    provider: str
    latency_ms: int
    status: ControlStatus
    score: Optional[float] = Field(
        default=None, description="Numeric score (e.g. jailbreak probability)"
    )
    result_hash: Optional[str] = None
    stage: Literal["input", "output"] = "input"


class ControlPlaneResults(BaseModel):
    """Control plane results (SDK convenience model).

    On the wire, Attestation.control_plane_results is dict[str, Any].
    This typed model serializes via .model_dump() before being set on the attestation.
    """

    model_config = ConfigDict(populate_by_name=True)

    policy: PolicyContext
    determination: Determination
    controls: list[ControlExecution] = Field(default_factory=list)


# ==============================================================================
# Attestation (v1.2 — unified model)
# ==============================================================================


class Attestation(BaseModel):
    """Unified attestation model (v1.2 spec).

    Every attestation carries Arbiter public_key + signature.
    Online mode: accompanied by a Receipt from the Notary.
    Offline mode: the signature is the sole proof of integrity.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    operation_id: str = Field(default="")
    operation_sequence: int = Field(default=0)
    service_id: str = Field(default="")
    operation_type: str = Field(default="")
    evidence_hash: str = Field(default="", description="SHA-256 of canonical JSON evidence")
    cpr_hash: Optional[str] = Field(default=None)
    supersedes: Optional[str] = Field(default=None, description="Attestation ID this replaces")
    control_plane_results: Optional[dict[str, Any]] = Field(default=None)
    evidence: Optional[Evidence] = Field(default=None, description="L1 sampled evidence")
    review: Optional[Review] = Field(default=None, description="L2 deep review")
    public_key: str = Field(default="", description="Arbiter Ed25519 public key (hex)")
    signature: str = Field(default="", description="Arbiter Ed25519 signature (hex)")
    sampling_decision: Optional[SamplingDecision] = Field(default=None)

    # SDK convenience (not on wire)
    is_offline: bool = Field(default=False)
    timestamp: Optional[int] = Field(
        default=None, description="Unix timestamp ms (SDK convenience)"
    )
    cpr_recovery_error: Optional[str] = Field(
        default=None,
        description=(
            "SDK convenience, never signed and never transmitted. Set when a "
            "receipt was reconstructed from storage that could not return its "
            "control_plane_results even though cpr_hash says the receipt was "
            "signed over some. The signed payload cannot be rebuilt, so "
            "verification of such a receipt fails with this string as the reason."
        ),
    )

    @property
    def witness_status(self) -> str:
        """``SELF_SIGNED`` for locally signed receipts, else ``LOGGED_UNVERIFIED``.

        ``WITNESSED`` is never derived from a flag on this object. It is
        issued only by ``glacis.witness.classify_envelope`` after the
        inclusion proof recomputes to a tree head signed under a *configured*
        log public key — see ``HostedArtifact.verification``. (0.8.1 returned
        ``WITNESSED`` for any ``is_offline=False`` object with zero
        verification; a self-signed or merely-logged receipt must never carry
        a server-attested label.)
        """
        return "SELF_SIGNED" if self.is_offline else "LOGGED_UNVERIFIED"


# ==============================================================================
# Receipt (v1.2 — contains Attestation, inverted nesting)
# ==============================================================================


class Receipt(BaseModel):
    """Notary receipt (v1.2 spec). Contains the attestation it covers."""

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field(default="1.0")
    attestation: Attestation
    timestamp: int = Field(description="Unix epoch timestamp ms")
    epoch_id: str = Field(default="")
    heartbeat_epoch: int = Field(default=0)
    attestation_hash: str = Field(default="", description="SHA-256 of canonical attestation")
    binary_hash: str = Field(default="")
    network_state_hash: str = Field(default="")
    mono_counter: int = Field(default=0)
    wall_time_ns: str = Field(default="")
    transparency_proofs: Optional[TransparencyProofs] = Field(default=None)
    public_key: str = Field(default="", description="Notary Ed25519 public key")
    signature: str = Field(default="", description="Notary Ed25519 signature")


# ==============================================================================
# Hosted (server-attested) mint — 0.9.0
# ==============================================================================


class WitnessVerification(BaseModel):
    """What the SDK itself verified about a hosted mint, fail-closed.

    ``witness_status`` is ``WITNESSED`` only when ``inclusion_verified`` and
    ``sth_signature_verified`` are both True under a configured log key.
    Everything else is ``LOGGED_UNVERIFIED`` with ``reason`` naming the first
    missing piece. ``contradicted`` marks the one case where a configured key
    vouched for a root and this receipt's proof does not lead to it.
    """

    model_config = ConfigDict(populate_by_name=True)

    witness_status: Literal["WITNESSED", "LOGGED_UNVERIFIED"]
    inclusion_verified: bool = Field(default=False)
    sth_signature_verified: bool = Field(default=False)
    log_public_key_hex: Optional[str] = Field(
        default=None, description="The configured log key that verified the tree head"
    )
    contradicted: bool = Field(default=False)
    reason: Optional[str] = Field(default=None)
    checked_at_ms: int = Field(default=0)


class WitnessBinding(BaseModel):
    """How the local attestation is bound to the gateway receipt.

    ``request_sha256`` is SHA-256 over the local attestation's exact signed
    bytes (``glacis.crypto.offline_signed_payload``: compact JSON, sorted
    keys, ``version:1``, ``mode:"offline"``). The gateway echoes it verbatim
    as ``receipt.commitments.request`` and commits to it inside the (private)
    receipt-hash preimage that becomes the log leaf. A public verifier can
    re-derive ``request_sha256`` from the attestation in this artifact and
    check the echo; recomputing ``receipt_id`` from it requires the private
    receipt shape and is not possible from this artifact alone.
    """

    model_config = ConfigDict(populate_by_name=True)

    scheme: Literal["glacis-attestation-binding/1"] = "glacis-attestation-binding/1"
    request_sha256: str = Field(
        description="sha256(offline_signed_payload bytes) of the local attestation"
    )


class HostedArtifact(BaseModel):
    """The composite artifact a hosted mint returns — one pasteable JSON.

    Top level is a superset of the ``{v, receipt, inclusion}`` permalink
    envelope, so the whole artifact parses at glacis.io/verify (its envelope
    unwrap keys on the top-level ``receipt`` object and ignores unknown
    fields). ``receipt`` and ``inclusion`` are the gateway's response
    verbatim — the SDK never reshapes them.
    """

    model_config = ConfigDict(populate_by_name=True)

    v: int = Field(default=1)
    artifact: Literal["glacis-hosted-mint/1"] = "glacis-hosted-mint/1"
    attestation_mode: Literal["server-attested"] = "server-attested"
    receipt: dict[str, Any] = Field(
        description="Projected receipt, verbatim from POST /v1/govern"
    )
    inclusion: dict[str, Any] = Field(
        description="Transparency-log record, verbatim from the gateway"
    )
    attestation: Attestation = Field(
        description="The locally signed attestation (identical to offline mode)"
    )
    binding: WitnessBinding
    verification: WitnessVerification

    @property
    def witness_status(self) -> str:
        return self.verification.witness_status

    def to_json(self, indent: int = 2) -> str:
        """Serialize to the single JSON file a user can paste at glacis.io/verify."""
        import json as _json

        return _json.dumps(self.model_dump(mode="json"), indent=indent)

    def save(self, path: Any) -> None:
        """Write ``to_json()`` to ``path``."""
        from pathlib import Path as _Path

        _Path(path).write_text(self.to_json() + "\n", encoding="utf-8")


# ==============================================================================
# Deprecation aliases (one release)
# ==============================================================================

AttestReceipt = Attestation
OfflineAttestReceipt = Attestation
FullReceipt = Receipt

# DeepInspection is folded into Review — keep name for import compatibility
DeepInspection = Review


# ==============================================================================
# Log Query Models
# ==============================================================================


class LogQueryParams(BaseModel):
    """Parameters for querying the log."""

    model_config = ConfigDict(populate_by_name=True)

    org_id: Optional[str] = Field(default=None)
    service_id: Optional[str] = Field(default=None)
    start: Optional[str] = Field(default=None, description="Start timestamp (ISO 8601)")
    end: Optional[str] = Field(default=None, description="End timestamp (ISO 8601)")
    limit: Optional[int] = Field(default=50, ge=1, le=1000)
    cursor: Optional[str] = Field(default=None, description="Pagination cursor")


# Note: LogEntry, LogQueryResult, AttestationEntry, OrgInfo, Verification keep
# camelCase aliases because they deserialize from the existing Notary API which
# sends camelCase JSON. These are API response DTOs, not spec wire-format models.


class LogEntry(BaseModel):
    """Log entry in query results."""

    model_config = ConfigDict(populate_by_name=True)

    attestation_id: str = Field(alias="attestationId")
    entry_id: Optional[str] = Field(alias="entryId", default=None)
    timestamp: Optional[str] = None
    org_id: Optional[str] = Field(alias="orgId", default=None)
    org_name: Optional[str] = Field(alias="orgName", default=None)
    service_id: Optional[str] = Field(alias="serviceId", default=None)
    operation_type: Optional[str] = Field(alias="operationType", default=None)
    evidence_hash: Optional[str] = Field(alias="evidenceHash", default=None)
    signature: Optional[str] = None
    leaf_index: Optional[int] = Field(alias="leafIndex", default=None)
    leaf_hash: Optional[str] = Field(alias="leafHash", default=None)


class LogQueryResult(BaseModel):
    """Result of querying the log."""

    model_config = ConfigDict(populate_by_name=True)

    entries: list[LogEntry] = Field(description="Log entries")
    has_more: bool = Field(alias="hasMore", description="Whether more results exist")
    next_cursor: Optional[str] = Field(
        alias="nextCursor", default=None, description="Cursor for next page"
    )
    count: int = Field(description="Number of entries returned")
    tree_head: Optional[SignedTreeHead] = Field(
        alias="treeHead", default=None, description="Current tree head"
    )


# ==============================================================================
# Verification Models
# ==============================================================================


class AttestationEntry(BaseModel):
    """Attestation entry from the log."""

    model_config = ConfigDict(populate_by_name=True)

    entry_id: str = Field(alias="entryId")
    timestamp: str
    org_id: str = Field(alias="orgId")
    service_id: str = Field(alias="serviceId")
    operation_type: str = Field(alias="operationType")
    evidence_hash: str = Field(alias="evidenceHash")
    signature: str
    leaf_index: int = Field(alias="leafIndex")
    leaf_hash: str = Field(alias="leafHash")


class OrgInfo(BaseModel):
    """Organization info."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    domain: Optional[str] = None
    public_key: Optional[str] = Field(alias="publicKey", default=None)
    verified_at: Optional[str] = Field(alias="verifiedAt", default=None)


class Verification(BaseModel):
    """Verification details."""

    model_config = ConfigDict(populate_by_name=True)

    signature_valid: bool = Field(alias="signatureValid", default=False)
    proof_valid: bool = Field(alias="proofValid", default=False)
    verified_at: Optional[str] = Field(alias="verifiedAt", default=None)


class VerifyResult(BaseModel):
    """Result of verifying an attestation."""

    model_config = ConfigDict(populate_by_name=True)

    valid: bool = Field(description="Whether the attestation is valid")
    attestation: Optional[AttestationEntry] = Field(
        default=None, description="The attestation entry (if valid)"
    )
    org: Optional[OrgInfo] = Field(default=None, description="Organization info")
    verification: Optional[Verification] = Field(default=None)
    proof: Optional[InclusionProof] = Field(default=None)
    tree_head: Optional[SignedTreeHead] = Field(
        alias="treeHead", default=None,
    )
    error: Optional[str] = Field(default=None)


class TreeHeadResponse(BaseModel):
    """Response from get_tree_head."""

    model_config = ConfigDict(populate_by_name=True)

    tree_size: int = Field(alias="treeSize")
    root_hash: str = Field(alias="rootHash")
    timestamp: str
    signature: str


class OfflineVerifyResult(BaseModel):
    """Verification result for offline attestations."""

    model_config = ConfigDict(populate_by_name=True)

    valid: bool = Field(description="Whether the signature is valid")
    witness_status: Literal["SELF_SIGNED"] = Field(default="SELF_SIGNED")
    signature_valid: bool
    attestation: Optional[Attestation] = Field(
        default=None, description="The verified attestation"
    )
    error: Optional[str] = Field(default=None)


# ==============================================================================
# Errors
# ==============================================================================


class GlacisApiError(Exception):
    """Error from the GLACIS API."""

    def __init__(
        self,
        message: str,
        status: int,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.details = details


class GlacisRateLimitError(GlacisApiError):
    """Rate limit error."""

    def __init__(self, message: str, retry_after_ms: Optional[int] = None):
        super().__init__(message, 429, "RATE_LIMITED")
        self.retry_after_ms = retry_after_ms


class GlacisMintError(Exception):
    """A hosted mint could not produce a bound artifact.

    Raised when the gateway's response cannot be tied to the request the SDK
    made (e.g. ``receipt.commitments.request`` differs from the
    ``request_sha256`` that was sent). This is distinct from a transport or
    HTTP error: the server answered, and the answer does not bind.
    """
