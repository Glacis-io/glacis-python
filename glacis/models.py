"""
Pydantic models for the GLACIS API (v1.2 spec).

These models match the glacis-specification-v1.2 schemas.
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


class SignedTreeHead(BaseModel):
    """Cryptographic commitment to Merkle tree state."""

    model_config = ConfigDict(populate_by_name=True)

    tree_size: int = Field(alias="treeSize", description="Total number of leaves")
    timestamp: str = Field(description="ISO 8601 timestamp when signed")
    root_hash: str = Field(alias="rootHash", description="Root hash (hex-encoded)")
    public_key: Optional[str] = Field(
        alias="publicKey", default=None, description="Ed25519 public key (hex)"
    )
    signature: str = Field(description="Ed25519 signature (base64-encoded)")


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
    """L2 Attestation - Deep review record (v1.2 spec).

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
    signature: str = Field(default="", description="Arbiter Ed25519 signature (base64)")
    sampling_decision: Optional[SamplingDecision] = Field(default=None)

    # SDK convenience (not on wire)
    is_offline: bool = Field(default=False)
    timestamp: Optional[int] = Field(
        default=None, description="Unix timestamp ms (SDK convenience)"
    )

    @property
    def witness_status(self) -> str:
        return "UNVERIFIED" if self.is_offline else "WITNESSED"


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
    witness_status: Literal["UNVERIFIED"] = Field(default="UNVERIFIED")
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
