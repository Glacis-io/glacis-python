"""
Pydantic models for the GLACIS API.

These models match the API responses from the management-api service
and the TypeScript SDK types.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


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


class MerkleInclusionProof(BaseModel):
    """Merkle inclusion proof structure."""

    leaf_index: int = Field(alias="leafIndex", description="Index of the leaf (0-based)")
    tree_size: int = Field(alias="treeSize", description="Total leaves when proof generated")
    hashes: list[str] = Field(description="Sibling hashes (hex-encoded)")

    class Config:
        populate_by_name = True


class SignedTreeHead(BaseModel):
    """Signed Tree Head - cryptographic commitment to tree state."""

    tree_size: int = Field(alias="treeSize", description="Total number of leaves")
    timestamp: str = Field(description="ISO 8601 timestamp when signed")
    root_hash: str = Field(alias="rootHash", description="Root hash (hex-encoded)")
    signature: str = Field(description="Ed25519 signature (base64-encoded)")

    class Config:
        populate_by_name = True


class AttestationMetadata(BaseModel):
    """Metadata for attestation requests - supports correlation and revision chains."""

    phase: Optional[Literal["input", "output"]] = Field(
        default=None, description="Distinguishes input vs output attestation"
    )
    correlation_id: Optional[str] = Field(
        alias="correlationId",
        default=None,
        description="UUID linking all attestations in an operation",
    )
    sequence_index: Optional[int] = Field(
        alias="sequenceIndex",
        default=None,
        description="Order within phase (0,1,2... for inputs; 0 for output)",
    )
    supersedes: Optional[str] = Field(
        default=None, description="Attestation ID this replaces (revision chains)"
    )

    class Config:
        populate_by_name = True


class AttestInput(BaseModel):
    """Input for attestation."""

    service_id: str = Field(alias="serviceId", description="Service identifier")
    operation_type: str = Field(
        alias="operationType",
        description="Type of operation (inference, embedding, completion, classification)",
    )
    input: Any = Field(description="Input data (hashed locally, never sent)")
    output: Any = Field(description="Output data (hashed locally, never sent)")
    metadata: Optional[AttestationMetadata] = Field(
        default=None, description="Optional metadata for correlation and revision chains"
    )

    class Config:
        populate_by_name = True


class InclusionProof(BaseModel):
    """Merkle inclusion proof from transparency log."""

    leaf_index: int = Field(alias="leaf_index", description="Leaf index in tree")
    tree_size: int = Field(alias="tree_size", description="Tree size when proof generated")
    hashes: list[str] = Field(description="Sibling hashes")
    root_hash: str = Field(alias="root_hash", description="Root hash")

    class Config:
        populate_by_name = True


class STH(BaseModel):
    """Signed Tree Head."""

    tree_size: int = Field(alias="tree_size")
    timestamp: str
    root_hash: str = Field(alias="root_hash")
    signature: str

    class Config:
        populate_by_name = True


class TransparencyProofs(BaseModel):
    """Transparency proofs from receipt-service."""

    inclusion_proof: InclusionProof
    sth_curr: STH
    sth_prev: STH
    consistency_path: list[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class FullReceipt(BaseModel):
    """Full receipt from receipt-service."""

    schema_version: str = Field(default="1.0")
    attestation_hash: str
    heartbeat_epoch: int
    binary_hash: str
    network_state_hash: str
    mono_counter: int
    wall_time_ns: str
    witness_signature: str
    transparency_proofs: Optional[TransparencyProofs] = Field(default=None)

    class Config:
        populate_by_name = True


class AttestReceipt(BaseModel):
    """Receipt returned from attestation.

    # TODO: Badge/verify URL functionality may be added back as a computed property
    # in the future once the verification endpoint design is finalized.
    """

    id: str = Field(alias="attestationId", description="Unique attestation ID")
    evidence_hash: str = Field(alias="evidenceHash", description="Evidence content hash (hex-encoded)")
    timestamp: int = Field(description="Unix timestamp in milliseconds")
    leaf_index: int = Field(alias="leafIndex", description="Merkle tree leaf index")
    tree_size: int = Field(alias="treeSize", description="Tree size")
    epoch_id: Optional[str] = Field(alias="epochId", default=None)
    receipt: Optional[FullReceipt] = Field(default=None, description="Full receipt with proofs")
    control_plane_results: Optional["ControlPlaneResults"] = Field(
        alias="controlPlaneResults",
        default=None,
        description="Control plane results from executed controls (L0)",
    )
    evidence: Optional["Evidence"] = Field(
        default=None,
        description="L1 attestation - evidence persistence layer",
    )
    review: Optional["Review"] = Field(
        default=None,
        description="L2 attestation - deep review layer",
    )

    # Computed properties for convenience
    @property
    def witness_status(self) -> str:
        """Return witness status based on receipt presence."""
        return "WITNESSED" if self.receipt else "PENDING"

    class Config:
        populate_by_name = True


class AttestationEntry(BaseModel):
    """Attestation entry from the log."""

    entry_id: str = Field(alias="entryId")
    timestamp: str
    org_id: str = Field(alias="orgId")
    service_id: str = Field(alias="serviceId")
    operation_type: str = Field(alias="operationType")
    payload_hash: str = Field(alias="payloadHash")
    signature: str
    leaf_index: int = Field(alias="leafIndex")
    leaf_hash: str = Field(alias="leafHash")

    class Config:
        populate_by_name = True


class OrgInfo(BaseModel):
    """Organization info."""

    id: str
    name: str
    domain: Optional[str] = None
    public_key: Optional[str] = Field(alias="publicKey", default=None)
    verified_at: Optional[str] = Field(alias="verifiedAt", default=None)

    class Config:
        populate_by_name = True


class Verification(BaseModel):
    """Verification details."""

    signature_valid: bool = Field(alias="signatureValid", default=False)
    proof_valid: bool = Field(alias="proofValid", default=False)
    verified_at: Optional[str] = Field(alias="verifiedAt", default=None)

    class Config:
        populate_by_name = True


class VerifyResult(BaseModel):
    """Result of verifying an attestation."""

    valid: bool = Field(description="Whether the attestation is valid")
    attestation: Optional[AttestationEntry] = Field(
        default=None, description="The attestation entry (if valid)"
    )
    org: Optional[OrgInfo] = Field(default=None, description="Organization info")
    verification: Optional[Verification] = Field(default=None, description="Verification details")
    proof: Optional[MerkleInclusionProof] = Field(default=None, description="Merkle proof")
    tree_head: Optional[SignedTreeHead] = Field(
        alias="treeHead", default=None, description="Current tree head"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if validation failed"
    )

    class Config:
        populate_by_name = True


class LogQueryParams(BaseModel):
    """Parameters for querying the log."""

    org_id: Optional[str] = Field(alias="orgId", default=None)
    service_id: Optional[str] = Field(alias="serviceId", default=None)
    start: Optional[str] = Field(default=None, description="Start timestamp (ISO 8601)")
    end: Optional[str] = Field(default=None, description="End timestamp (ISO 8601)")
    limit: Optional[int] = Field(default=50, ge=1, le=1000)
    cursor: Optional[str] = Field(default=None, description="Pagination cursor")

    class Config:
        populate_by_name = True


class LogEntry(BaseModel):
    """Log entry in query results."""

    # Server returns attestationId as the primary identifier
    attestation_id: str = Field(alias="attestationId")
    entry_id: Optional[str] = Field(alias="entryId", default=None)
    timestamp: Optional[str] = None
    org_id: Optional[str] = Field(alias="orgId", default=None)
    org_name: Optional[str] = Field(alias="orgName", default=None)
    service_id: Optional[str] = Field(alias="serviceId", default=None)
    operation_type: Optional[str] = Field(alias="operationType", default=None)
    payload_hash: Optional[str] = Field(alias="payloadHash", default=None)
    signature: Optional[str] = None
    leaf_index: Optional[int] = Field(alias="leafIndex", default=None)
    leaf_hash: Optional[str] = Field(alias="leafHash", default=None)

    class Config:
        populate_by_name = True


class LogQueryResult(BaseModel):
    """Result of querying the log."""

    entries: list[LogEntry] = Field(description="Log entries")
    has_more: bool = Field(alias="hasMore", description="Whether more results exist")
    next_cursor: Optional[str] = Field(
        alias="nextCursor", default=None, description="Cursor for next page"
    )
    count: int = Field(description="Number of entries returned")
    tree_head: Optional[SignedTreeHead] = Field(
        alias="treeHead", default=None, description="Current tree head"
    )

    class Config:
        populate_by_name = True


class TreeHeadResponse(BaseModel):
    """Response from get_tree_head."""

    size: int
    root_hash: str = Field(alias="rootHash")
    timestamp: str
    signature: str

    class Config:
        populate_by_name = True


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


# ==============================================================================
# Control Plane Attestation Models
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

ControlStatus = Literal["pass", "flag", "block", "error"]


class ModelInfo(BaseModel):
    """Model information for policy context."""

    model_id: str = Field(alias="modelId")
    provider: str
    system_prompt_hash: Optional[str] = Field(alias="systemPromptHash", default=None)
    temperature: Optional[float] = Field(default=None, description="Model temperature setting")

    class Config:
        populate_by_name = True


class PolicyScope(BaseModel):
    """Scope for policy application."""

    environment: str = Field(description="Environment (e.g., 'production', 'staging')")
    tags: list[str] = Field(default_factory=list, description="Custom tags")

    class Config:
        populate_by_name = True


class PolicyContext(BaseModel):
    """Policy context for attestation."""

    id: str
    version: str
    model: Optional[ModelInfo] = None
    scope: PolicyScope

    class Config:
        populate_by_name = True


class Determination(BaseModel):
    """Final determination for the request."""

    action: Literal["forwarded", "redacted", "blocked"]
    trigger: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)

    class Config:
        populate_by_name = True


class ControlExecution(BaseModel):
    """Record of a control execution."""

    id: str
    type: ControlType
    version: str
    provider: str  # "aws", "azure", "glacis", "custom", etc.
    latency_ms: int = Field(alias="latencyMs")
    status: ControlStatus
    result_hash: Optional[str] = Field(alias="resultHash", default=None)

    class Config:
        populate_by_name = True


class SafetyScores(BaseModel):
    """Aggregated safety scores."""

    overall_risk: float = Field(alias="overallRisk", ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)

    class Config:
        populate_by_name = True


class DeepInspection(BaseModel):
    """Deep inspection results from L2 verification."""

    judge_ids: list[str] = Field(alias="judgeIds", default_factory=list)
    nonconformity_score: float = Field(alias="nonconformityScore", ge=0.0, le=1.0)
    recommendation: Literal["uphold", "borderline", "escalate"]
    evaluation_rationale: str = Field(alias="evaluationRationale")

    class Config:
        populate_by_name = True


class Evidence(BaseModel):
    """L1 Attestation - Evidence persistence layer.

    Contains sampling probability and opaque evidence data blob.
    The sample_probability is typed for Rust to read for sampling decisions.
    """

    sample_probability: float = Field(
        alias="sampleProbability",
        ge=0.0,
        le=1.0,
        description="Probability of attestation evidence being sampled for persistence",
    )
    evidence_data: dict[str, Any] = Field(
        alias="evidenceData",
        default_factory=dict,
        description="Opaque evidence data blob",
    )

    class Config:
        populate_by_name = True


class Review(BaseModel):
    """L2 Attestation - Deep review layer.

    Contains sampling probability and review data (DeepInspection results).
    The sample_probability is typed for Rust to read for sampling decisions.
    """

    sample_probability: float = Field(
        alias="sampleProbability",
        ge=0.0,
        le=1.0,
        description="Probability of attestation being sampled for review",
    )
    review_data: DeepInspection = Field(
        alias="reviewData",
        description="Deep inspection results from L2 verification",
    )

    class Config:
        populate_by_name = True


class ControlPlaneResults(BaseModel):
    """Control plane results capturing policy, controls, and safety metadata (L0 only).

    Note: PII/jailbreak detection results are captured in controls[].
    Sampling and deep_inspection are moved to Evidence (L1) and Review (L2).
    """

    schema_version: Literal["1.0"] = "1.0"
    policy: PolicyContext
    determination: Determination
    controls: list[ControlExecution] = Field(default_factory=list)
    safety: SafetyScores

    class Config:
        populate_by_name = True


# Offline Mode Models


class OfflineAttestReceipt(BaseModel):
    """Receipt for offline/local attestations.

    Unlike server receipts, offline receipts are signed locally and do not
    have Merkle tree proofs or server-side tree heads. They can be verified
    locally using the public key, but are not witnessed by the transparency log.
    """

    id: str = Field(
        alias="attestationId", description="Local attestation ID (oatt_xxx)"
    )
    evidence_hash: str = Field(
        alias="evidenceHash", description="Evidence content hash (hex-encoded)"
    )
    timestamp: int = Field(description="Unix timestamp in milliseconds")
    service_id: str = Field(alias="serviceId", description="Service identifier")
    operation_type: str = Field(
        alias="operationType", description="Type of operation"
    )
    payload_hash: str = Field(
        alias="payloadHash", description="SHA-256 hash of input+output (hex)"
    )
    signature: str = Field(description="Ed25519 signature (base64)")
    public_key: str = Field(
        alias="publicKey", description="Public key derived from seed (hex)"
    )
    is_offline: bool = Field(default=True, alias="isOffline")
    witness_status: Literal["UNVERIFIED"] = Field(
        default="UNVERIFIED",
        alias="witnessStatus",
        description="Always UNVERIFIED for offline receipts",
    )
    control_plane_results: Optional[ControlPlaneResults] = Field(
        alias="controlPlaneResults",
        default=None,
        description="Control plane results from executed controls",
    )

    class Config:
        populate_by_name = True


class OfflineVerifyResult(BaseModel):
    """Verification result for offline receipts.

    Offline receipts can only have their signatures verified locally.
    The witness_status is always UNVERIFIED since there is no server-side
    transparency log entry.
    """

    valid: bool = Field(description="Whether the signature is valid")
    witness_status: Literal["UNVERIFIED"] = Field(
        default="UNVERIFIED", alias="witnessStatus"
    )
    signature_valid: bool = Field(alias="signatureValid")
    attestation: Optional[OfflineAttestReceipt] = Field(
        default=None, description="The verified offline receipt"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if verification failed"
    )

    class Config:
        populate_by_name = True
