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


class AttestInput(BaseModel):
    """Input for attestation."""

    service_id: str = Field(alias="serviceId", description="Service identifier")
    operation_type: str = Field(
        alias="operationType",
        description="Type of operation (inference, embedding, completion, classification)",
    )
    input: Any = Field(description="Input data (hashed locally, never sent)")
    output: Any = Field(description="Output data (hashed locally, never sent)")
    metadata: Optional[dict[str, str]] = Field(
        default=None, description="Optional metadata (sent to server)"
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
    transparency_proofs: TransparencyProofs

    class Config:
        populate_by_name = True


class AttestReceipt(BaseModel):
    """Receipt returned from attestation."""

    attestation_id: str = Field(alias="attestationId", description="Unique attestation ID")
    attestation_hash: str = Field(alias="attestation_hash", description="Content hash")
    timestamp: str = Field(description="ISO 8601 timestamp")
    leaf_index: int = Field(alias="leafIndex", description="Merkle tree leaf index")
    tree_size: int = Field(alias="treeSize", description="Tree size")
    epoch_id: Optional[str] = Field(alias="epochId", default=None)
    receipt: Optional[FullReceipt] = Field(default=None, description="Full receipt with proofs")
    verify_url: str = Field(alias="verifyUrl", description="Verification endpoint URL")

    # Computed properties for convenience
    @property
    def witness_status(self) -> str:
        """Return witness status based on receipt presence."""
        return "WITNESSED" if self.receipt else "PENDING"

    @property
    def badge_url(self) -> str:
        """Return badge/verify URL."""
        return self.verify_url

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

    signature_valid: bool = Field(alias="signatureValid")
    proof_valid: bool = Field(alias="proofValid")
    verified_at: str = Field(alias="verifiedAt")

    class Config:
        populate_by_name = True


class VerifyResult(BaseModel):
    """Result of verifying an attestation."""

    valid: bool = Field(description="Whether the attestation is valid")
    attestation: Optional[AttestationEntry] = Field(
        default=None, description="The attestation entry (if valid)"
    )
    org: Optional[OrgInfo] = Field(default=None, description="Organization info")
    verification: Verification = Field(description="Verification details")
    proof: MerkleInclusionProof = Field(description="Merkle proof")
    tree_head: SignedTreeHead = Field(alias="treeHead", description="Current tree head")
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

    entry_id: str = Field(alias="entryId")
    timestamp: str
    org_id: str = Field(alias="orgId")
    org_name: Optional[str] = Field(alias="orgName", default=None)
    service_id: str = Field(alias="serviceId")
    operation_type: str = Field(alias="operationType")
    payload_hash: str = Field(alias="payloadHash")
    signature: str
    leaf_index: int = Field(alias="leafIndex")
    leaf_hash: str = Field(alias="leafHash")

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
    tree_head: SignedTreeHead = Field(alias="treeHead", description="Current tree head")

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


# Offline Mode Models


class OfflineAttestReceipt(BaseModel):
    """Receipt for offline/local attestations.

    Unlike server receipts, offline receipts are signed locally and do not
    have Merkle tree proofs or server-side tree heads. They can be verified
    locally using the public key, but are not witnessed by the transparency log.
    """

    attestation_id: str = Field(
        alias="attestationId", description="Local attestation ID (oatt_xxx)"
    )
    timestamp: str = Field(description="ISO 8601 timestamp")
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
