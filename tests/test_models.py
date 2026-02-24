"""
Tests for Pydantic models (glacis/models.py).
"""

from typing import Any

import pytest
from pydantic import ValidationError

from glacis.models import (
    Attestation,
    AttestReceipt,
    ControlExecution,
    ControlPlaneResults,
    Determination,
    Evidence,
    GlacisApiError,
    GlacisRateLimitError,
    InclusionProof,
    LogEntry,
    LogQueryResult,
    MerkleInclusionProof,
    ModelInfo,
    OfflineAttestReceipt,
    OfflineVerifyResult,
    PolicyContext,
    Receipt,
    Review,
    SamplingDecision,
    SignedTreeHead,
    VerifyResult,
)


class TestAttestation:
    """Tests for unified Attestation model (v1.2)."""

    def test_parse_attestation(self, sample_attestation_data: dict[str, Any]):
        """Parse attestation from snake_case data."""
        att = Attestation.model_validate(sample_attestation_data)

        assert att.id == "att_test123abc"
        assert att.evidence_hash == "a" * 64
        assert att.public_key == "d" * 64
        assert att.signature == "c" * 128
        assert att.timestamp == 1704110400000

    def test_witness_status_online(self, sample_attestation_data: dict[str, Any]):
        """Online attestation has WITNESSED status."""
        att = Attestation.model_validate(sample_attestation_data)
        assert att.witness_status == "WITNESSED"

    def test_witness_status_offline(self, sample_offline_attestation_data: dict[str, Any]):
        """Offline attestation has UNVERIFIED status."""
        att = Attestation.model_validate(sample_offline_attestation_data)
        assert att.is_offline is True
        assert att.witness_status == "UNVERIFIED"

    def test_operation_fields(self):
        """Operation ID and sequence are tracked."""
        att = Attestation(
            id="att_test",
            operation_id="op_123",
            operation_sequence=5,
            evidence_hash="a" * 64,
        )
        assert att.operation_id == "op_123"
        assert att.operation_sequence == 5

    def test_supersedes_field(self):
        """supersedes links revision chains."""
        att = Attestation(
            id="att_v2",
            supersedes="att_v1",
            evidence_hash="a" * 64,
        )
        assert att.supersedes == "att_v1"

    def test_control_plane_results_is_dict(self):
        """CPR on Attestation is dict[str, Any] (generic on wire)."""
        att = Attestation(
            id="att_test",
            evidence_hash="a" * 64,
            control_plane_results={"schema_version": "1.0", "determination": {"action": "forwarded"}},
        )
        assert att.control_plane_results["schema_version"] == "1.0"

    def test_defaults(self):
        """Attestation defaults are sensible."""
        att = Attestation(id="att_test")
        assert att.operation_id == ""
        assert att.operation_sequence == 0
        assert att.is_offline is False
        assert att.supersedes is None
        assert att.cpr_hash is None


class TestReceipt:
    """Tests for Receipt model (v1.2 — contains Attestation)."""

    def test_receipt_contains_attestation(self):
        """Receipt wraps an Attestation."""
        att = Attestation(id="att_test", evidence_hash="a" * 64)
        receipt = Receipt(
            attestation=att,
            timestamp=1704067200000,
            epoch_id="epoch_001",
            attestation_hash="b" * 64,
        )
        assert receipt.attestation.id == "att_test"
        assert receipt.timestamp == 1704067200000


class TestDeprecatedAliases:
    """Tests for backward-compatible aliases."""

    def test_attest_receipt_is_attestation(self):
        """AttestReceipt is an alias for Attestation."""
        assert AttestReceipt is Attestation

    def test_offline_attest_receipt_is_attestation(self):
        """OfflineAttestReceipt is an alias for Attestation."""
        assert OfflineAttestReceipt is Attestation

    def test_merkle_inclusion_proof_is_inclusion_proof(self):
        """MerkleInclusionProof is an alias for InclusionProof."""
        assert MerkleInclusionProof is InclusionProof


class TestVerifyResult:
    """Tests for VerifyResult model."""

    def test_parse_valid_result(self, sample_verify_response: dict[str, Any]):
        """Parse valid verification result."""
        result = VerifyResult.model_validate(sample_verify_response)

        assert result.valid is True
        assert result.verification is not None
        assert result.verification.signature_valid is True
        assert result.verification.proof_valid is True
        assert result.proof is not None
        assert result.tree_head is not None

    def test_parse_invalid_result(self):
        """Parse invalid verification result."""
        data = {"valid": False, "error": "Attestation not found"}
        result = VerifyResult.model_validate(data)

        assert result.valid is False
        assert result.error == "Attestation not found"


class TestOfflineVerifyResult:
    """Tests for OfflineVerifyResult model."""

    def test_parse_offline_verify_result(self):
        """Parse offline verification result."""
        data = {
            "valid": True,
            "witness_status": "UNVERIFIED",
            "signature_valid": True,
        }
        result = OfflineVerifyResult.model_validate(data)

        assert result.valid is True
        assert result.witness_status == "UNVERIFIED"
        assert result.signature_valid is True


class TestControlPlaneResults:
    """Tests for ControlPlaneResults model."""

    def test_parse_full_control_plane(self, sample_control_plane_data: dict[str, Any]):
        """Parse complete control plane results."""
        results = ControlPlaneResults.model_validate(sample_control_plane_data)

        assert results.policy.id == "policy-001"
        assert results.policy.model.model_id == "gpt-4"
        assert results.policy.environment == "production"
        assert results.policy.tags == ["healthcare", "hipaa"]
        assert results.determination.action == "forwarded"
        assert len(results.controls) == 1
        assert results.controls[0].latency_ms == 15

    def test_determination_action_only(self):
        """Determination only has action field."""
        det = Determination(action="forwarded")
        dumped = det.model_dump()
        assert dumped == {"action": "forwarded"}

    def test_control_execution_score(self):
        """ControlExecution can carry a numeric score."""
        ce = ControlExecution(
            id="jb", type="jailbreak", version="0.3.0",
            provider="glacis", latency_ms=50, status="forward", score=0.02,
        )
        assert ce.score == 0.02

    def test_control_execution_score_optional(self):
        """ControlExecution score defaults to None."""
        ce = ControlExecution(
            id="pii", type="pii", version="0.3.0",
            provider="glacis", latency_ms=10, status="flag",
        )
        assert ce.score is None

    def test_policy_context_flat_environment(self):
        """PolicyContext has environment and tags directly (no scope)."""
        pc = PolicyContext(
            id="test", version="1.0",
            model=ModelInfo(model_id="gpt-4", provider="openai"),
            environment="staging", tags=["hipaa"],
        )
        assert pc.environment == "staging"
        assert pc.tags == ["hipaa"]


class TestEvidence:
    """Tests for Evidence (L1) model."""

    def test_evidence_sample_probability_bounds(self):
        """Evidence sample_probability must be 0-1."""
        with pytest.raises(ValidationError):
            Evidence(sample_probability=1.5, data={})

        with pytest.raises(ValidationError):
            Evidence(sample_probability=-0.1, data={})

    def test_evidence_with_data(self):
        """Evidence with data."""
        evidence = Evidence(
            sample_probability=0.5,
            data={"key": "value"},
        )
        assert evidence.sample_probability == 0.5
        assert evidence.data == {"key": "value"}


class TestReview:
    """Tests for Review (L2) model (flattened from DeepInspection)."""

    def test_review_fields(self):
        """Review has all flattened fields."""
        review = Review(
            sample_probability=0.1,
            judge_ids=["judge_1", "judge_2"],
            conformity_score=0.3,
            recommendation="uphold",
            rationale="Looks good",
        )
        assert review.sample_probability == 0.1
        assert len(review.judge_ids) == 2
        assert review.recommendation == "uphold"

    def test_review_recommendation_values(self):
        """Review recommendation must be uphold/borderline/escalate."""
        with pytest.raises(ValidationError):
            Review(
                sample_probability=0.1,
                conformity_score=0.0,
                recommendation="invalid",
                rationale="",
            )


class TestSamplingDecision:
    """Tests for SamplingDecision model (v1.2)."""

    def test_sampling_decision_defaults(self):
        """SamplingDecision defaults."""
        sd = SamplingDecision(level="L0")
        assert sd.level == "L0"
        assert sd.sample_value == 0
        assert sd.prf_tag == []

    def test_sampling_decision_with_prf_tag(self):
        """SamplingDecision with prf_tag (HMAC-SHA256 bytes)."""
        sd = SamplingDecision(
            level="L1",
            sample_value=12345678,
            prf_tag=[1, 2, 3, 4],
        )
        assert sd.level == "L1"
        assert sd.sample_value == 12345678
        assert sd.prf_tag == [1, 2, 3, 4]

    def test_sample_value_is_int(self):
        """sample_value is now int (was Optional[str] in v1.1)."""
        sd = SamplingDecision(level="L0", sample_value=42)
        assert isinstance(sd.sample_value, int)


class TestLogEntry:
    """Tests for LogEntry model."""

    def test_log_entry_optional_fields(self):
        """LogEntry handles optional fields."""
        data = {"attestationId": "att_test123"}
        entry = LogEntry.model_validate(data)

        assert entry.attestation_id == "att_test123"
        assert entry.timestamp is None
        assert entry.service_id is None

    def test_log_entry_full_fields(self):
        """LogEntry with all fields."""
        data = {
            "attestationId": "att_test123",
            "entryId": "entry_001",
            "timestamp": "2024-01-01T00:00:00Z",
            "orgId": "org_xxx",
            "orgName": "Test Org",
            "serviceId": "my-service",
            "operationType": "inference",
            "evidenceHash": "a" * 64,
            "leafIndex": 42,
        }
        entry = LogEntry.model_validate(data)

        assert entry.attestation_id == "att_test123"
        assert entry.org_name == "Test Org"
        assert entry.leaf_index == 42
        assert entry.evidence_hash == "a" * 64


class TestLogQueryResult:
    """Tests for LogQueryResult model."""

    def test_log_query_result_empty(self):
        """LogQueryResult with no entries."""
        data = {"entries": [], "hasMore": False, "count": 0}
        result = LogQueryResult.model_validate(data)

        assert result.entries == []
        assert result.has_more is False
        assert result.count == 0

    def test_log_query_result_with_entries(self):
        """LogQueryResult with entries."""
        data = {
            "entries": [
                {"attestationId": "att_1"},
                {"attestationId": "att_2"},
            ],
            "hasMore": True,
            "nextCursor": "cursor123",
            "count": 2,
        }
        result = LogQueryResult.model_validate(data)

        assert len(result.entries) == 2
        assert result.has_more is True
        assert result.next_cursor == "cursor123"


class TestErrorModels:
    """Tests for error models."""

    def test_glacis_api_error(self):
        """GlacisApiError construction."""
        error = GlacisApiError("Something went wrong", 400, "BAD_REQUEST", {"field": "invalid"})

        assert str(error) == "Something went wrong"
        assert error.status == 400
        assert error.code == "BAD_REQUEST"
        assert error.details == {"field": "invalid"}

    def test_rate_limit_error(self):
        """GlacisRateLimitError construction."""
        error = GlacisRateLimitError("Rate limit exceeded", retry_after_ms=60000)

        assert error.status == 429
        assert error.code == "RATE_LIMITED"
        assert error.retry_after_ms == 60000


class TestMerkleProofs:
    """Tests for Merkle proof models."""

    def test_inclusion_proof_snake_case(self):
        """InclusionProof parses snake_case."""
        data = {
            "leaf_index": 42,
            "tree_size": 100,
            "hashes": ["a" * 64, "b" * 64],
            "root_hash": "c" * 64,
        }
        proof = InclusionProof.model_validate(data)

        assert proof.leaf_index == 42
        assert proof.tree_size == 100
        assert proof.root_hash == "c" * 64
        assert len(proof.hashes) == 2

    def test_signed_tree_head(self):
        """SignedTreeHead parsing."""
        data = {
            "tree_size": 100,
            "timestamp": "2024-01-01T00:00:00Z",
            "root_hash": "a" * 64,
            "signature": "sig123",
        }
        sth = SignedTreeHead.model_validate(data)

        assert sth.tree_size == 100
        assert sth.root_hash == "a" * 64

    def test_signed_tree_head_with_public_key(self):
        """SignedTreeHead with optional public_key."""
        data = {
            "tree_size": 100,
            "timestamp": "2024-01-01T00:00:00Z",
            "root_hash": "a" * 64,
            "signature": "sig123",
            "public_key": "d" * 64,
        }
        sth = SignedTreeHead.model_validate(data)

        assert sth.public_key == "d" * 64
