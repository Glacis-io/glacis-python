"""
Tests for Pydantic models (glacis/models.py).
"""

from typing import Any

import pytest
from pydantic import ValidationError

from glacis.models import (
    AttestReceipt,
    ControlPlaneResults,
    Determination,
    Evidence,
    GlacisApiError,
    GlacisRateLimitError,
    InclusionProof,
    LogEntry,
    LogQueryResult,
    MerkleInclusionProof,
    OfflineAttestReceipt,
    OfflineVerifyResult,
    SafetyScores,
    SignedTreeHead,
    VerifyResult,
)


class TestAttestReceipt:
    """Tests for AttestReceipt model."""

    def test_parse_from_camel_case(self, sample_online_receipt_data: dict[str, Any]):
        """Parse receipt from camelCase JSON."""
        receipt = AttestReceipt.model_validate(sample_online_receipt_data)

        assert receipt.id == "att_test123abc"
        assert receipt.leaf_index == 42
        assert receipt.tree_size == 100

    def test_witness_status_witnessed(self):
        """witness_status returns WITNESSED when receipt present."""
        data = {
            "attestationId": "att_test",
            "evidenceHash": "a" * 64,  # Spec: evidenceHash
            "timestamp": 1704067200000,  # Spec: Unix ms
            "leafIndex": 1,
            "treeSize": 1,
            "receipt": {
                "schema_version": "1.0",
                "attestation_hash": "a" * 64,
                "heartbeat_epoch": 1,
                "binary_hash": "b" * 64,
                "network_state_hash": "c" * 64,
                "mono_counter": 1,
                "wall_time_ns": "123456789",
                "witness_signature": "sig",
                "transparency_proofs": {
                    "inclusion_proof": {
                        "leaf_index": 1,
                        "tree_size": 1,
                        "hashes": [],
                        "root_hash": "d" * 64,
                    },
                    "sth_curr": {
                        "tree_size": 1,
                        "timestamp": "2024-01-01T00:00:00Z",
                        "root_hash": "d" * 64,
                        "signature": "sig",
                    },
                    "sth_prev": {
                        "tree_size": 0,
                        "timestamp": "2024-01-01T00:00:00Z",
                        "root_hash": "",
                        "signature": "sig",
                    },
                },
            },
        }
        receipt = AttestReceipt.model_validate(data)
        assert receipt.witness_status == "WITNESSED"

    def test_witness_status_unverified(self, sample_online_receipt_data: dict[str, Any]):
        """witness_status returns UNVERIFIED when no receipt."""
        receipt = AttestReceipt.model_validate(sample_online_receipt_data)
        assert receipt.witness_status == "UNVERIFIED"

    def test_evidence_hash_field(self, sample_online_receipt_data: dict[str, Any]):
        """evidence_hash (formerly attestation_hash) is accessible."""
        receipt = AttestReceipt.model_validate(sample_online_receipt_data)
        assert receipt.evidence_hash == "a" * 64


class TestOfflineAttestReceipt:
    """Tests for OfflineAttestReceipt model."""

    def test_parse_offline_receipt(self, sample_offline_receipt_data: dict[str, Any]):
        """Parse offline receipt."""
        receipt = OfflineAttestReceipt.model_validate(sample_offline_receipt_data)

        assert receipt.id == "oatt_test123abc"
        assert receipt.service_id == "test-service"
        assert receipt.is_offline is True
        assert receipt.witness_status == "UNVERIFIED"

    def test_witness_status_always_unverified(self):
        """Offline receipts always have UNVERIFIED status."""
        data = {
            "attestationId": "oatt_test",
            "evidenceHash": "a" * 64,
            "timestamp": 1704067200000,  # Unix ms
            "serviceId": "test",
            "operationType": "inference",
            "payloadHash": "a" * 64,
            "signature": "b" * 128,
            "publicKey": "c" * 64,
            "witnessStatus": "UNVERIFIED",
        }
        receipt = OfflineAttestReceipt.model_validate(data)
        assert receipt.witness_status == "UNVERIFIED"


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
            "witnessStatus": "UNVERIFIED",
            "signatureValid": True,
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

        assert results.schema_version == "1.0"
        assert results.policy.id == "policy-001"
        assert results.determination.action == "forwarded"
        assert len(results.controls) == 1
        assert results.safety.overall_risk == 0.1

    def test_determination_confidence_bounds(self):
        """Determination confidence must be 0-1."""
        with pytest.raises(ValidationError):
            Determination(action="forwarded", confidence=1.5)

        with pytest.raises(ValidationError):
            Determination(action="forwarded", confidence=-0.1)

    def test_safety_scores_bounds(self):
        """Safety overall_risk must be 0-1."""
        with pytest.raises(ValidationError):
            SafetyScores(overall_risk=1.5, scores={})


class TestEvidence:
    """Tests for Evidence (L1) model."""

    def test_evidence_sample_probability_bounds(self):
        """Evidence sample_probability must be 0-1."""
        with pytest.raises(ValidationError):
            Evidence(sample_probability=1.5, evidence_data={})

        with pytest.raises(ValidationError):
            Evidence(sample_probability=-0.1, evidence_data={})

    def test_evidence_with_data(self):
        """Evidence with data."""
        evidence = Evidence(
            sample_probability=0.5,
            evidence_data={"key": "value"},
        )
        assert evidence.sample_probability == 0.5
        assert evidence.evidence_data == {"key": "value"}


class TestLogEntry:
    """Tests for LogEntry model."""

    def test_log_entry_optional_fields(self):
        """LogEntry handles optional fields."""
        # Minimal data
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
            "payloadHash": "a" * 64,
            "leafIndex": 42,
        }
        entry = LogEntry.model_validate(data)

        assert entry.attestation_id == "att_test123"
        assert entry.org_name == "Test Org"
        assert entry.leaf_index == 42


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

    def test_inclusion_proof_camel_case(self):
        """InclusionProof parses camelCase (API response format)."""
        data = {
            "leafIndex": 42,
            "treeSize": 100,
            "hashes": ["a" * 64, "b" * 64],
            "rootHash": "c" * 64,
        }
        proof = InclusionProof.model_validate(data)

        assert proof.leaf_index == 42
        assert proof.tree_size == 100
        assert proof.root_hash == "c" * 64
        assert len(proof.hashes) == 2

    def test_inclusion_proof_snake_case(self):
        """InclusionProof parses snake_case (receipt-service wire format)."""
        data = {
            "leaf_index": 18,
            "tree_size": 19,
            "hashes": ["a" * 64],
            "root_hash": "d" * 64,
        }
        proof = InclusionProof.model_validate(data)

        assert proof.leaf_index == 18
        assert proof.tree_size == 19
        assert proof.root_hash == "d" * 64

    def test_merkle_inclusion_proof_backward_compat(self):
        """MerkleInclusionProof is an alias for InclusionProof."""
        assert MerkleInclusionProof is InclusionProof

    def test_signed_tree_head(self):
        """SignedTreeHead parsing."""
        data = {
            "treeSize": 100,
            "timestamp": "2024-01-01T00:00:00Z",
            "rootHash": "a" * 64,
            "signature": "sig123",
        }
        sth = SignedTreeHead.model_validate(data)

        assert sth.tree_size == 100
        assert sth.root_hash == "a" * 64
