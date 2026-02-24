"""
Shared test fixtures for glacis-python tests.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest

# Load environment variables from tests/.env
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


@pytest.fixture
def signing_seed() -> bytes:
    """Generate a random 32-byte Ed25519 signing seed."""
    return os.urandom(32)


@pytest.fixture
def fixed_seed() -> bytes:
    """Fixed seed for deterministic tests."""
    return bytes(32)  # All zeros


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def sample_attestation_data() -> dict[str, Any]:
    """Sample online attestation data (v1.2 spec, snake_case)."""
    return {
        "id": "att_test123abc",
        "operation_id": "op_test456",
        "operation_sequence": 0,
        "service_id": "test-service",
        "operation_type": "completion",
        "evidence_hash": "a" * 64,
        "public_key": "d" * 64,
        "signature": "c" * 128,
        "timestamp": 1704110400000,
    }


# Backward-compatible alias
@pytest.fixture
def sample_online_receipt_data(sample_attestation_data: dict[str, Any]) -> dict[str, Any]:
    """Alias for sample_attestation_data (backward compat)."""
    return sample_attestation_data


@pytest.fixture
def sample_offline_attestation_data() -> dict[str, Any]:
    """Sample offline attestation data (v1.2 spec)."""
    return {
        "id": "oatt_test123abc",
        "operation_id": "op_test789",
        "operation_sequence": 0,
        "service_id": "test-service",
        "operation_type": "inference",
        "evidence_hash": "b" * 64,
        "public_key": "d" * 64,
        "signature": "c" * 128,
        "is_offline": True,
        "timestamp": 1704110400000,
    }


# Backward-compatible alias
@pytest.fixture
def sample_offline_receipt_data(sample_offline_attestation_data: dict[str, Any]) -> dict[str, Any]:
    """Alias for sample_offline_attestation_data (backward compat)."""
    return sample_offline_attestation_data


@pytest.fixture
def sample_verify_response() -> dict[str, Any]:
    """Standard verify API response."""
    return {
        "valid": True,
        "attestation": {
            "entryId": "att_test123",
            "timestamp": "2024-01-01T12:00:00Z",
            "orgId": "org_xxx",
            "serviceId": "test-service",
            "operationType": "inference",
            "evidenceHash": "a" * 64,
            "signature": "sig123",
            "leafIndex": 42,
            "leafHash": "hash123",
        },
        "verification": {
            "signatureValid": True,
            "proofValid": True,
            "verifiedAt": "2024-01-01T12:00:01Z",
        },
        "proof": {"leafIndex": 42, "treeSize": 100, "hashes": ["e" * 64]},
        "treeHead": {
            "treeSize": 100,
            "timestamp": "2024-01-01T12:00:00Z",
            "rootHash": "f" * 64,
            "signature": "sig_sth",
        },
    }


@pytest.fixture
def sample_control_plane_data() -> dict[str, Any]:
    """Sample control plane results data (v1.2, snake_case)."""
    return {
        "policy": {
            "id": "policy-001",
            "version": "1.0",
            "model": {
                "model_id": "gpt-4",
                "provider": "openai",
                "system_prompt_hash": "a" * 64,
            },
            "environment": "production",
            "tags": ["healthcare", "hipaa"],
        },
        "determination": {
            "action": "forwarded",
        },
        "controls": [
            {
                "id": "pii-001",
                "type": "pii",
                "version": "1.0",
                "provider": "glacis",
                "latency_ms": 15,
                "status": "forward",
                "result_hash": "b" * 64,
            }
        ],
    }


@pytest.fixture
def temp_receipt_file(sample_attestation_data: dict[str, Any]) -> Generator[Path, None, None]:
    """Create a temporary receipt JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_attestation_data, f)
        f.flush()
        yield Path(f.name)
    os.unlink(f.name)


@pytest.fixture
def temp_offline_receipt_file(
    sample_offline_attestation_data: dict[str, Any],
) -> Generator[Path, None, None]:
    """Create a temporary offline receipt JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_offline_attestation_data, f)
        f.flush()
        yield Path(f.name)
    os.unlink(f.name)
