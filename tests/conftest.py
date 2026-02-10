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
def sample_online_receipt_data() -> dict[str, Any]:
    """Sample online receipt JSON data (per spec)."""
    return {
        "attestationId": "att_test123abc",
        "evidenceHash": "a" * 64,  # Spec: evidenceHash (camelCase wire format)
        "timestamp": 1704110400000,  # Spec: Unix ms (2024-01-01T12:00:00Z)
        "leafIndex": 42,
        "treeSize": 100,
        "epochId": "epoch_001",
    }


@pytest.fixture
def sample_offline_receipt_data() -> dict[str, Any]:
    """Sample offline receipt JSON data."""
    return {
        "attestationId": "oatt_test123abc",
        "evidenceHash": "b" * 64,
        "timestamp": 1704110400000,  # Unix ms (2024-01-01T12:00:00Z)
        "serviceId": "test-service",
        "operationType": "inference",
        "payloadHash": "b" * 64,
        "signature": "c" * 128,  # Base64 Ed25519 signature
        "publicKey": "d" * 64,
        "isOffline": True,
        "witnessStatus": "UNVERIFIED",
    }


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
            "payloadHash": "a" * 64,
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
    """Sample control plane results data (L0 only).

    Note: piiPhi, jailbreak, and sampling fields removed - they're now captured
    in controls[] (for pii/jailbreak) or in Evidence/Review (for sampling).
    """
    return {
        "schema_version": "1.0",
        "policy": {
            "id": "policy-001",
            "version": "1.0",
            "model": {
                "modelId": "gpt-4",
                "provider": "openai",
                "systemPromptHash": "a" * 64,
            },
            "scope": {
                "environment": "production",
                "tags": ["healthcare", "hipaa"],
            },
        },
        "determination": {
            "action": "forwarded",
            "trigger": None,
            "confidence": 0.95,
        },
        "controls": [
            {
                "id": "pii-001",
                "type": "pii",
                "version": "1.0",
                "provider": "glacis",
                "latencyMs": 15,
                "status": "pass",
                "resultHash": "b" * 64,
            }
        ],
        "safety": {
            "overallRisk": 0.1,
            "scores": {"pii": 0.0, "jailbreak": 0.05},
        },
    }


@pytest.fixture
def temp_receipt_file(sample_online_receipt_data: dict[str, Any]) -> Generator[Path, None, None]:
    """Create a temporary receipt JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_online_receipt_data, f)
        f.flush()
        yield Path(f.name)
    os.unlink(f.name)


@pytest.fixture
def temp_offline_receipt_file(
    sample_offline_receipt_data: dict[str, Any],
) -> Generator[Path, None, None]:
    """Create a temporary offline receipt JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_offline_receipt_data, f)
        f.flush()
        yield Path(f.name)
    os.unlink(f.name)
