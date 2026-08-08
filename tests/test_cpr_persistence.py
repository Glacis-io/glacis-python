"""
Persistence of signed control_plane_results (CPR).

`control_plane_results` is *inside* the offline signature — `_attest_offline`
puts the whole structure into the payload it signs. Up to and including 0.8.0
neither storage backend persisted it, so a receipt reloaded from the store no
longer carried the content its own signature covered: the signed payload could
not be rebuilt and independent Ed25519 verification failed on a receipt that
had verified perfectly a moment earlier.

These tests pin the fix and, just as importantly, pin the honest behaviour for
rows written before the fix: the content is gone and cannot be recovered, so it
stays absent and the loss is named rather than reconstructed as "this receipt
had no control-plane results".
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

import pytest
from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

from glacis import Glacis
from glacis.storage import JsonStorageBackend, ReceiptStorage

SEED = bytes.fromhex(
    "9a3f1c0b7d2e4a5f8091c2d3e4f50617a8b9cadbec0d1e2f30415263748596a7"
)

CPR: dict[str, Any] = {
    "policy": {
        "id": "claims-policy",
        "version": "1.0",
        "environment": "production",
        "tags": ["healthcare"],
    },
    "determination": {"action": "forwarded"},
    "controls": [
        {
            "id": "pii-001",
            "type": "pii",
            "version": "1.0",
            "provider": "glacis",
            "latency_ms": 15,
            "status": "forward",
            "result_hash": "b" * 64,
            "stage": "input",
        }
    ],
}


# ---------------------------------------------------------------------------
# The independent verifier published on /verify/ — third-party recipe, no SDK
# internals. This is the check that a stored-and-reloaded receipt has to pass.
# ---------------------------------------------------------------------------


def signed_message(r: dict[str, Any]) -> bytes:
    body: dict[str, Any] = {
        "version": 1,
        "service_id": r["service_id"],
        "operation_type": r["operation_type"],
        "evidence_hash": r["evidence_hash"],
        "timestamp_ms": str(r["timestamp"]),
        "operation_id": r["operation_id"],
        "operation_sequence": r["operation_sequence"],
        "mode": "offline",
    }
    if r.get("control_plane_results"):
        body["control_plane_results"] = r["control_plane_results"]
    if r.get("supersedes"):
        body["supersedes"] = r["supersedes"]
    return json.dumps(body, separators=(",", ":"), sort_keys=True).encode()


def independently_verifies(receipt: Any) -> bool:
    """True when a third party holding only this receipt can verify it."""
    r = json.loads(json.dumps(receipt.model_dump(), default=str))
    try:
        VerifyKey(bytes.fromhex(r["public_key"])).verify(
            signed_message(r), bytes.fromhex(r["signature"])
        )
        return True
    except (BadSignatureError, ValueError):
        return False


def _client(tmp_path: Path, backend: str) -> Glacis:
    return Glacis(
        mode="offline",
        signing_seed=SEED,
        storage_backend=backend,
        storage_path=tmp_path / backend,
    )


def _attest_with_cpr(g: Glacis) -> Any:
    return g.attest(
        service_id="claims-triage",
        operation_type="classification",
        input={"claim_id": "C-1029"},
        output={"decision": "escalate"},
        control_plane_results=CPR,
    )


# ---------------------------------------------------------------------------
# Round trip: attest → store → reload → independent verification still passes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["sqlite", "json"])
class TestCprRoundTrip:
    def test_receipt_verifies_before_storage(self, tmp_path: Path, backend: str):
        g = _client(tmp_path, backend)
        try:
            assert independently_verifies(_attest_with_cpr(g))
        finally:
            g.close()

    def test_reloaded_receipt_still_carries_the_signed_cpr(
        self, tmp_path: Path, backend: str
    ):
        g = _client(tmp_path, backend)
        try:
            receipt = _attest_with_cpr(g)
            assert g._storage is not None
            reloaded = g._storage.get_receipt(receipt.id)
        finally:
            g.close()

        assert reloaded is not None
        assert reloaded.control_plane_results == CPR
        assert reloaded.cpr_hash == receipt.cpr_hash
        assert reloaded.cpr_recovery_error is None

    def test_reloaded_receipt_passes_independent_ed25519_verification(
        self, tmp_path: Path, backend: str
    ):
        """The regression that mattered: this failed on 0.8.0."""
        g = _client(tmp_path, backend)
        try:
            receipt = _attest_with_cpr(g)
            assert g._storage is not None
            reloaded = g._storage.get_receipt(receipt.id)
            assert reloaded is not None
            assert independently_verifies(reloaded)

            # get_last_receipt() goes through the same reconstruction.
            last = g.get_last_receipt()
            assert last is not None
            assert independently_verifies(last)

            # And the SDK's own verify() agrees.
            assert g.verify(reloaded).valid is True
        finally:
            g.close()

    def test_receipt_without_cpr_round_trips_and_reports_no_degradation(
        self, tmp_path: Path, backend: str
    ):
        """Absent CPR is a real state, not a degradation. Do not confuse them."""
        g = _client(tmp_path, backend)
        try:
            receipt = g.attest(
                service_id="svc",
                operation_type="inference",
                input={"a": 1},
                output={"b": 2},
            )
            assert receipt.cpr_hash is None
            assert g._storage is not None
            reloaded = g._storage.get_receipt(receipt.id)
        finally:
            g.close()

        assert reloaded is not None
        assert reloaded.control_plane_results is None
        assert reloaded.cpr_recovery_error is None
        assert independently_verifies(reloaded)

    def test_supersedes_and_cpr_together_round_trip(self, tmp_path: Path, backend: str):
        g = _client(tmp_path, backend)
        try:
            first = _attest_with_cpr(g)
            revised = g.attest(
                service_id="claims-triage",
                operation_type="classification",
                input={"claim_id": "C-1029"},
                output={"decision": "pay"},
                control_plane_results=CPR,
                supersedes=first.id,
            )
            assert g._storage is not None
            reloaded = g._storage.get_receipt(revised.id)
        finally:
            g.close()

        assert reloaded is not None
        assert reloaded.supersedes == first.id
        assert reloaded.control_plane_results == CPR
        assert independently_verifies(reloaded)


# ---------------------------------------------------------------------------
# Rows written before the fix: named degradation, never a fabricated absence
# ---------------------------------------------------------------------------


V4_SCHEMA = """
CREATE TABLE offline_receipts (
    attestation_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    service_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    evidence_hash TEXT NOT NULL,
    signature TEXT NOT NULL,
    public_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    input_preview TEXT,
    output_preview TEXT,
    metadata_json TEXT,
    operation_id TEXT,
    operation_sequence INTEGER DEFAULT 0,
    supersedes TEXT,
    cpr_hash TEXT
);
CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
INSERT INTO schema_version (version) VALUES (4);
"""


def _write_legacy_v4_row(db_path: Path, receipt: Any) -> None:
    """Write the row exactly as glacis 0.8.0 wrote it: no CPR column at all."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(V4_SCHEMA)
    conn.execute(
        """
        INSERT INTO offline_receipts
        (attestation_id, timestamp, service_id, operation_type, evidence_hash,
         signature, public_key, created_at, input_preview, output_preview,
         metadata_json, operation_id, operation_sequence, supersedes, cpr_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            receipt.id,
            receipt.timestamp,
            receipt.service_id,
            receipt.operation_type,
            receipt.evidence_hash,
            receipt.signature,
            receipt.public_key,
            "2026-08-08T00:00:00+00:00",
            None,
            None,
            None,
            receipt.operation_id,
            receipt.operation_sequence,
            receipt.supersedes,
            receipt.cpr_hash,
        ),
    )
    conn.commit()
    conn.close()


def _make_receipt_with_cpr(tmp_path: Path) -> Any:
    g = Glacis(
        mode="offline",
        signing_seed=SEED,
        storage_backend="json",
        storage_path=tmp_path / "scratch",
    )
    try:
        return _attest_with_cpr(g)
    finally:
        g.close()


class TestLegacyRowsDegradeByName:
    def test_v4_database_migrates_to_v5(self, tmp_path: Path):
        receipt = _make_receipt_with_cpr(tmp_path)
        db_path = tmp_path / "legacy.db"
        _write_legacy_v4_row(db_path, receipt)

        with ReceiptStorage(db_path) as storage:
            conn = storage._get_connection()
            cols = {r[1] for r in conn.execute("PRAGMA table_info(offline_receipts)")}
            assert "control_plane_json" in cols
            version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
            assert version == 5

    def test_legacy_row_does_not_fabricate_an_absence(self, tmp_path: Path):
        receipt = _make_receipt_with_cpr(tmp_path)
        db_path = tmp_path / "legacy.db"
        _write_legacy_v4_row(db_path, receipt)

        with ReceiptStorage(db_path) as storage:
            reloaded = storage.get_receipt(receipt.id)

        assert reloaded is not None
        # The signed content is gone. It is not invented, and it is not
        # quietly reported as "there was none".
        assert reloaded.control_plane_results is None
        assert reloaded.cpr_hash == receipt.cpr_hash
        assert reloaded.cpr_recovery_error is not None
        assert "cannot be rebuilt" in reloaded.cpr_recovery_error
        assert "0.8.0" in reloaded.cpr_recovery_error

    def test_legacy_row_fails_independent_verification(self, tmp_path: Path):
        """It fails, and that is correct — the signed bytes cannot be rebuilt."""
        receipt = _make_receipt_with_cpr(tmp_path)
        db_path = tmp_path / "legacy.db"
        _write_legacy_v4_row(db_path, receipt)

        with ReceiptStorage(db_path) as storage:
            reloaded = storage.get_receipt(receipt.id)

        assert reloaded is not None
        assert independently_verifies(receipt) is True
        assert independently_verifies(reloaded) is False

    def test_sdk_verify_fails_closed_with_the_reason(self, tmp_path: Path):
        receipt = _make_receipt_with_cpr(tmp_path)
        db_path = tmp_path / "legacy.db"
        _write_legacy_v4_row(db_path, receipt)

        g = Glacis(mode="offline", signing_seed=SEED, db_path=db_path)
        try:
            reloaded = g._storage.get_receipt(receipt.id)  # type: ignore[union-attr]
            assert reloaded is not None
            result = g.verify(reloaded)
        finally:
            g.close()

        assert result.valid is False
        assert result.signature_valid is False
        assert result.error == reloaded.cpr_recovery_error

    def test_legacy_row_without_cpr_hash_is_not_a_degradation(self, tmp_path: Path):
        """A 0.8.0 row for a receipt that never had CPR reads back clean."""
        g = Glacis(
            mode="offline",
            signing_seed=SEED,
            storage_backend="json",
            storage_path=tmp_path / "scratch",
        )
        try:
            receipt = g.attest(
                service_id="svc",
                operation_type="inference",
                input={"a": 1},
                output={"b": 2},
            )
        finally:
            g.close()

        db_path = tmp_path / "legacy-no-cpr.db"
        _write_legacy_v4_row(db_path, receipt)

        with ReceiptStorage(db_path) as storage:
            reloaded = storage.get_receipt(receipt.id)

        assert reloaded is not None
        assert reloaded.cpr_recovery_error is None
        assert independently_verifies(reloaded)

    def test_legacy_jsonl_line_degrades_by_name(self, tmp_path: Path):
        """A receipts.jsonl line written by 0.8.0 has no CPR key at all."""
        receipt = _make_receipt_with_cpr(tmp_path)
        base = tmp_path / "legacy-json"
        base.mkdir()
        legacy_line: dict[str, Optional[Any]] = {
            "attestation_id": receipt.id,
            "timestamp": receipt.timestamp,
            "service_id": receipt.service_id,
            "operation_type": receipt.operation_type,
            "evidence_hash": receipt.evidence_hash,
            "signature": receipt.signature,
            "public_key": receipt.public_key,
            "created_at": "2026-08-08T00:00:00+00:00",
            "input_preview": None,
            "output_preview": None,
            "metadata": None,
            "operation_id": receipt.operation_id,
            "operation_sequence": receipt.operation_sequence,
            "supersedes": receipt.supersedes,
            "cpr_hash": receipt.cpr_hash,
        }
        (base / "receipts.jsonl").write_text(json.dumps(legacy_line) + "\n")

        reloaded = JsonStorageBackend(base).get_receipt(receipt.id)
        assert reloaded is not None
        assert reloaded.control_plane_results is None
        assert reloaded.cpr_recovery_error is not None
        assert independently_verifies(reloaded) is False


class TestCprRecoveryErrorIsNotSigned:
    def test_the_marker_never_reaches_the_signed_payload(self, tmp_path: Path):
        """cpr_recovery_error is an SDK convenience, not receipt content."""
        g = _client(tmp_path, "json")
        try:
            receipt = _attest_with_cpr(g)
        finally:
            g.close()

        assert receipt.cpr_recovery_error is None
        body = json.loads(signed_message(json.loads(receipt.model_dump_json())).decode())
        assert "cpr_recovery_error" not in body
