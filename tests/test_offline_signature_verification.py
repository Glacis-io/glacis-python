"""
`verify()` performs a real Ed25519 check, not a structural one.

Up to and including 0.8.0 the SDK's offline verification compared the public
key derived from the *local seed* against the key on the receipt and called
that a signature check. It was not one: a receipt whose signed fields had been
edited — including its `control_plane_results`, edited straight in the store —
was reported `valid=True` although independent verification failed.

These tests pin the fix. Every signed field is tampered with **after storage**
and the receipt must fail, naming the signature. The unsigned fields must keep
behaving as /verify/what-a-check-proves/ says they do: editing them does not
break the check, and the one that does (`public_key`) breaks it for the reason
the page gives.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from nacl.signing import SigningKey

from glacis import Glacis
from glacis.crypto import get_ed25519_runtime, offline_signed_payload_for
from glacis.models import Attestation
from glacis.storage import ReceiptStorage
from glacis.verify import verify_offline

SEED = bytes.fromhex(
    "3f1c0b7d2e4a5f8091c2d3e4f50617a8b9cadbec0d1e2f30415263748596a79a"
)

CPR: dict[str, Any] = {
    "policy": {"id": "claims-policy", "version": "1.0", "tags": ["healthcare"]},
    "determination": {"action": "forwarded"},
    "controls": [
        {
            "id": "pii-001",
            "type": "pii",
            "status": "forward",
            "result_hash": "b" * 64,
        }
    ],
}


def _client(tmp_path: Path, backend: str = "sqlite") -> Glacis:
    return Glacis(
        mode="offline",
        signing_seed=SEED,
        storage_backend=backend,
        storage_path=tmp_path / backend,
    )


def _attest(g: Glacis, **kw: Any) -> Attestation:
    params: dict[str, Any] = dict(
        service_id="claims-triage",
        operation_type="classification",
        input={"claim_id": "C-1029"},
        output={"decision": "escalate"},
    )
    params.update(kw)
    return g.attest(**params)


# ---------------------------------------------------------------------------
# Honest receipts still pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["sqlite", "json"])
class TestHonestReceiptsPass:
    def test_fresh_receipt_verifies(self, tmp_path: Path, backend: str):
        g = _client(tmp_path, backend)
        try:
            result = g.verify(_attest(g))
        finally:
            g.close()
        assert result.valid is True
        assert result.signature_valid is True
        assert result.error is None

    def test_receipt_with_cpr_verifies(self, tmp_path: Path, backend: str):
        g = _client(tmp_path, backend)
        try:
            result = g.verify(_attest(g, control_plane_results=CPR))
        finally:
            g.close()
        assert result.valid is True
        assert result.error is None

    def test_receipt_reloaded_from_storage_verifies(self, tmp_path: Path, backend: str):
        g = _client(tmp_path, backend)
        try:
            receipt = _attest(g, control_plane_results=CPR)
            assert g._storage is not None
            reloaded = g._storage.get_receipt(receipt.id)
            assert reloaded is not None
            result = g.verify(reloaded)
            by_id = g.verify(receipt.id)
        finally:
            g.close()
        assert result.valid is True
        assert result.error is None  # cpr_hash still matches the reloaded CPR
        assert by_id.valid is True
        assert by_id.error is None

    def test_receipt_with_supersedes_verifies(self, tmp_path: Path, backend: str):
        g = _client(tmp_path, backend)
        try:
            first = _attest(g)
            second = _attest(g, supersedes=first.id, control_plane_results=CPR)
            result = g.verify(second)
        finally:
            g.close()
        assert result.valid is True


# ---------------------------------------------------------------------------
# Every signed field, tampered after storage
# ---------------------------------------------------------------------------


SIGNED_FIELD_TAMPERS: list[tuple[str, Any]] = [
    ("service_id", "some-other-service"),
    ("operation_type", "something-else"),
    ("evidence_hash", "0" * 64),
    ("operation_id", "00000000-0000-0000-0000-000000000000"),
    ("signature", "00" * 64),
]


class TestSignedFieldTamperingFailsClosed:
    @pytest.fixture()
    def stored(self, tmp_path: Path):
        g = _client(tmp_path)
        receipt = _attest(g, control_plane_results=CPR)
        assert g._storage is not None
        reloaded = g._storage.get_receipt(receipt.id)
        assert reloaded is not None
        yield g, reloaded
        g.close()

    @pytest.mark.parametrize("field,value", SIGNED_FIELD_TAMPERS)
    def test_signed_field_tamper_fails_naming_the_signature(
        self, stored, field: str, value: Any
    ):
        g, reloaded = stored
        tampered = reloaded.model_copy(deep=True)
        setattr(tampered, field, value)

        result = g.verify(tampered)

        assert result.valid is False
        assert result.signature_valid is False
        assert result.error is not None
        assert result.error.startswith("signature_invalid: ")

    def test_timestamp_tamper_fails(self, stored):
        g, reloaded = stored
        tampered = reloaded.model_copy(deep=True)
        assert tampered.timestamp is not None
        tampered.timestamp = tampered.timestamp + 1

        result = g.verify(tampered)

        assert result.valid is False
        assert result.error is not None
        assert result.error.startswith("signature_invalid: ")

    def test_operation_sequence_tamper_fails(self, stored):
        g, reloaded = stored
        tampered = reloaded.model_copy(deep=True)
        tampered.operation_sequence = reloaded.operation_sequence + 1

        assert g.verify(tampered).valid is False

    def test_cpr_content_tamper_fails(self, stored):
        """The load-bearing case: signed control-plane content, edited."""
        g, reloaded = stored
        tampered = reloaded.model_copy(deep=True)
        tampered.control_plane_results = {
            "policy": {"id": "claims-policy", "version": "1.0", "tags": ["healthcare"]},
            "determination": {"action": "blocked"},
            "controls": reloaded.control_plane_results["controls"],  # type: ignore[index]
        }

        result = g.verify(tampered)

        assert result.valid is False
        assert result.error is not None
        assert result.error.startswith("signature_invalid: ")

    def test_supersedes_added_to_a_receipt_signed_without_one_fails(self, stored):
        g, reloaded = stored
        tampered = reloaded.model_copy(deep=True)
        tampered.supersedes = "oatt_invented"

        assert g.verify(tampered).valid is False

    def test_public_key_swapped_alone_fails(self, stored):
        """Unsigned, and yet it breaks the check — the key derives the verifier."""
        g, reloaded = stored
        tampered = reloaded.model_copy(deep=True)
        tampered.public_key = bytes(SigningKey(b"\x11" * 32).verify_key).hex()

        result = g.verify(tampered)

        assert result.valid is False
        assert result.error is not None
        assert result.error.startswith("signature_invalid: ")

    def test_malformed_key_or_signature_is_structural_not_a_signature_verdict(
        self, stored
    ):
        g, reloaded = stored
        for field, value in (("public_key", "zzzz"), ("signature", "not-hex")):
            tampered = reloaded.model_copy(deep=True)
            setattr(tampered, field, value)
            result = g.verify(tampered)
            assert result.valid is False
            assert result.error is not None
            assert result.error.startswith("structural: ")

    def test_missing_timestamp_is_structural(self, stored):
        g, reloaded = stored
        tampered = reloaded.model_copy(deep=True)
        tampered.timestamp = None

        result = g.verify(tampered)

        assert result.valid is False
        assert result.error is not None
        assert result.error.startswith("structural: ")


# ---------------------------------------------------------------------------
# The unsigned fields keep behaving as the boundary page says
# ---------------------------------------------------------------------------


class TestUnsignedFieldsDoNotBreakTheCheck:
    @pytest.fixture()
    def stored(self, tmp_path: Path):
        g = _client(tmp_path)
        receipt = _attest(g, control_plane_results=CPR)
        yield g, receipt
        g.close()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("id", "oatt_not-the-real-id"),
            ("is_offline", True),
        ],
    )
    def test_unsigned_field_edit_still_verifies(self, stored, field: str, value: Any):
        g, receipt = stored
        tampered = receipt.model_copy(deep=True)
        setattr(tampered, field, value)

        result = g.verify(tampered)

        assert result.valid is True
        assert result.error is None

    def test_cpr_hash_is_unsigned_so_the_signature_still_verifies(self, stored):
        """...and the inconsistency is reported by name, not swallowed."""
        g, receipt = stored
        tampered = receipt.model_copy(deep=True)
        tampered.cpr_hash = "0" * 64

        result = g.verify(tampered)

        assert result.valid is True
        assert result.signature_valid is True
        assert result.error is not None
        assert result.error.startswith("cpr_hash_mismatch: ")

    def test_cpr_stripped_with_its_hash_left_behind_fails(self, stored):
        """cpr_hash orphaned: the signed content is gone, so the check fails."""
        g, receipt = stored
        tampered = receipt.model_copy(deep=True)
        tampered.control_plane_results = None

        result = g.verify(tampered)

        assert result.valid is False
        assert result.error is not None
        assert result.error.startswith("signature_invalid: ")
        assert "cpr_hash_orphaned" in result.error


# ---------------------------------------------------------------------------
# Tampering with the *store*, not the object — the threat model in the finding
# ---------------------------------------------------------------------------


class TestTamperedStorageFailsClosed:
    def _stored_receipt(self, tmp_path: Path) -> tuple[Path, Attestation]:
        db_path = tmp_path / "glacis.db"
        g = Glacis(mode="offline", signing_seed=SEED, db_path=db_path)
        try:
            receipt = _attest(g, control_plane_results=CPR)
        finally:
            g.close()
        return db_path, receipt

    def _reload_and_verify(self, db_path: Path, attestation_id: str):
        g = Glacis(mode="offline", signing_seed=SEED, db_path=db_path)
        try:
            assert g._storage is not None
            reloaded = g._storage.get_receipt(attestation_id)
            assert reloaded is not None
            return reloaded, g.verify(reloaded)
        finally:
            g.close()

    def test_cpr_edited_in_the_database_fails(self, tmp_path: Path):
        db_path, receipt = self._stored_receipt(tmp_path)

        forged = dict(CPR, determination={"action": "blocked"})
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE offline_receipts SET control_plane_json = ? WHERE attestation_id = ?",
            (json.dumps(forged, separators=(",", ":")), receipt.id),
        )
        conn.commit()
        conn.close()

        reloaded, result = self._reload_and_verify(db_path, receipt.id)

        assert reloaded.control_plane_results == forged
        assert result.valid is False
        assert result.error is not None
        assert result.error.startswith("signature_invalid: ")

    def test_cpr_and_cpr_hash_both_stripped_from_the_row_fails(self, tmp_path: Path):
        """No `cpr_recovery_error` is possible here — only the signature catches it."""
        db_path, receipt = self._stored_receipt(tmp_path)

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE offline_receipts SET control_plane_json = NULL, cpr_hash = NULL "
            "WHERE attestation_id = ?",
            (receipt.id,),
        )
        conn.commit()
        conn.close()

        reloaded, result = self._reload_and_verify(db_path, receipt.id)

        # The row now looks exactly like a receipt that never had CPR, so
        # nothing structural is left to notice. The signature notices.
        assert reloaded.control_plane_results is None
        assert reloaded.cpr_hash is None
        assert reloaded.cpr_recovery_error is None
        assert result.valid is False
        assert result.error is not None
        assert result.error.startswith("signature_invalid: ")

    def test_service_id_edited_in_the_database_fails(self, tmp_path: Path):
        db_path, receipt = self._stored_receipt(tmp_path)

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE offline_receipts SET service_id = ? WHERE attestation_id = ?",
            ("laundered-service", receipt.id),
        )
        conn.commit()
        conn.close()

        _, result = self._reload_and_verify(db_path, receipt.id)

        assert result.valid is False
        assert result.error is not None
        assert result.error.startswith("signature_invalid: ")

    def test_jsonl_line_edited_on_disk_fails(self, tmp_path: Path):
        base = tmp_path / "jsonl"
        g = Glacis(
            mode="offline", signing_seed=SEED, storage_backend="json", storage_path=base
        )
        try:
            receipt = _attest(g, control_plane_results=CPR)
        finally:
            g.close()

        path = base / "receipts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            if row.get("attestation_id") == receipt.id:
                row["control_plane_results"] = dict(
                    CPR, determination={"action": "blocked"}
                )
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        g2 = Glacis(
            mode="offline", signing_seed=SEED, storage_backend="json", storage_path=base
        )
        try:
            assert g2._storage is not None
            reloaded = g2._storage.get_receipt(receipt.id)
            assert reloaded is not None
            result = g2.verify(reloaded)
        finally:
            g2.close()

        assert result.valid is False
        assert result.error is not None
        assert result.error.startswith("signature_invalid: ")


# ---------------------------------------------------------------------------
# Verification needs no seed, and the CLI runs the same check
# ---------------------------------------------------------------------------


class TestVerificationNeedsNoSeed:
    def test_a_receipt_signed_by_another_key_verifies(self, tmp_path: Path):
        """The public key on the receipt is the verifier. That is all it takes."""
        other_seed = bytes.fromhex("11" * 32)
        g_other = Glacis(
            mode="offline",
            signing_seed=other_seed,
            storage_backend="json",
            storage_path=tmp_path / "other",
        )
        try:
            foreign = _attest(g_other, control_plane_results=CPR)
        finally:
            g_other.close()

        # A different client, a different seed, no shared secret.
        g = _client(tmp_path)
        try:
            result = g.verify(foreign)
        finally:
            g.close()

        assert result.valid is True
        assert result.signature_valid is True

    def test_module_level_verify_offline_agrees_with_the_client(self, tmp_path: Path):
        g = _client(tmp_path)
        try:
            receipt = _attest(g, control_plane_results=CPR)
            tampered = receipt.model_copy(deep=True)
            tampered.evidence_hash = "0" * 64
            client_ok = g.verify(receipt)
            client_bad = g.verify(tampered)
        finally:
            g.close()

        assert verify_offline(receipt).valid == client_ok.valid is True
        assert verify_offline(tampered).valid == client_bad.valid is False
        assert verify_offline(tampered).error == client_bad.error

    def test_runtime_verify_round_trips(self, tmp_path: Path):
        g = _client(tmp_path)
        try:
            receipt = _attest(g, control_plane_results=CPR)
        finally:
            g.close()

        runtime = get_ed25519_runtime()
        message = offline_signed_payload_for(receipt)

        assert runtime.verify(receipt.public_key, message, receipt.signature) is True
        assert runtime.verify(receipt.public_key, b"other bytes", receipt.signature) is False


class TestCliVerifiesForReal:
    def _write(self, tmp_path: Path, receipt: Attestation, **edits: Any) -> Path:
        doc = receipt.model_dump()
        doc.update(edits)
        path = tmp_path / "receipt.json"
        path.write_text(json.dumps(doc, indent=2, default=str))
        return path

    def _run(self, path: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "glacis", "verify", str(path)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )

    @pytest.fixture()
    def receipt(self, tmp_path: Path) -> Attestation:
        g = _client(tmp_path)
        try:
            return _attest(g, control_plane_results=CPR)
        finally:
            g.close()

    def test_honest_receipt_passes(self, tmp_path: Path, receipt: Attestation):
        result = self._run(self._write(tmp_path, receipt))
        assert result.returncode == 0
        assert "Status: VALID" in result.stdout
        assert "Signature: PASS" in result.stdout

    def test_zeroed_signature_now_fails(self, tmp_path: Path, receipt: Attestation):
        """0.8.0 printed VALID here. That was the whole complaint."""
        result = self._run(self._write(tmp_path, receipt, signature="00" * 64))
        assert result.returncode == 1
        assert "Status: INVALID" in result.stdout
        assert "signature_invalid" in result.stdout

    def test_edited_cpr_fails(self, tmp_path: Path, receipt: Attestation):
        result = self._run(
            self._write(
                tmp_path,
                receipt,
                control_plane_results=dict(CPR, determination={"action": "blocked"}),
            )
        )
        assert result.returncode == 1
        assert "signature_invalid" in result.stdout

    def test_edited_cpr_hash_passes_with_a_named_note(
        self, tmp_path: Path, receipt: Attestation
    ):
        result = self._run(self._write(tmp_path, receipt, cpr_hash="0" * 64))
        assert result.returncode == 0
        assert "Status: VALID" in result.stdout
        assert "cpr_hash_mismatch" in result.stdout


class TestLegacyStoreStillDegradesByName:
    def test_cpr_unrecoverable_is_named_separately_from_the_signature(
        self, tmp_path: Path
    ):
        """A pre-0.8.1 row cannot be checked at all — a distinct answer."""
        g = Glacis(
            mode="offline",
            signing_seed=SEED,
            storage_backend="json",
            storage_path=tmp_path / "scratch",
        )
        try:
            receipt = _attest(g, control_plane_results=CPR)
        finally:
            g.close()

        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """
            CREATE TABLE offline_receipts (
                attestation_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
                service_id TEXT NOT NULL, operation_type TEXT NOT NULL,
                evidence_hash TEXT NOT NULL, signature TEXT NOT NULL,
                public_key TEXT NOT NULL, created_at TEXT NOT NULL,
                input_preview TEXT, output_preview TEXT, metadata_json TEXT,
                operation_id TEXT, operation_sequence INTEGER DEFAULT 0,
                supersedes TEXT, cpr_hash TEXT
            );
            CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
            INSERT INTO schema_version (version) VALUES (4);
            """
        )
        conn.execute(
            "INSERT INTO offline_receipts (attestation_id, timestamp, service_id, "
            "operation_type, evidence_hash, signature, public_key, created_at, "
            "operation_id, operation_sequence, cpr_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                receipt.id,
                receipt.timestamp,
                receipt.service_id,
                receipt.operation_type,
                receipt.evidence_hash,
                receipt.signature,
                receipt.public_key,
                "2026-08-08T00:00:00+00:00",
                receipt.operation_id,
                receipt.operation_sequence,
                receipt.cpr_hash,
            ),
        )
        conn.commit()
        conn.close()

        with ReceiptStorage(db_path) as storage:
            reloaded = storage.get_receipt(receipt.id)
        assert reloaded is not None

        result = verify_offline(reloaded)
        assert result.valid is False
        assert result.error is not None
        assert result.error.startswith("cpr_unrecoverable: ")
