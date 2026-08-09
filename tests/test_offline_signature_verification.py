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

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from nacl.signing import SigningKey

from glacis import Glacis
from glacis.crypto import get_ed25519_runtime, offline_signed_payload_for
from glacis.models import Attestation, OfflineVerifyResult, VerifyResult
from glacis.storage import ReceiptStorage
from glacis.verify import verify_command, verify_offline
from tests.conftest import V4_EVIDENCE_TABLE

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

    def test_uncanonicalizable_cpr_returns_a_verdict_not_a_crash(self, stored):
        """verify() runs on receipts strangers hand you. A NaN/Infinity in the
        control_plane_results (legal JSON via the common extension, and a value
        json.load accepts) must yield valid=False, never an unhandled exception
        that kills the caller. Regression for the uncaught hash_payload raise in
        the cpr-degradation step, which ran before the signature check."""
        g, reloaded = stored
        tampered = reloaded.model_copy(deep=True)
        tampered.control_plane_results = {"determination": {"score": float("nan")}}

        result = g.verify(tampered)  # must not raise

        assert result.valid is False
        assert result.error is not None

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
            # `is_offline` used to be listed here as ("is_offline", True) on a
            # receipt whose is_offline was already True — a no-op that asserted
            # nothing. It is not an inert field at all: it selects the
            # verification route. A real flip of it is exercised through public
            # dispatch in TestIsOfflineCannotBypassTheSignatureCheck below.
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
# The unsigned routing fields cannot skip the signature check
# ---------------------------------------------------------------------------


def _stub_online(g: Glacis, response: dict[str, Any]) -> list[str]:
    """Stub the online transport. Returns the ids it gets asked about.

    Nothing here opens a socket: `_request_with_retry` is the seam between the
    SDK and httpx, and replacing it is what "the server said X" means in these
    tests. The dispatch under test is the public one — `Glacis.verify()` — not
    any internal verifier.
    """
    asked: list[str] = []

    def _stub(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        asked.append(url.rsplit("/", 1)[-1])
        return response

    g._request_with_retry = _stub  # type: ignore[method-assign, assignment]
    return asked


def _log_entry(**fields: Any) -> dict[str, Any]:
    entry = {
        "entryId": "att_real-online-attestation",
        "timestamp": "2026-08-08T00:00:00Z",
        "orgId": "org_real",
        "serviceId": "claims-triage",
        "operationType": "classification",
        "evidenceHash": "9" * 64,
        "signature": "7" * 128,
        "leafIndex": 12,
        "leafHash": "8" * 64,
    }
    entry.update(fields)
    return {
        "valid": True,
        "attestation": entry,
        "verification": {"signatureValid": True, "proofValid": True},
    }


class TestIsOfflineCannotBypassTheSignatureCheck:
    """`is_offline` and `id` are unsigned. Neither may skip a signature.

    Codex's pass-4 finding: `Glacis.verify(Attestation)` chose its verifier
    from the unsigned `is_offline` field, so setting it to False and pointing
    `id` at a valid online attestation routed the call to a server lookup and
    the supplied object's own bad signature was never examined. The lookup
    answered about the id and the caller read it as an answer about the bytes
    they held: `valid=True` for a forged receipt.

    Every test here goes through the public `Glacis.verify()`.
    """

    @pytest.fixture()
    def honest(self, tmp_path: Path):
        g = _client(tmp_path)
        try:
            yield g, _attest(g, control_plane_results=CPR)
        finally:
            g.close()

    def test_a_forged_receipt_reclassified_as_online_fails_naming_the_signature(
        self, honest
    ):
        """The pass-4 attack, executed exactly as reported."""
        g, receipt = honest
        forged = receipt.model_copy(deep=True)
        forged.control_plane_results = dict(CPR, determination={"action": "blocked"})
        forged.is_offline = False
        forged.id = "att_real-online-attestation"

        asked = _stub_online(g, _log_entry())
        result = g.verify(forged)

        assert result.valid is False
        assert result.error is not None
        assert result.error.startswith("signature_invalid: ")
        # The lookup was allowed to happen and simply did not rescue it.
        assert asked == ["att_real-online-attestation"]
        assert "unbound: " in result.error

    def test_a_zeroed_signature_reclassified_as_online_still_fails(self, honest):
        g, receipt = honest
        forged = receipt.model_copy(
            deep=True, update={"signature": "00" * 64, "is_offline": False}
        )
        forged.id = "att_real-online-attestation"

        _stub_online(g, _log_entry())
        result = g.verify(forged)

        assert result.valid is False
        assert "signature_invalid: " in (result.error or "")

    def test_the_server_saying_valid_does_not_make_the_object_valid(self, honest):
        """`valid` describes the supplied bytes, never the id they claim."""
        g, receipt = honest
        forged = receipt.model_copy(
            deep=True, update={"evidence_hash": "0" * 64, "is_offline": False}
        )
        forged.id = "att_real-online-attestation"

        _stub_online(g, _log_entry(valid=True))
        result = g.verify(forged)

        assert result.valid is False
        # Nothing from the server's record is reported for bytes it did not
        # describe — not the org, not the proof, not the tree head.
        assert isinstance(result, OfflineVerifyResult)
        assert result.attestation is not None
        assert result.attestation.id == "att_real-online-attestation"

    def test_flipping_is_offline_on_an_honest_receipt_still_verifies(self, honest):
        """A real flip, not the no-op the pass-4 review found at line 257."""
        g, receipt = honest
        flipped = receipt.model_copy(deep=True, update={"is_offline": False})
        assert receipt.is_offline is True

        asked = _stub_online(g, _log_entry())
        result = g.verify(flipped)

        assert result.valid is True
        assert isinstance(result, OfflineVerifyResult)
        assert result.signature_valid is True
        # The routing is named rather than silently taken: the id was looked
        # up, the answer described a different attestation, so it was dropped.
        assert asked == [receipt.id]
        assert "unbound: " in (result.error or "")
        assert "Only the supplied attestation's own Ed25519 signature was checked" in (
            result.error or ""
        )

    def test_an_object_that_binds_to_the_log_entry_gets_the_servers_verdict(
        self, honest
    ):
        """The other half: binding is what lets a lookup speak for an object."""
        g, receipt = honest
        flipped = receipt.model_copy(deep=True, update={"is_offline": False})

        _stub_online(
            g,
            _log_entry(
                entryId=receipt.id,
                signature=receipt.signature,
                evidenceHash=receipt.evidence_hash,
                serviceId=receipt.service_id,
                operationType=receipt.operation_type,
            ),
        )
        result = g.verify(flipped)

        assert isinstance(result, VerifyResult)
        assert result.valid is True
        assert (result.error or "").startswith("bound: ")
        # And it says what the entry cannot vouch for.
        assert "control_plane_results" in (result.error or "")

    def test_a_bound_object_whose_record_is_invalid_is_invalid(self, honest):
        g, receipt = honest
        flipped = receipt.model_copy(deep=True, update={"is_offline": False})

        _stub_online(
            g,
            dict(
                _log_entry(
                    entryId=receipt.id,
                    signature=receipt.signature,
                    evidenceHash=receipt.evidence_hash,
                    serviceId=receipt.service_id,
                    operationType=receipt.operation_type,
                ),
                valid=False,
                error="revoked",
            ),
        )
        result = g.verify(flipped)

        assert result.valid is False
        assert "revoked" in (result.error or "")

    def test_a_bound_object_whose_own_check_failed_is_invalid(self, honest):
        """Binding compares strings; a string-equal signature is not a verified
        one. The pass-5 review found the server's valid=True being returned
        here — an object copying a real entry's signature and evidence_hash
        bound, and its own failed Ed25519 check was demoted to a note."""
        g, receipt = honest
        forged = receipt.model_copy(
            deep=True,
            update={"is_offline": False, "timestamp": receipt.timestamp + 1},
        )

        _stub_online(
            g,
            _log_entry(
                entryId=receipt.id,
                signature=receipt.signature,
                evidenceHash=receipt.evidence_hash,
                serviceId=receipt.service_id,
                operationType=receipt.operation_type,
            ),
        )
        result = g.verify(forged)

        assert isinstance(result, OfflineVerifyResult)
        assert result.valid is False
        assert "signature_invalid" in (result.error or "")
        assert "bound-but-unverified: " in (result.error or "")
        # Nothing of the server's answer is laundered onto the failed bytes.
        assert not (result.error or "").startswith("bound: ")

    def test_a_bound_object_that_cannot_be_checked_stays_unverified(self, honest):
        """The structural half of the same hole: an undecodable key means the
        local check could not run, and binding must not stand in for it."""
        g, receipt = honest
        stripped = receipt.model_copy(
            deep=True, update={"is_offline": False, "public_key": "z" * 64}
        )

        _stub_online(
            g,
            _log_entry(
                entryId=receipt.id,
                signature=receipt.signature,
                evidenceHash=receipt.evidence_hash,
                serviceId=receipt.service_id,
                operationType=receipt.operation_type,
            ),
        )
        result = g.verify(stripped)

        assert isinstance(result, OfflineVerifyResult)
        assert result.valid is False
        assert (result.error or "").startswith("structural: ")
        assert "bound-but-unverified: " in (result.error or "")

    def test_flipping_is_offline_to_true_never_reaches_the_network(self, honest):
        """Flipping the other way narrows the check; it cannot widen it."""
        g, receipt = honest
        online_shaped = Attestation(
            id="att_real-online-attestation",
            service_id="claims-triage",
            operation_type="classification",
            evidence_hash="9" * 64,
            public_key=receipt.public_key,
            signature="7" * 128,
            is_offline=True,
        )

        asked = _stub_online(g, _log_entry())
        result = g.verify(online_shaped)

        assert asked == []
        assert result.valid is False
        assert (result.error or "").startswith("structural: ")

    def test_the_cli_reclassification_path_fails_the_same_way(self, tmp_path: Path):
        """`python -m glacis verify` had the same two-unsigned-field bypass."""
        g = _client(tmp_path)
        try:
            receipt = _attest(g, control_plane_results=CPR)
        finally:
            g.close()

        doc = receipt.model_dump()
        doc["control_plane_results"] = dict(CPR, determination={"action": "blocked"})
        doc["is_offline"] = False
        doc["id"] = "att_real-online-attestation"
        path = tmp_path / "reclassified.json"
        path.write_text(json.dumps(doc, indent=2, default=str))

        calls: list[str] = []

        def _lookup(attestation_id: str, base_url: str) -> VerifyResult:
            calls.append(attestation_id)
            return VerifyResult.model_validate(_log_entry())

        args = argparse.Namespace(receipt=str(path), base_url="https://example.invalid")
        with mock.patch("glacis.verify.verify_online", side_effect=_lookup):
            with pytest.raises(SystemExit) as exit_info:
                verify_command(args)

        assert exit_info.value.code == 1
        assert calls == ["att_real-online-attestation"]


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
            + V4_EVIDENCE_TABLE
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
