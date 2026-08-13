"""Tests for glacis.witness — the hosted-mint verification and label machine.

The anchor is the cross-repo vector ``portal-permalink-envelope-v1.json``
(committed verbatim from glacis-web-prod ``assets/receipts/``; a byte-identical
twin lives in the portal branch). It carries a real signed tree head and a
real inclusion proof, so a WITNESSED verdict here proves this SDK, the portal,
and glacis.io/verify agree on every byte of the contract: the detector shape,
the STH signing preimage, and the RFC 6962 leaf/path recompute.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import nacl.signing
import pytest

from glacis.witness import (
    HOSTED_TASK_CLASSES,
    classify_envelope,
    is_projected_witness_receipt,
    recompute_log_root,
    rfc6962_leaf_hash,
    sth_signing_preimage,
    verify_sth_signature,
)
from tests._reference_log import ReferenceLog, leaf_hash, mth, path

FIXTURE = Path(__file__).parent / "fixtures" / "portal-permalink-envelope-v1.json"

# The canonical copy this fixture was taken from. When the sibling repo is
# present (dev machines), assert the two never drift; absent (CI), skip.
CROSS_REPO_FIXTURE = Path(
    "/Users/joe/00_dev/keep-receipts/glacis-web-prod/assets/receipts/"
    "portal-permalink-envelope-v1.json"
)


@pytest.fixture
def vector() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def envelope(vector: dict) -> dict:
    return json.loads(vector["envelope_json"])


class TestCrossRepoVector:
    """The committed permalink vector must verify end to end."""

    def test_fixture_matches_cross_repo_copy(self):
        if not CROSS_REPO_FIXTURE.exists():
            pytest.skip("cross-repo checkout not present")
        assert FIXTURE.read_bytes() == CROSS_REPO_FIXTURE.read_bytes()

    def test_detector_parity(self, envelope: dict):
        # verify.html isProjectedWitnessReceipt: receipt_id/task/outcome
        # strings, and NO signature string.
        assert is_projected_witness_receipt(envelope["receipt"])

    def test_envelope_shape(self, envelope: dict):
        assert envelope["v"] == 1
        assert envelope["inclusion"]["status"] == "included"
        # Nulls are present, not omitted — the projection emits every key.
        assert envelope["receipt"]["prev"] is None
        assert envelope["receipt"]["commitments"]["response"] is None

    def test_sth_signature_under_fixture_log_key(self, vector: dict, envelope: dict):
        sth = envelope["inclusion"]["sth"]
        assert (
            verify_sth_signature(sth, [vector["log_public_key_hex"]])
            == vector["log_public_key_hex"]
        )

    def test_full_classification_is_witnessed(self, vector: dict, envelope: dict):
        v = classify_envelope(envelope, [vector["log_public_key_hex"]])
        assert v.witness_status == "WITNESSED"
        assert v.inclusion_verified and v.sth_signature_verified
        assert v.log_public_key_hex == vector["log_public_key_hex"]
        assert v.reason is None

    def test_wrong_key_is_not_witnessed(self, envelope: dict):
        other = bytes(nacl.signing.SigningKey(b"\x09" * 32).verify_key).hex()
        v = classify_envelope(envelope, [other])
        assert v.witness_status == "LOGGED_UNVERIFIED"
        assert "did not verify under any configured log key" in v.reason

    def test_no_key_configured_is_honest(self, envelope: dict):
        v = classify_envelope(envelope, [])
        assert v.witness_status == "LOGGED_UNVERIFIED"
        assert "GLACIS_LOG_PUBLIC_KEY_HEX" in v.reason
        assert not v.sth_signature_verified

    def test_tampered_receipt_id_contradicts(self, vector: dict, envelope: dict):
        envelope["receipt"]["receipt_id"] = "0" * 64
        v = classify_envelope(envelope, [vector["log_public_key_hex"]])
        assert v.witness_status == "LOGGED_UNVERIFIED"
        assert v.contradicted
        assert v.sth_signature_verified  # the key vouched; the proof failed

    def test_tampered_audit_path_contradicts(self, vector: dict, envelope: dict):
        proof = envelope["inclusion"]["inclusion_proof"]
        proof["audit_path"][0] = "f" * 64
        v = classify_envelope(envelope, [vector["log_public_key_hex"]])
        assert v.witness_status == "LOGGED_UNVERIFIED"
        assert v.contradicted

    def test_pending_inclusion_is_unverified(self, vector: dict, envelope: dict):
        envelope["inclusion"] = {"status": "pending", "eta_ms": 1000}
        v = classify_envelope(envelope, [vector["log_public_key_hex"]])
        assert v.witness_status == "LOGGED_UNVERIFIED"
        assert "pending" in v.reason

    def test_leaf_index_disagreement_contradicts(self, vector: dict, envelope: dict):
        envelope["inclusion"]["leaf_index"] = 4  # proof says 5
        v = classify_envelope(envelope, [vector["log_public_key_hex"]])
        assert v.witness_status == "LOGGED_UNVERIFIED"
        assert v.contradicted


class TestSthPreimage:
    """The tree head signs declaration-order compact JSON — not sorted keys."""

    HEAD = {
        "log_id": "glacis-log/test",
        "tree_size": 12,
        "root_hash": "ab" * 32,
        "timestamp_ms": 1_800_000_000_000,
    }

    def test_declaration_order_bytes(self):
        assert sth_signing_preimage(self.HEAD) == (
            b'{"log_id":"glacis-log/test","tree_size":12,'
            b'"root_hash":"' + b"ab" * 32 + b'","timestamp_ms":1800000000000}'
        )

    def test_sorted_keys_would_be_different_bytes(self):
        sorted_bytes = json.dumps(
            self.HEAD, separators=(",", ":"), sort_keys=True
        ).encode()
        assert sth_signing_preimage(self.HEAD) != sorted_bytes

    def test_signature_over_sorted_preimage_fails(self):
        key = nacl.signing.SigningKey(b"\x02" * 32)
        sorted_bytes = json.dumps(
            self.HEAD, separators=(",", ":"), sort_keys=True
        ).encode()
        sth = dict(self.HEAD)
        sth["signature"] = base64.b64encode(key.sign(sorted_bytes).signature).decode()
        assert verify_sth_signature(sth, [bytes(key.verify_key).hex()]) is None

    def test_signature_over_declaration_preimage_passes(self):
        key = nacl.signing.SigningKey(b"\x02" * 32)
        sth = dict(self.HEAD)
        sig = key.sign(sth_signing_preimage(sth)).signature
        sth["signature"] = base64.b64encode(sig).decode()
        key_hex = bytes(key.verify_key).hex()
        assert verify_sth_signature(sth, [key_hex]) == key_hex

    def test_non_ascii_log_id_uses_utf8_not_escapes(self):
        # JSON.stringify emits UTF-8 via TextEncoder; \uXXXX escapes would be
        # different bytes and a signature that never checks out.
        head = dict(self.HEAD, log_id="glacis-log/é")
        assert "glacis-log/é".encode("utf-8") in sth_signing_preimage(head)


class TestRfc6962Recompute:
    """The iterative verifier against an independent recursive producer."""

    @staticmethod
    def _leaves(n: int) -> list[bytes]:
        return [
            leaf_hash(hashlib.sha256(f"receipt-{i}".encode()).digest())
            for i in range(n)
        ]

    def test_all_indices_all_sizes(self):
        for n in range(1, 17):
            leaves = self._leaves(n)
            root = mth(leaves)
            for m in range(n):
                audit = [h.hex() for h in path(m, leaves)]
                got = recompute_log_root(leaves[m], m, audit, n)
                assert got == root, f"(m={m}, n={n})"

    def test_single_leaf_tree(self):
        leaves = self._leaves(1)
        assert recompute_log_root(leaves[0], 0, [], 1) == leaves[0]

    def test_path_too_short_refused(self):
        leaves = self._leaves(8)
        audit = [h.hex() for h in path(3, leaves)]
        assert recompute_log_root(leaves[3], 3, audit[:-1], 8) is None

    def test_path_too_long_refused(self):
        leaves = self._leaves(8)
        audit = [h.hex() for h in path(3, leaves)] + ["c" * 64]
        assert recompute_log_root(leaves[3], 3, audit, 8) is None

    def test_wrong_index_fails(self):
        leaves = self._leaves(8)
        root = mth(leaves)
        audit = [h.hex() for h in path(3, leaves)]
        assert recompute_log_root(leaves[3], 2, audit, 8) != root

    def test_index_out_of_range_refused(self):
        leaves = self._leaves(4)
        assert recompute_log_root(leaves[0], 4, [], 4) is None
        assert recompute_log_root(leaves[0], 0, [], 0) is None

    def test_leaf_domain_separation(self):
        data = b"\xaa" * 32
        assert rfc6962_leaf_hash(data) != hashlib.sha256(data).digest()
        assert rfc6962_leaf_hash(data) == hashlib.sha256(b"\x00" + data).digest()


class TestReferenceLogSelfConsistency:
    """The test helper's signed records classify WITNESSED under its own key
    — the harness the mocked-gateway integration tests stand on."""

    def test_reference_log_round_trip(self):
        log = ReferenceLog()
        receipt_hash = hashlib.sha256(b"the-receipt").hexdigest()
        for i in range(7):
            log.append(hashlib.sha256(b"filler-%d" % i).hexdigest())
        idx = log.append(receipt_hash)
        envelope = {
            "v": 1,
            "receipt": {
                "receipt_id": receipt_hash,
                "prev": None,
                "task": "default",
                "outcome": "ADMITTED",
                "governed": True,
                "charter_version": "1.0.0",
                "charter_hash": "b" * 64,
                "commitments": {"request": "a" * 64, "response": None},
                "latency_ms": 3,
                "at_ms": 1_800_000_000_000,
            },
            "inclusion": log.inclusion(idx),
        }
        v = classify_envelope(envelope, [log.public_key_hex])
        assert v.witness_status == "WITNESSED"


class TestTaskClasses:
    def test_the_eight_public_safe_labels(self):
        assert HOSTED_TASK_CLASSES == {
            "cost-batch-summarize",
            "quality-chat",
            "regulated-claims",
            "regulated-intake",
            "regulated-fresh-deploy",
            "safety-chat",
            "safety-drift",
            "default",
        }
