"""L2 end-to-end: the "instrument your product" promise, proven offline.

The launch sells this rung directly: `pip install glacis`, sign a receipt
locally with your own key, and have it verify — no network, no account, no
Glacis in the middle. This test is that promise as an executable guard, run
against the same public API a customer imports. It is deliberately written to
the shipped surface (`get_ed25519_runtime`, `offline_signed_payload_for`,
`verify_offline`), so a regression that broke the round trip fails here rather
than on a customer's first receipt.

The tamper case pins the 0.8.1 correctness fix (GLA-1711): the signature — not
the shape of the receipt — decides validity. Altering a signed field after
signing must read back as `signature_invalid`, and altering the UNSIGNED
`cpr_hash` must NOT flip validity (it is a degradation note, never a verdict).
"""

import os
import time

from glacis.crypto import get_ed25519_runtime
from glacis.models import Attestation
from glacis.verify import offline_signed_payload_for, verify_offline


def _signed_receipt():
    rt = get_ed25519_runtime()
    seed = os.urandom(32)
    att = Attestation(
        id="att_l2",
        service_id="user-service",
        operation_type="inference",
        evidence_hash="ab" * 32,
        operation_id="op_1",
        operation_sequence=1,
        timestamp=int(time.time() * 1000),
        public_key=rt.get_public_key_hex(seed),
        signature="00" * 64,
        control_plane_results={"score": 0.02, "verdict": "allow"},
        is_offline=True,
    )
    att.signature = rt.sign(seed, offline_signed_payload_for(att)).hex()
    return att


def test_honest_offline_receipt_verifies():
    result = verify_offline(_signed_receipt())
    assert result.valid is True, result.error
    assert result.error is None


def test_tampered_signed_field_is_rejected():
    att = _signed_receipt()
    att.control_plane_results = {"score": 0.02, "verdict": "DENY-flipped"}
    result = verify_offline(att)
    assert result.valid is False
    assert (result.error or "").startswith("signature_invalid")


def test_unsigned_cpr_hash_edit_does_not_flip_validity():
    # cpr_hash is outside the signature by design (0.8.1): editing it is a
    # named degradation, never a verdict. The signature still decides.
    att = _signed_receipt()
    att.cpr_hash = "ff" * 32
    result = verify_offline(att)
    assert result.valid is True, result.error
