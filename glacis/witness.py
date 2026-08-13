"""Server-attested witness verification for hosted mints.

The hosted mint surface (``POST {base}/v1/govern`` on api.glacis.io) returns a
projected receipt plus a transparency-log record::

    {"receipt": <projected receipt>,
     "inclusion": {"status": "included", "leaf_index": ...,
                   "inclusion_proof": {"leaf_index": ..., "audit_path": [...]},
                   "sth": {"log_id": ..., "tree_size": ..., "root_hash": ...,
                           "timestamp_ms": ..., "signature": ...,
                           "countersignature": ...}}}

This module checks that record locally, the same way glacis.io/verify does
(verify.html: ``sthSigningPreimage``, ``rfc6962Leaf``/``rfc6962Node``,
``recomputeLogRoot``, ``isProjectedWitnessReceipt``) and the same way the
monorepo reference implementation does (mvp-product ``packages/glacis-py``
``merkle.py``), fail-closed:

* The log leaf appended for a governed call is the 32 **raw bytes** of the
  receipt hash, which the projected receipt returns verbatim as
  ``receipt_id``. Recomputing the leaf from the receipt in front of us is the
  binding — it stops a proof for some other entry vouching for this one.
* RFC 6962 domain separation: ``leaf = sha256(0x00 || data)``,
  ``node = sha256(0x01 || left || right)``.
* The tree head signs the compact JSON of its first four fields **in
  declaration order** — ``log_id``, ``tree_size``, ``root_hash``,
  ``timestamp_ms``. Sorted keys would be different bytes and a signature that
  never checks out.
* The signature is base64 Ed25519 under a log key the caller **configures**
  (``GLACIS_LOG_PUBLIC_KEY_HEX`` or pinned keys) — never a key carried in the
  payload itself, because whoever wrote the payload wrote that key too. The
  SDK ships no baked-in production key. The ``countersignature`` (a dev
  witness key, not pinned) is ignored for verification in 0.9.0.

Label state machine (fail-closed):

* ``LOG_INCLUSION_VERIFIED`` — the inclusion proof recomputes from this
  receipt's own leaf to a root that a *configured* log key signed. That is
  the CEILING for this shape: the leaf commits only to the opaque
  ``receipt_id``, so the projection fields beside it (``task``, ``outcome``,
  ``commitments``) are NOT covered by the proof. glacis.io/verify applies
  the same cap — its verdict for this shape is never green — and this SDK
  never claims more than that page would for the same bytes.
* ``LOGGED_UNVERIFIED`` — the mint returned, but the record could not be
  verified here. The reason is always named.
* ``WITNESSED`` — reserved for shapes carrying a verified signature over
  their semantic fields; never produced by this module today.
* ``SELF_SIGNED`` — offline receipts (see ``Attestation.witness_status``);
  never produced by this module.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from typing import Any, Mapping, Optional

from glacis.crypto import CryptoError, get_ed25519_runtime
from glacis.models import WitnessVerification

__all__ = [
    "HOSTED_TASK_CLASSES",
    "classify_envelope",
    "is_projected_witness_receipt",
    "recompute_log_root",
    "rfc6962_leaf_hash",
    "rfc6962_node_hash",
    "sth_signing_preimage",
    "verify_sth_signature",
]

#: The gateway's public-safe task-class label set. ``POST /v1/govern`` rejects
#: anything else (deny-unknown), so the SDK validates before spending a mint.
HOSTED_TASK_CLASSES = frozenset(
    {
        "cost-batch-summarize",
        "quality-chat",
        "regulated-claims",
        "regulated-intake",
        "regulated-fresh-deploy",
        "safety-chat",
        "safety-drift",
        "default",
    }
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# =============================================================================
# RFC 6962 primitives
# =============================================================================


def rfc6962_leaf_hash(data: bytes) -> bytes:
    """RFC 6962 §2.1 leaf hash: ``sha256(0x00 || data)``."""
    return hashlib.sha256(b"\x00" + data).digest()


def rfc6962_node_hash(left: bytes, right: bytes) -> bytes:
    """RFC 6962 §2.1 node hash: ``sha256(0x01 || left || right)``."""
    return hashlib.sha256(b"\x01" + left + right).digest()


def recompute_log_root(
    leaf: bytes, leaf_index: int, audit_path: list[str], tree_size: int
) -> Optional[bytes]:
    """RFC 9162 §2.1.3.2 — recompute a root from a leaf hash and audit path.

    Returns the computed root bytes, or ``None`` when the path is malformed
    for this ``(leaf_index, tree_size)``: a path too short or too long is
    refused rather than allowed to land on the right root by coincidence.
    Mirrors verify.html ``recomputeLogRoot`` step for step.
    """
    if tree_size <= 0 or leaf_index >= tree_size:
        return None
    node_index = leaf_index
    last_index = tree_size - 1
    r = leaf
    for step in audit_path:
        if last_index == 0:
            return None  # path longer than the tree needs
        if not isinstance(step, str) or not _HEX64_RE.match(step):
            return None
        p = bytes.fromhex(step)
        if node_index % 2 == 1 or node_index == last_index:
            r = rfc6962_node_hash(p, r)
            if node_index % 2 == 0:
                while node_index % 2 == 0 and node_index != 0:
                    node_index //= 2
                    last_index //= 2
        else:
            r = rfc6962_node_hash(r, p)
        node_index //= 2
        last_index //= 2
    if last_index != 0:
        return None  # path shorter than the tree needs
    return r


# =============================================================================
# Signed tree head
# =============================================================================


def sth_signing_preimage(sth: Mapping[str, Any]) -> bytes:
    """The bytes a log signs over its tree head.

    Compact JSON of the head minus both signature fields, in the log's own
    declaration order: ``log_id``, ``tree_size``, ``root_hash``,
    ``timestamp_ms``. The order is load-bearing — do not "tidy" it into
    sorted order — and ``ensure_ascii=False`` is load-bearing too: the
    browser verifier encodes ``JSON.stringify`` output as UTF-8, not with
    ``\\uXXXX`` escapes.
    """
    return json.dumps(
        {
            "log_id": sth["log_id"],
            "tree_size": sth["tree_size"],
            "root_hash": sth["root_hash"],
            "timestamp_ms": sth["timestamp_ms"],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _read_log_sth(value: Any) -> Optional[dict[str, Any]]:
    """Shape check for a signed tree head (parity with verify.html readLogSth)."""
    if not isinstance(value, dict):
        return None
    if not isinstance(value.get("log_id"), str):
        return None
    tree_size = value.get("tree_size")
    if not isinstance(tree_size, int) or isinstance(tree_size, bool) or tree_size < 0:
        return None
    root_hash = value.get("root_hash")
    if not isinstance(root_hash, str) or not _HEX64_RE.match(root_hash):
        return None
    ts = value.get("timestamp_ms")
    if not isinstance(ts, int) or isinstance(ts, bool) or ts < 0:
        return None
    sig = value.get("signature")
    if not isinstance(sig, str) or not sig:
        return None
    return value


def _read_log_proof(value: Any) -> Optional[dict[str, Any]]:
    """Shape check for an inclusion proof (parity with verify.html readLogProof)."""
    if not isinstance(value, dict):
        return None
    leaf_index = value.get("leaf_index")
    if not isinstance(leaf_index, int) or isinstance(leaf_index, bool) or leaf_index < 0:
        return None
    audit_path = value.get("audit_path")
    if not isinstance(audit_path, list):
        return None
    for h in audit_path:
        if not isinstance(h, str) or not _HEX64_RE.match(h):
            return None
    return value


def verify_sth_signature(
    sth: Mapping[str, Any], log_public_keys: list[str]
) -> Optional[str]:
    """Verify the tree head's Ed25519 signature under the configured keys.

    Returns the hex of the key that verified, or ``None``. The signature is
    base64 on the wire; keys are 32-byte hex. A key that does not decode is
    skipped, never treated as a pass.
    """
    try:
        sig = base64.b64decode(sth["signature"], validate=True)
    except Exception:
        return None
    message = sth_signing_preimage(sth)
    runtime = get_ed25519_runtime()
    for key_hex in log_public_keys:
        key_hex = key_hex.strip().lower()
        try:
            if runtime.verify(key_hex, message, sig.hex()):
                return key_hex
        except CryptoError:
            continue
    return None


# =============================================================================
# Format detection & classification
# =============================================================================


def is_projected_witness_receipt(receipt: Any) -> bool:
    """Detector parity with glacis.io/verify ``isProjectedWitnessReceipt``.

    The projected receipt is hashes and labels with NO signature of its own —
    what stands behind it is the transparency-log entry beside it.
    """
    if not isinstance(receipt, dict):
        return False
    return (
        isinstance(receipt.get("receipt_id"), str)
        and isinstance(receipt.get("task"), str)
        and isinstance(receipt.get("outcome"), str)
        and not isinstance(receipt.get("signature"), str)
    )


def classify_envelope(
    envelope: Mapping[str, Any],
    log_public_keys: Optional[list[str]],
    checked_at_ms: Optional[int] = None,
) -> WitnessVerification:
    """Classify a ``{v, receipt, inclusion}`` envelope, fail-closed.

    ``LOG_INCLUSION_VERIFIED`` requires ALL of: a projected-witness-shaped
    receipt with a 64-hex ``receipt_id``; an inclusion proof whose recomputed
    root equals the tree head's root; and that tree head's signature
    verifying under a key from ``log_public_keys``. Anything less is
    ``LOGGED_UNVERIFIED`` with the first missing piece named. Semantics
    ported from the monorepo reference (mvp-product packages/glacis-py:
    inclusion requires leaf + inclusion + signed STH under a configured log
    key).

    ``LOG_INCLUSION_VERIFIED`` is also the ceiling: the leaf is the opaque
    ``receipt_id``, so the projection's ``task``/``outcome``/``commitments``
    are not covered by the proof, and altering them in a saved envelope does
    not (cannot) invalidate it. ``WITNESSED`` is never issued for this
    shape.
    """
    now_ms = (
        checked_at_ms if checked_at_ms is not None else time.time_ns() // 1_000_000
    )

    def unverified(reason: str, **kw: Any) -> WitnessVerification:
        return WitnessVerification(
            witness_status="LOGGED_UNVERIFIED",
            reason=reason,
            checked_at_ms=now_ms,
            **kw,
        )

    receipt = envelope.get("receipt") if isinstance(envelope, dict) else None
    if not is_projected_witness_receipt(receipt):
        return unverified(
            "the envelope's receipt is not in the projected witness shape "
            "(receipt_id/task/outcome strings, no signature field)"
        )
    assert isinstance(receipt, dict)

    receipt_id = receipt["receipt_id"].strip().lower()
    if not _HEX64_RE.match(receipt_id):
        return unverified(
            "receipt_id is not a 64-hex log-leaf identifier, so no proof can "
            "be tied to this receipt"
        )

    inclusion = envelope.get("inclusion")
    if not isinstance(inclusion, dict):
        return unverified("the envelope carries no inclusion record")
    if inclusion.get("status") != "included":
        return unverified(
            "the log reports inclusion status %r — the leaf has not anchored "
            "under a signed tree head yet" % inclusion.get("status")
        )

    sth = _read_log_sth(inclusion.get("sth"))
    if sth is None:
        return unverified(
            "the inclusion record has no checkable signed tree head "
            "(needs log_id, tree_size, 64-hex root_hash, timestamp_ms, signature)"
        )

    proof = _read_log_proof(inclusion.get("inclusion_proof"))
    if proof is None:
        return unverified(
            "the inclusion record has no checkable inclusion proof "
            "(needs leaf_index and an audit_path of 64-hex hashes)"
        )
    reported_index = inclusion.get("leaf_index")
    if (
        isinstance(reported_index, int)
        and not isinstance(reported_index, bool)
        and reported_index != proof["leaf_index"]
    ):
        return unverified(
            "the record reports leaf %s but its proof describes leaf %s — "
            "the proof does not describe this record"
            % (reported_index, proof["leaf_index"]),
            contradicted=True,
        )

    if not log_public_keys:
        return unverified(
            "no log public key is configured — set GLACIS_LOG_PUBLIC_KEY_HEX "
            "or pass log_public_keys. The mint succeeded and the record is "
            "attached, but nothing was verified here, so this artifact is "
            "not WITNESSED."
        )

    key_hex = verify_sth_signature(sth, log_public_keys)
    if key_hex is None:
        return unverified(
            "the signed tree head did not verify under any configured log key"
        )

    leaf = rfc6962_leaf_hash(bytes.fromhex(receipt_id))
    computed = recompute_log_root(
        leaf, proof["leaf_index"], proof["audit_path"], sth["tree_size"]
    )
    if computed is None or computed.hex() != sth["root_hash"]:
        return unverified(
            "the audit path from this receipt's own leaf does NOT lead to the "
            "root the tree head signed — a configured key vouched for that "
            "root, so the proof does not put this receipt in that tree",
            sth_signature_verified=True,
            log_public_key_hex=key_hex,
            contradicted=True,
        )

    return WitnessVerification(
        witness_status="LOG_INCLUSION_VERIFIED",
        inclusion_verified=True,
        sth_signature_verified=True,
        log_public_key_hex=key_hex,
        reason=None,
        scope=(
            "proven: receipt_id %s was included in log %r at tree size %s, "
            "under a log key this client was configured with. NOT proven: "
            "the task, outcome, or commitments shown beside it — the log "
            "leaf commits only to the receipt identifier, and the projected "
            "receipt carries no signature over those fields."
            % (receipt_id, sth["log_id"], sth["tree_size"])
        ),
        checked_at_ms=now_ms,
    )
