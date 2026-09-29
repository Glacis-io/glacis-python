"""Reference RFC 6962 log for tests — an independent implementation.

Ports the recursive MTH/PATH definitions straight from RFC 6962 §2.1/§2.1.1,
matching the two production references byte for byte:

* translog-svc (lite demo backend, ``src/main.rs``: ``mth``/``subproof`` over
  ``leaf_hash = sha256(0x00 || leaf)``), which produces the gateway's real
  inclusion proofs, and
* the monorepo parity oracle (mvp-product ``packages/glacis-py``
  ``src/glacis_py/merkle.py``).

The SDK's verifier (``glacis.witness.recompute_log_root``) is the *iterative*
RFC 9162 §2.1.3.2 algorithm; proving it against this independent recursive
producer is the point — the same structure translog-svc's own tests use.
"""

from __future__ import annotations

import base64
import hashlib
import json

import nacl.signing


def leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _split_point(n: int) -> int:
    """Largest power of two strictly less than n (RFC 6962: k < n <= 2k)."""
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def mth(leaves: list[bytes]) -> bytes:
    """RFC 6962 MTH over precomputed leaf hashes."""
    if len(leaves) == 0:
        return hashlib.sha256(b"").digest()
    if len(leaves) == 1:
        return leaves[0]
    k = _split_point(len(leaves))
    return node_hash(mth(leaves[:k]), mth(leaves[k:]))


def path(m: int, leaves: list[bytes]) -> list[bytes]:
    """RFC 6962 §2.1.1 PATH(m, D[n]) over precomputed leaf hashes."""
    n = len(leaves)
    if n == 1:
        return []
    k = _split_point(n)
    if m < k:
        return path(m, leaves[:k]) + [mth(leaves[k:])]
    return path(m - k, leaves[k:]) + [mth(leaves[:k])]


class ReferenceLog:
    """A tiny signed transparency log producing gateway-shaped records."""

    def __init__(self, log_id: str = "glacis-log/test", seed: bytes = b"\x01" * 32):
        self.log_id = log_id
        self._key = nacl.signing.SigningKey(seed)
        self.public_key_hex = bytes(self._key.verify_key).hex()
        self._receipt_hashes: list[str] = []

    def append(self, receipt_hash_hex: str) -> int:
        self._receipt_hashes.append(receipt_hash_hex)
        return len(self._receipt_hashes) - 1

    def _leaf_hashes(self) -> list[bytes]:
        return [leaf_hash(bytes.fromhex(h)) for h in self._receipt_hashes]

    def sth(self, timestamp_ms: int = 1_800_000_000_000) -> dict:
        leaves = self._leaf_hashes()
        head = {
            "log_id": self.log_id,
            "tree_size": len(leaves),
            "root_hash": mth(leaves).hex(),
            "timestamp_ms": timestamp_ms,
        }
        # Declaration-order compact JSON — the exact preimage translog signs.
        preimage = json.dumps(head, separators=(",", ":"), ensure_ascii=False)
        sig = self._key.sign(preimage.encode("utf-8")).signature
        head["signature"] = base64.b64encode(sig).decode("ascii")
        return head

    def inclusion(self, leaf_index: int) -> dict:
        leaves = self._leaf_hashes()
        return {
            "status": "included",
            "leaf_index": leaf_index,
            "inclusion_proof": {
                "leaf_index": leaf_index,
                "audit_path": [h.hex() for h in path(leaf_index, leaves)],
            },
            "sth": self.sth(),
        }
