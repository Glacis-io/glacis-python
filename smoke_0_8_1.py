"""Smoke test for glacis 0.8.1 — proves the headline fixes are live.

Run against an installed build:  python smoke_0_8_1.py
It exits non-zero and says PROVED/FAILED for each of the three things 0.8.1
fixed, so a passing run is evidence the release does what it claims.
"""

import os
import sys
import tempfile
from pathlib import Path

import glacis
from glacis import Glacis

ok = True


def result(name: str, passed: bool, detail: str = "") -> None:
    global ok
    ok = ok and passed
    print(f"  [{'PROVED' if passed else 'FAILED'}] {name}" + (f" — {detail}" if detail else ""))


print(f"glacis {glacis.__version__} — offline signature + CPR-persistence smoke\n")

seed = os.urandom(32)
db = Path(tempfile.mkdtemp(prefix="glacis-smoke-")) / "receipts.sqlite"
cpr = {"controls": [{"name": "phi_redaction", "outcome": "pass"}], "determination": {"action": "allow"}}

# 1. A fresh offline receipt with control-plane results verifies.
g = Glacis(mode="offline", signing_seed=seed, storage_backend="sqlite", storage_path=db)
receipt = g.attest(
    service_id="smoke",
    operation_type="inference",
    input={"prompt": "does this patient have diabetes?"},
    output={"response": "cannot disclose"},
    control_plane_results=cpr,
)
fresh = g.verify(receipt)
result("a fresh receipt verifies", fresh.valid, f"error={fresh.error!r}")
g.close()

# 2. THE headline fix: reload from storage in a NEW client and re-verify.
#    Under 0.8.0 the signed control_plane_results were dropped on write, so the
#    reloaded receipt failed independent verification. It must survive now.
g2 = Glacis(mode="offline", signing_seed=seed, storage_backend="sqlite", storage_path=db)
reloaded = g2._storage.get_receipt(receipt.id)
result("the receipt survives a storage round-trip", reloaded is not None)
if reloaded is not None:
    result(
        "its control_plane_results survived (0.8.0 dropped them)",
        reloaded.control_plane_results is not None,
    )
    after = g2.verify(reloaded)
    result("the reloaded receipt still verifies", after.valid, f"error={after.error!r}")
g2.close()

# 3. Real Ed25519, not string length: a tampered signed field must be rejected.
#    Under 0.8.0 verify() never looked at the signature, so this passed.
tampered = receipt.model_copy(deep=True, update={"service_id": "smoke-TAMPERED"})
verdict = Glacis(mode="offline", signing_seed=seed).verify(tampered)
result(
    "a tampered signed field is rejected",
    verdict.valid is False,
    f"error={verdict.error!r}",
)

print()
if ok:
    print("ALL PROVED — 0.8.1 verifies signatures for real and persists CPR.")
    sys.exit(0)
print("SOMETHING FAILED — do not treat this build as the fix.")
sys.exit(1)
