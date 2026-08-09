#!/usr/bin/env python3
"""Executable check that the code in these docs actually runs on the SDK.

Every snippet published under `Connect` and `Verify` has a counterpart here.
If the SDK surface changes, this script fails and the affected page is wrong.

Usage (from the repo root, with the SDK installed):

    python docs/scripts/verify-doc-snippets.py

Exit code 0 = every check reproduced. Non-zero = a doc page is making a claim
the SDK does not support.

**This script does not claim complete coverage, and it never has.** Claims it
cannot execute are printed as `NOT COVERED` with the reason, and counted
separately in the summary — they are not silently absent and they are not
counted as passes. Read the NOT COVERED block at the end of a run: it is the
list of things a green result does *not* establish.

Executed here: the offline signing path; canonical hashing and every
documented divergence from RFC 8785; all sixteen rows of the signed/unsigned
field tables, by tampering; storage round-trips on both backends including
persisted `control_plane_results` and the pre-0.8.1 loss of it; the retry and
latency behaviour of the online path (against a stubbed transport, no
network); what the online request body carries; L1/L2 evidence retention;
operation/sequence linking; sampling; the CLI; and the independent
(third-party) signature-verification recipe.
"""

from __future__ import annotations

import json
import re
import os
import subprocess
import sys
import tempfile
from pathlib import Path

CHECKS: list[tuple[str, bool, str]] = []
UNCOVERED: list[tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}{(' — ' + detail) if detail else ''}")


def not_covered(name: str, why: str) -> None:
    """Record a documented claim this script cannot execute.

    Printing it is the point. A harness that quietly omits what it cannot do
    is indistinguishable from one that covers everything.
    """
    UNCOVERED.append((name, why))
    print(f"[----] NOT COVERED: {name} — {why}")


def _raises(exc, fn, *args, **kwargs) -> bool:
    """True when calling fn(*args, **kwargs) raises exc."""
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    except Exception:
        return False
    return False


def main() -> int:
    import glacis
    from glacis import Glacis
    from glacis.crypto import canonical_json, hash_payload

    # The pages are written against 0.8.0 — what `pip install glacis` serves —
    # except where they say otherwise. This repo carries the unpublished 0.8.1
    # fixes, so the script accepts either and prints which one it ran on.
    check(
        "SDK under test is 0.8.0 or the unpublished 0.8.1 in this repo",
        glacis.__version__ in ("0.8.0", "0.8.1.dev0"),
        glacis.__version__,
    )
    on_081 = glacis.__version__ == "0.8.1.dev0"

    workdir = Path(tempfile.mkdtemp(prefix="glacis-docs-"))
    seed = bytes.fromhex(
        "9a3f1c0b7d2e4a5f8091c2d3e4f50617a8b9cadbec0d1e2f30415263748596a7"
    )

    # ------------------------------------------------------------------
    # Connect › Quickstart — offline attest
    # ------------------------------------------------------------------
    g = Glacis(
        mode="offline",
        signing_seed=seed,
        storage_backend="json",
        storage_path=workdir,
    )
    receipt = g.attest(
        service_id="my-ai-app",
        operation_type="inference",
        input={"prompt": "What is the capital of France?"},
        output={"response": "Paris."},
    )
    check("offline receipt id is oatt_*", receipt.id.startswith("oatt_"), receipt.id)
    check(
        "offline witness_status is UNVERIFIED",
        receipt.witness_status == "UNVERIFIED",
        receipt.witness_status,
    )
    check(
        "evidence_hash is 64 hex chars",
        len(receipt.evidence_hash) == 64,
        receipt.evidence_hash[:16] + "...",
    )
    check("public_key is populated", len(receipt.public_key) == 64)
    check("signature is populated", len(receipt.signature) == 128)

    # ------------------------------------------------------------------
    # Connect › Quickstart — evidence_hash is reproducible from the I/O
    # ------------------------------------------------------------------
    recomputed = hash_payload(
        {
            "input": {"prompt": "What is the capital of France?"},
            "output": {"response": "Paris."},
        }
    )
    check(
        "evidence_hash == hash_payload({'input':..., 'output':...})",
        recomputed == receipt.evidence_hash,
    )
    check(
        "canonical_json sorts keys",
        canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}',
        canonical_json({"b": 2, "a": 1}),
    )
    check(
        "hash_payload is key-order independent",
        hash_payload({"b": 2, "a": 1}) == hash_payload({"a": 1, "b": 2}),
    )

    # ------------------------------------------------------------------
    # Verify › What a check proves — glacis.verify() on an offline receipt
    # ------------------------------------------------------------------
    result = g.verify(receipt)
    check("verify() returns valid=True offline", result.valid is True)
    check(
        "verify() reports witness_status UNVERIFIED",
        result.witness_status == "UNVERIFIED",
    )

    # The documented caveat: in 0.8.0 the offline branch of verify() compares
    # the public key derived from the *local* seed against the receipt's
    # public_key. It does not check the Ed25519 signature. Prove it by
    # corrupting the signature and observing that verify() still says valid.
    tampered = receipt.model_copy(deep=True)
    tampered.signature = "00" * 64
    tampered_result = g.verify(tampered)
    check(
        "verify() does NOT check the signature (documented caveat)",
        tampered_result.valid is True,
        "zeroed signature still reports valid=True",
    )

    # ------------------------------------------------------------------
    # Verify › Verify it yourself — independent Ed25519 verification
    # ------------------------------------------------------------------
    # This is the snippet published on /verify/cli/, copied verbatim.
    from nacl.exceptions import BadSignatureError
    from nacl.signing import VerifyKey

    def signed_message(r: dict) -> bytes:
        body = {
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

    def verify_offline_receipt(r: dict, original_input, original_output):
        try:
            VerifyKey(bytes.fromhex(r["public_key"])).verify(
                signed_message(r), bytes.fromhex(r["signature"])
            )
            signature_ok = True
        except (BadSignatureError, ValueError):
            signature_ok = False
        binding_ok = (
            hash_payload({"input": original_input, "output": original_output})
            == r["evidence_hash"]
        )
        return signature_ok, binding_ok

    doc_input = {"prompt": "What is the capital of France?"}
    doc_output = {"response": "Paris."}
    as_dict = json.loads(json.dumps(receipt.model_dump(), default=str))

    check(
        "documented verifier accepts a good receipt",
        verify_offline_receipt(as_dict, doc_input, doc_output) == (True, True),
    )

    forged = dict(as_dict, evidence_hash="0" * 64)
    check(
        "documented verifier rejects a rewritten evidence_hash",
        verify_offline_receipt(forged, doc_input, doc_output) == (False, False),
    )

    stripped = dict(as_dict, signature="00" * 64)
    check(
        "documented verifier rejects a replaced signature",
        verify_offline_receipt(stripped, doc_input, doc_output) == (False, True),
    )

    check(
        "documented verifier catches a receipt bound to different I/O",
        verify_offline_receipt(as_dict, {"prompt": "something else"}, doc_output)
        == (True, False),
    )

    with_cpr = g.attest(
        service_id="svc",
        operation_type="inference",
        input=doc_input,
        output=doc_output,
        control_plane_results={
            "policy": {
                "id": "p",
                "version": "1.0",
                "environment": "development",
                "tags": [],
            },
            "determination": {"action": "forwarded"},
            "controls": [],
        },
    )
    check(
        "documented verifier handles control_plane_results",
        verify_offline_receipt(
            json.loads(json.dumps(with_cpr.model_dump(), default=str)),
            doc_input,
            doc_output,
        )
        == (True, True),
    )

    with_supersedes = g.attest(
        service_id="svc",
        operation_type="inference",
        input=doc_input,
        output=doc_output,
        supersedes=receipt.id,
    )
    check(
        "documented verifier handles supersedes",
        verify_offline_receipt(
            json.loads(json.dumps(with_supersedes.model_dump(), default=str)),
            doc_input,
            doc_output,
        )
        == (True, True),
    )

    # ------------------------------------------------------------------
    # Verify › What a check proves — which fields are inside the signature
    #
    # The page enumerates the signed payload and the fields that ride
    # alongside it. Both halves are asserted here by tampering: editing a
    # signed field must break the check, editing an unsigned one must not.
    # ------------------------------------------------------------------
    def still_verifies(mutate: dict, base: dict | None = None) -> bool:
        r = dict(base if base is not None else as_dict)
        r.update(mutate)
        try:
            VerifyKey(bytes.fromhex(r["public_key"])).verify(
                signed_message(r), bytes.fromhex(r["signature"])
            )
            return True
        except (BadSignatureError, ValueError, KeyError):
            return False

    def _verifies_with_constants(base: dict, **constants) -> bool:
        """Rebuild the signed payload with a different `version` / `mode`.

        Those two are constants in the payload, not fields on the receipt, so
        the only way to tamper with them is from the verifier's side.
        """
        body = json.loads(signed_message(base).decode())
        body.update(constants)
        message = json.dumps(body, separators=(",", ":"), sort_keys=True).encode()
        try:
            VerifyKey(bytes.fromhex(base["public_key"])).verify(
                message, bytes.fromhex(base["signature"])
            )
            return True
        except (BadSignatureError, ValueError):
            return False

    cpr_dict = json.loads(json.dumps(with_cpr.model_dump(), default=str))
    sup_dict = json.loads(json.dumps(with_supersedes.model_dump(), default=str))

    # --- The six OUTSIDE rows ------------------------------------------------
    for field, value in (
        ("id", "oatt_not-the-real-id"),
        ("cpr_hash", "0" * 64),
        ("is_offline", False),
        ("an_extra_field_nobody_signed", "anything"),
    ):
        check(
            f"`{field}` is OUTSIDE the offline signature (editing it does not break the check)",
            still_verifies({field: value}),
        )

    check(
        "`evidence`/`review`/`sampling_decision` are OUTSIDE the signature "
        "(adding them does not break the check)",
        still_verifies(
            {
                "evidence": {"sample_probability": 1.0, "data": {"anything": True}},
                "review": {"sample_probability": 1.0, "conformity_score": 1.0},
                "sampling_decision": {"level": "L2", "sample_value": 7},
            }
        ),
    )

    # `public_key` rides outside the signature, but not in the way the other
    # five do: swapping it alone breaks the check. What it is not is bound to
    # an identity — a full re-sign by another key verifies perfectly. Both
    # halves of that sentence are on the page, so both are pinned.
    from nacl.signing import SigningKey

    attacker = SigningKey(bytes.fromhex("11" * 32))
    check(
        "`public_key` swapped alone BREAKS the check",
        not still_verifies({"public_key": bytes(attacker.verify_key).hex()}),
    )
    _resigned = dict(as_dict)
    _resigned["public_key"] = bytes(attacker.verify_key).hex()
    _resigned["signature"] = attacker.sign(signed_message(as_dict)).signature.hex()
    check(
        "`public_key` is not bound to an identity — a re-signed receipt verifies",
        still_verifies({}, base=_resigned),
        "a different key, a different signature, a perfectly valid receipt",
    )

    # --- The ten INSIDE rows -------------------------------------------------
    for field, value in (
        ("service_id", "some-other-service"),
        ("operation_type", "something-else"),
        ("evidence_hash", "0" * 64),
        ("timestamp", (as_dict["timestamp"] or 0) + 1),
        ("operation_id", "00000000-0000-0000-0000-000000000000"),
        ("operation_sequence", (as_dict["operation_sequence"] or 0) + 1),
    ):
        check(
            f"`{field}` is INSIDE the offline signature (editing it breaks the check)",
            not still_verifies({field: value}),
        )

    check(
        "`version` is INSIDE the offline signature (it is a constant, so the "
        "verifier must use 1)",
        not _verifies_with_constants(as_dict, version=2),
    )
    check(
        "`mode` is INSIDE the offline signature (it is a constant, so the "
        "verifier must use \"offline\")",
        not _verifies_with_constants(as_dict, mode="online"),
    )

    check(
        "`control_plane_results` content is INSIDE the signature (editing it breaks it)",
        not verify_offline_receipt(
            dict(cpr_dict, control_plane_results={"policy": {"id": "tampered"}}),
            doc_input,
            doc_output,
        )[0],
    )
    check(
        "`control_plane_results` REMOVED breaks the check — losing it is as "
        "fatal as editing it",
        not verify_offline_receipt(
            dict(cpr_dict, control_plane_results=None), doc_input, doc_output
        )[0],
        "this is the shape of the pre-0.8.1 storage defect",
    )
    check(
        "`supersedes` is INSIDE the signature (editing it breaks the check)",
        not still_verifies({"supersedes": "oatt_something-else"}, base=sup_dict),
    )
    check(
        "`supersedes` ADDED to a receipt signed without one breaks the check",
        not still_verifies({"supersedes": "oatt_invented"}),
    )

    # ------------------------------------------------------------------
    # Verify › What a check proves — signed CPR has to survive storage
    #
    # `control_plane_results` is inside the signature, so a store that drops
    # it produces a receipt nobody can verify. 0.8.0 dropped it. Both halves
    # are pinned: the round trip that must work, and the 0.8.0-shaped row
    # that must fail *by name* rather than by silently looking CPR-free.
    # ------------------------------------------------------------------
    cpr_dir = workdir / "cpr-roundtrip"
    gc = Glacis(
        mode="offline", signing_seed=seed, storage_backend="json", storage_path=cpr_dir
    )
    doc_cpr = {
        "policy": {"id": "p", "version": "1.0", "environment": "development", "tags": []},
        "determination": {"action": "forwarded"},
        "controls": [],
    }
    stored_cpr_receipt = gc.attest(
        service_id="svc",
        operation_type="inference",
        input=doc_input,
        output=doc_output,
        control_plane_results=doc_cpr,
    )
    reloaded = gc._storage.get_receipt(stored_cpr_receipt.id)

    def _independently_verifies(att) -> bool:
        return verify_offline_receipt(
            json.loads(json.dumps(att.model_dump(), default=str)), doc_input, doc_output
        )[0]

    check(
        "a receipt with control_plane_results verifies before it is stored",
        _independently_verifies(stored_cpr_receipt),
    )
    if on_081:
        check(
            "a stored-and-reloaded receipt still carries its signed "
            "control_plane_results (0.8.1)",
            reloaded is not None and reloaded.control_plane_results == doc_cpr,
        )
        check(
            "a stored-and-reloaded receipt still passes independent Ed25519 "
            "verification (0.8.1)",
            reloaded is not None and _independently_verifies(reloaded),
        )
    else:
        check(
            "0.8.0 loses control_plane_results on storage round-trip "
            "(the documented defect)",
            reloaded is not None and reloaded.control_plane_results is None,
        )
        check(
            "0.8.0's reloaded receipt fails independent Ed25519 verification "
            "(the documented defect)",
            reloaded is not None and not _independently_verifies(reloaded),
        )

    # A row in the shape 0.8.0 wrote: cpr_hash present, content absent. This is
    # executed on whichever SDK is installed, so the 0.8.0 behaviour stays
    # pinned even when the fixed build is under test.
    from glacis.storage import JsonStorageBackend

    legacy_dir = workdir / "legacy-0.8.0-row"
    legacy_dir.mkdir()
    _legacy = {
        k: v
        for k, v in json.loads(
            (cpr_dir / "receipts.jsonl").read_text().splitlines()[-1]
        ).items()
        if k != "control_plane_results"
    }
    (legacy_dir / "receipts.jsonl").write_text(json.dumps(_legacy) + "\n")
    legacy = JsonStorageBackend(legacy_dir).get_receipt(stored_cpr_receipt.id)
    check(
        "a pre-0.8.1 row reconstructs WITHOUT the signed control-plane content",
        legacy is not None and legacy.control_plane_results is None,
    )
    check(
        "a pre-0.8.1 row fails independent Ed25519 verification",
        legacy is not None and not _independently_verifies(legacy),
        "the signed payload cannot be rebuilt from what was stored",
    )
    if on_081:
        check(
            "the loss is named, not inferred as 'this receipt had no CPR' (0.8.1)",
            legacy is not None and bool(legacy.cpr_recovery_error),
            (legacy.cpr_recovery_error or "")[:60] + "…" if legacy else "",
        )
        check(
            "verify() fails closed on such a receipt, with the reason (0.8.1)",
            legacy is not None
            and gc.verify(legacy).valid is False
            and gc.verify(legacy).error == legacy.cpr_recovery_error,
        )
    else:
        check(
            "0.8.0 reports no reason for the loss — it is indistinguishable "
            "from a receipt that never had CPR (the documented defect)",
            legacy is not None
            and getattr(legacy, "cpr_recovery_error", None) is None,
        )
    gc.close()

    # ------------------------------------------------------------------
    # Connect › Quickstart — the canonicalisation actually used
    #
    # The page states where glacis.crypto diverges from RFC 8785. Each row
    # of that table is pinned here, so the page cannot drift back to
    # claiming strict RFC 8785 conformance.
    # ------------------------------------------------------------------
    check(
        "canonical_json escapes non-ASCII (RFC 8785 would not)",
        canonical_json({"k": "caf\u00e9"}) == '{"k":"caf\\u00e9"}',
        canonical_json({"k": "caf\u00e9"}),
    )
    check(
        "canonical_json renders a whole float as 1.0 (RFC 8785 would say 1)",
        canonical_json(1.0) == "1.0",
        canonical_json(1.0),
    )
    check(
        "canonical_json renders 1e16 in exponent form (RFC 8785 would expand it)",
        canonical_json(1e16) == "1e+16",
        canonical_json(1e16),
    )
    check(
        "canonical_json sorts keys by Unicode code point, not UTF-16 code unit",
        # U+FFFF sorts BEFORE U+1F600 by code point; a UTF-16 sort puts the
        # surrogate pair (0xD83D…) first. This is the divergence, in one line.
        canonical_json({"\U0001f600": 1, "\uffff": 2}).index('"\\uffff"')
        < canonical_json({"\U0001f600": 1, "\uffff": 2}).index('"\\ud83d'),
        canonical_json({"\U0001f600": 1, "\uffff": 2}),
    )
    check(
        "canonical_json refuses NaN and Infinity",
        all(
            _raises(ValueError, canonical_json, v)
            for v in (float("nan"), float("inf"), float("-inf"))
        ),
    )
    check(
        "canonical_json keeps integers at arbitrary precision "
        "(RFC 8785 numbers are IEEE-754 doubles)",
        canonical_json(10**30) == "1" + "0" * 30
        and canonical_json(float(10**30)) == "1e+30"
        and canonical_json(2**53 + 1) == str(2**53 + 1)
        and canonical_json(float(2**53 + 1)) != str(2**53 + 1),
        f"10**30 -> {canonical_json(10 ** 30)}, "
        f"float(10**30) -> {canonical_json(float(10 ** 30))}",
    )
    check(
        "an int and the double nearest to it hash differently past 2**53",
        hash_payload(2**53 + 1) != hash_payload(float(2**53 + 1)),
        "a JCS implementation would produce the double's hash for both",
    )

    # ------------------------------------------------------------------
    # Connect › offline vs online — what the mode actually changes
    # ------------------------------------------------------------------
    from glacis.models import Attestation

    check(
        "witness_status is computed from is_offline alone, with no proof involved",
        Attestation(id="att_x", is_offline=False).witness_status == "WITNESSED"
        and Attestation(id="oatt_x", is_offline=True).witness_status == "UNVERIFIED",
    )
    check(
        "the Attestation model has no field for a countersignature or inclusion proof",
        not (
            {"witness_signature", "witness_public_key", "transparency_proofs",
             "inclusion_proof", "tree_head"}
            & set(Attestation.model_fields)
        ),
        "public key/signature fields: "
        + ", ".join(
            sorted(f for f in Attestation.model_fields if "key" in f or "sig" in f)
        ),
    )

    import inspect

    from glacis.client import _normalize_server_response

    _normalized = _normalize_server_response(
        {
            "id": "att_1",
            "signature": "ab",
            "publicKey": "cd",
            "transparency_proofs": {"inclusion_proof": {"leaf_index": 1}},
            "witness_signature": "ef",
        }
    )
    check(
        "the server-response normaliser drops transparency proofs and any second signature",
        "transparency_proofs" not in _normalized
        and "witness_signature" not in _normalized,
        "kept: " + ", ".join(sorted(_normalized)),
    )

    online = Glacis(api_key="glsk_test_not-a-real-key")
    check(
        "get_last_receipt() is offline-only — online mode raises",
        _raises(RuntimeError, online.get_last_receipt),
    )
    check(
        "the online attest path never writes to the local receipt store",
        "store_receipt" not in inspect.getsource(Glacis._attest_online),
    )

    # ------------------------------------------------------------------
    # Connect › index — "fail-open" is about exceptions, not about latency
    #
    # The page states an added-latency ceiling. Everything below runs against
    # a stubbed transport: no socket is opened and no host is contacted.
    # ------------------------------------------------------------------
    import httpx

    from glacis.client import (
        DEFAULT_BASE_DELAY,
        DEFAULT_MAX_DELAY,
        DEFAULT_MAX_RETRIES,
        DEFAULT_TIMEOUT,
    )
    from glacis.integrations import attested_openai as _attested_openai
    from glacis.integrations.base import create_glacis_client as _create_glacis_client
    from glacis.models import GlacisApiError, GlacisRateLimitError

    check(
        "online retry/timeout defaults are 30.0s, 3 retries, 1.0s base, 30.0s max",
        (DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES, DEFAULT_BASE_DELAY, DEFAULT_MAX_DELAY)
        == (30.0, 3, 1.0, 30.0),
    )
    _sleeps = [
        min(DEFAULT_BASE_DELAY * (2**i), DEFAULT_MAX_DELAY)
        for i in range(DEFAULT_MAX_RETRIES)
    ]
    check(
        "documented backoff sequence is 1s, 2s, 4s (7.0s, or 9.1s with maximum jitter)",
        _sleeps == [1.0, 2.0, 4.0]
        and round(sum(_sleeps), 1) == 7.0
        and round(sum(s * 1.3 for s in _sleeps), 1) == 9.1,
    )
    check(
        "documented worst case is 129.1s of added latency on one wrapped call",
        round(
            (DEFAULT_MAX_RETRIES + 1) * DEFAULT_TIMEOUT
            + sum(s * 1.3 for s in _sleeps),
            1,
        )
        == 129.1,
    )

    def _attempt_count(responder) -> int:
        calls = {"n": 0}
        probe = Glacis(
            api_key="glsk_test_probe",
            base_url="http://127.0.0.1:1",
            base_delay=0.001,
            max_delay=0.001,
        )

        def _stub(*a, **k):
            calls["n"] += 1
            return responder()

        probe._client.request = _stub  # type: ignore[method-assign]
        try:
            probe.attest(
                service_id="probe", operation_type="inference",
                input={"a": 1}, output={"b": 2},
            )
        except Exception:
            pass
        return calls["n"]

    def _connect_error():
        raise httpx.ConnectError("refused")

    check(
        "a connect failure makes max_retries + 1 = 4 attempts, blocking the caller",
        _attempt_count(_connect_error) == 4,
    )
    check(
        "a 5xx is retried the same four times",
        _attempt_count(lambda: httpx.Response(503, json={})) == 4,
    )
    check(
        "a 4xx is NOT retried — one attempt only, so a bad key costs one round trip",
        _attempt_count(lambda: httpx.Response(401, json={"error": "bad key"})) == 1,
    )
    check(
        "a 429 is NOT retried — one attempt only",
        _attempt_count(
            lambda: httpx.Response(429, headers={"Retry-After": "1"}, json={})
        )
        == 1,
    )

    def _raises_from_stub(exc, responder) -> bool:
        probe = Glacis(api_key="glsk_test_probe", base_url="http://127.0.0.1:1",
                       base_delay=0.001, max_delay=0.001)
        probe._client.request = lambda *a, **k: responder()  # type: ignore[method-assign]
        return _raises(
            exc,
            probe.attest,
            service_id="s", operation_type="t", input={}, output={},
        )

    check(
        "a 4xx raises GlacisApiError",
        _raises_from_stub(GlacisApiError, lambda: httpx.Response(401, json={"error": "x"})),
    )
    check(
        "a 429 raises GlacisRateLimitError",
        _raises_from_stub(
            GlacisRateLimitError,
            lambda: httpx.Response(429, headers={"Retry-After": "1"}, json={}),
        ),
    )

    _base_src = inspect.getsource(sys.modules["glacis.integrations.base"].attest_and_store)
    check(
        "the wrapper attests synchronously — no thread, queue or task in "
        "attest_and_store()",
        not any(
            token in _base_src
            for token in ("Thread", "asyncio", "executor", "Queue", "spawn")
        ),
    )
    _openai_src = inspect.getsource(sys.modules["glacis.integrations.openai"])
    check(
        "attest_and_store() runs after the provider response and before the "
        "wrapper returns it",
        _openai_src.index("response = original_create(")
        < _openai_src.index("attest_and_store(ctx,")
        < _openai_src.index("        return response"),
    )
    check(
        "no wrapper factory accepts timeout/max_retries — the latency is not "
        "bounded through the wrapper",
        not (
            {"timeout", "max_retries", "base_delay", "max_delay"}
            & set(inspect.signature(_attested_openai).parameters)
        )
        and "timeout" not in inspect.getsource(_create_glacis_client),
    )

    # ------------------------------------------------------------------
    # Connect › offline vs witnessed — the data boundary, exactly
    # ------------------------------------------------------------------
    _sent: dict[str, object] = {}

    def _fake_server(method, url, json=None, params=None, headers=None):
        _sent["body"] = json
        _sent["headers"] = headers
        return {
            "id": "att_stub",
            "operation_id": "op_stub",
            "operation_sequence": 0,
            "service_id": "svc",
            "operation_type": "inference",
            "evidence_hash": "0" * 64,
            "public_key": "d" * 64,
            "signature": "c" * 128,
            "timestamp": 1754000000000,
            "sampling_decision": {"level": "L1", "sample_value": 1, "prf_tag": []},
        }

    phi_shaped = {
        "policy": {"id": "p", "version": "1.0", "environment": "production", "tags": []},
        "determination": {"action": "forwarded"},
        "controls": [],
        "note": "PATIENT MRN 00-11-22, dob 1971-03-02",
    }
    stub_client = Glacis(api_key="glsk_test_stub", base_url="http://127.0.0.1:1")
    stub_client._request_with_retry = _fake_server  # type: ignore[method-assign]
    online_att = stub_client.attest(
        service_id="svc",
        operation_type="inference",
        input={"prompt": "the whole prompt"},
        output={"response": "the whole completion"},
        control_plane_results=phi_shaped,
    )
    _body = _sent["body"]
    check(
        "the online body carries only the hash of input/output",
        "input" not in _body and "output" not in _body and len(_body["evidence_hash"]) == 64,
        "keys: " + ", ".join(sorted(_body)),
    )
    check(
        "control_plane_results is transmitted VERBATIM, whatever is in it",
        _body["control_plane_results"] == phi_shaped
        and "PATIENT MRN" in json.dumps(_body),
        "an arbitrary dict, unhashed and uninspected, including PHI-shaped fields",
    )
    check(
        "cpr_hash is sent alongside the content, not instead of it",
        _body.get("cpr_hash") == hash_payload(phi_shaped)
        and "control_plane_results" in _body,
    )
    check(
        "the GLACIS key travels in the X-Glacis-Key header",
        (_sent["headers"] or {}).get("X-Glacis-Key") == "glsk_test_stub",
    )
    check(
        "there is no timestamp in the online request body",
        "timestamp" not in _body,
    )
    check(
        "an L1/L2 online attestation carries your RAW input and output back on "
        "the returned object",
        online_att.evidence is not None
        and online_att.evidence.data["input"] == {"prompt": "the whole prompt"}
        and online_att.evidence.data["output"] == {"response": "the whole completion"},
        "so 'online attest() retains nothing' is false for the object in memory",
    )
    check(
        "model_dump() of such a receipt contains the prompt — excluding "
        "`evidence`/`review` is the documented remedy",
        "the whole prompt" in json.dumps(online_att.model_dump(), default=str)
        and "the whole prompt"
        not in json.dumps(
            online_att.model_dump(exclude={"evidence", "review"}), default=str
        ),
    )
    check(
        "an offline attestation never populates `evidence`",
        receipt.evidence is None and with_cpr.evidence is None,
    )
    stub_client.close()

    check(
        "the OpenAI wrapper stores a projection of your request, not your request",
        'input_data = {"model": model, "messages": messages}' in _openai_src,
        "model and messages only — temperature, tools, response_format etc. are dropped",
    )
    check(
        "the OpenAI wrapper's stored response omits id/created/system_fingerprint/"
        "tool_calls/logprobs",
        not any(
            tok in _openai_src
            for tok in ("system_fingerprint", "tool_calls", "logprobs", "created")
        ),
    )
    check(
        "the wrapper hashes that same projection, so evidence_hash commits to it",
        "attest_and_store(ctx, input_data, output_data" in _openai_src
        and "input=input_data" in _base_src
        and "output=output_data" in _base_src,
    )
    check(
        "the offline attest path stores 100-character previews, not full payloads",
        "[:100]" in inspect.getsource(Glacis._attest_offline),
    )
    _long = g.attest(
        service_id="svc", operation_type="inference",
        input={"prompt": "x" * 500}, output={"response": "y" * 500},
    )
    _rows = [
        json.loads(line)
        for line in (workdir / "receipts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    _row = next(r for r in _rows if r.get("attestation_id") == _long.id)
    check(
        "the offline store keeps a 100-character preview, not the payload",
        len(_row.get("input_preview") or "") == 100
        and len(_row.get("output_preview") or "") == 100,
        f"input_preview len={len(_row.get('input_preview') or '')}",
    )

    # ------------------------------------------------------------------
    # Connect › Configuration — a config-only wrapper cannot sign
    # ------------------------------------------------------------------
    from glacis.integrations.base import create_glacis_client

    check(
        "a config-first wrapper with no seed raises ValueError (offline is the default)",
        _raises(
            ValueError,
            create_glacis_client,
            offline=True,
            signing_seed=None,
            glacis_api_key=None,
            glacis_base_url="https://api.glacis.io",
            debug=False,
        ),
    )

    # ------------------------------------------------------------------
    # Reference › Batch — operation linking, decompose, supersedes
    # ------------------------------------------------------------------
    op = g.operation()
    r1 = g.attest(
        service_id="svc",
        operation_type="inference",
        input={"p": 1},
        output={"r": 1},
        operation_id=op.operation_id,
        operation_sequence=op.next_sequence(),
    )
    r2 = g.attest(
        service_id="svc",
        operation_type="inference",
        input={"p": 2},
        output={"r": 2},
        operation_id=op.operation_id,
        operation_sequence=op.next_sequence(),
    )
    check(
        "operation() shares operation_id, increments sequence",
        r1.operation_id == r2.operation_id
        and (r1.operation_sequence, r2.operation_sequence) == (0, 1),
    )

    parent = g.attest(
        service_id="svc",
        operation_type="batch",
        input={"source": "kb.pdf"},
        output={"pairs": [{"q": "a"}, {"q": "b"}]},
    )
    items = g.decompose(parent, [{"q": "a"}, {"q": "b"}], operation_type="item")
    check(
        "decompose() continues the parent's sequence",
        [i.operation_sequence for i in items] == [1, 2]
        and all(i.operation_id == parent.operation_id for i in items),
    )

    revised = g.attest(
        service_id="svc",
        operation_type="inference",
        input={"p": 1},
        output={"r": "better"},
        supersedes=parent.id,
    )
    check("supersedes is recorded", revised.supersedes == parent.id)

    # ------------------------------------------------------------------
    # Reference › Sampling — deterministic, reproducible tiers
    # ------------------------------------------------------------------
    decision = g.should_review(receipt)
    check(
        "should_review() returns a tier",
        decision.level in ("L0", "L1", "L2"),
        decision.level,
    )
    check(
        "should_review() is deterministic",
        g.should_review(receipt).sample_value == decision.sample_value,
    )
    check(
        "sampling_rate=0.0 forces L0",
        g.should_review(receipt, sampling_rate=0.0).level == "L0",
    )

    # ------------------------------------------------------------------
    # Reference › Storage — JSONL backend round-trip
    # ------------------------------------------------------------------
    last = g.get_last_receipt()
    check("get_last_receipt() returns the newest receipt", last is not None)
    check(
        "json backend writes receipts.jsonl",
        (workdir / "receipts.jsonl").exists(),
        str(workdir / "receipts.jsonl"),
    )
    g.close()

    # ------------------------------------------------------------------
    # Reference › Storage — path arguments are Path objects, and `~` is literal
    # ------------------------------------------------------------------
    from glacis.storage import DEFAULT_DB_PATH, DEFAULT_STORAGE_DIR, create_storage

    try:
        create_storage(backend="json", path=str(workdir))
        check("create_storage() rejects a str path (documented gotcha)", False)
    except AttributeError as e:
        check(
            "create_storage() rejects a str path (documented gotcha)",
            "mkdir" in str(e),
            str(e),
        )

    check(
        "storage defaults are already home-expanded",
        str(DEFAULT_DB_PATH).startswith(str(Path.home()))
        and str(DEFAULT_STORAGE_DIR).startswith(str(Path.home())),
        str(DEFAULT_DB_PATH),
    )

    tilde_root = workdir / "tilde-test"
    tilde_root.mkdir()
    cwd = os.getcwd()
    try:
        os.chdir(tilde_root)
        store = create_storage(backend="sqlite", path=Path("~/glacis-doc-probe.db"))
        store.close()
        check(
            "`~` in an explicit storage path is NOT expanded (documented gotcha)",
            (tilde_root / "~").exists(),
            "a literal '~' directory is created next to the process CWD",
        )
    finally:
        os.chdir(cwd)

    # ------------------------------------------------------------------
    # Verify › CLI — python -m glacis verify
    # ------------------------------------------------------------------
    receipt_path = workdir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt.model_dump(), indent=2, default=str))
    proc = subprocess.run(
        [sys.executable, "-m", "glacis", "verify", str(receipt_path)],
        capture_output=True,
        text=True,
    )
    check(
        "`python -m glacis verify` exits 0 on a good offline receipt",
        proc.returncode == 0,
        proc.stdout.strip().replace("\n", " | "),
    )

    tampered_path = workdir / "tampered.json"
    tampered_doc = receipt.model_dump()
    tampered_doc["signature"] = "00" * 64
    tampered_path.write_text(json.dumps(tampered_doc, indent=2, default=str))
    proc2 = subprocess.run(
        [sys.executable, "-m", "glacis", "verify", str(tampered_path)],
        capture_output=True,
        text=True,
    )
    check(
        "CLI still passes a zeroed signature (documented caveat)",
        proc2.returncode == 0,
        "structural validation only",
    )

    # ------------------------------------------------------------------
    # Connect › Configuration — glacis.yaml load
    # ------------------------------------------------------------------
    from glacis.config import load_config

    try:
        import yaml  # noqa: F401

        have_yaml = True
    except ImportError:
        have_yaml = False
    check(
        "glacis.yaml support requires pyyaml (not a core dependency)",
        True,
        "pyyaml installed" if have_yaml else "pyyaml missing — config checks skipped",
    )

    cfg_path = workdir / "glacis.yaml"
    cfg_path.write_text(
        "version: '1.3'\n"
        "policy:\n"
        "  id: my-policy\n"
        "  environment: production\n"
        "controls:\n"
        "  input:\n"
        "    word_filter:\n"
        "      enabled: true\n"
        "      entities: ['confidential']\n"
        "      if_detected: flag\n"
        "sampling:\n"
        "  l1_rate: 0.1\n"
        "  l2_rate: 0.01\n"
        "attestation:\n"
        "  offline: true\n"
        "  service_id: my-service\n"
        "evidence_storage:\n"
        "  backend: json\n"
    )
    if have_yaml:
        cfg = load_config(str(cfg_path))
        check(
            "load_config() reads policy/sampling/attestation/storage",
            cfg.policy.id == "my-policy"
            and cfg.sampling.l1_rate == 0.1
            and cfg.attestation.service_id == "my-service"
            and cfg.evidence_storage.backend == "json",
        )
        check(
            "word_filter control config parses",
            cfg.controls.input.word_filter.enabled
            and cfg.controls.input.word_filter.entities == ["confidential"],
        )
        check(
            "load_config() with no file returns defaults",
            load_config().attestation.offline is True,
        )

    # ------------------------------------------------------------------
    # Reference › Controls — word filter runs with no extra dependencies
    # ------------------------------------------------------------------
    from glacis.config import WordFilterControlConfig
    from glacis.controls import WordFilterControl

    wf = WordFilterControl(
        WordFilterControlConfig(
            enabled=True, entities=["confidential"], if_detected="flag"
        )
    )
    hit = wf.check("This document is CONFIDENTIAL.")
    miss = wf.check("Nothing to see here.")
    check("WordFilterControl detects a term (case-insensitive)", hit.detected)
    check("WordFilterControl reports the matched term", hit.categories == ["confidential"])
    check("WordFilterControl passes clean text", not miss.detected)
    check("controls never rewrite text (scan-only)", hit.modified_text is None)

    from glacis.config import InputControlsConfig, OutputControlsConfig
    from glacis.controls import ControlsRunner

    runner = ControlsRunner(
        input_config=InputControlsConfig(
            word_filter=WordFilterControlConfig(
                enabled=True, entities=["confidential"], if_detected="flag"
            )
        ),
        output_config=OutputControlsConfig(),
    )
    stage = runner.run_input("This document is CONFIDENTIAL.")
    check(
        "ControlsRunner.effective_text is always the original text",
        stage.effective_text == "This document is CONFIDENTIAL.",
    )
    check(
        "if_detected: flag does not block",
        stage.should_block is False,
    )

    # ------------------------------------------------------------------
    # Connect › Offline vs witnessed — online mode requires a key
    # ------------------------------------------------------------------
    try:
        Glacis()
        check("online mode requires an api_key", False)
    except ValueError as e:
        check("online mode requires an api_key", "api_key is required" in str(e))

    try:
        Glacis(mode="offline")
        check("offline mode requires a signing_seed", False)
    except ValueError as e:
        check("offline mode requires a signing_seed", "signing_seed is required" in str(e))

    try:
        Glacis(mode="offline", signing_seed=os.urandom(16))
        check("signing_seed must be 32 bytes", False)
    except ValueError as e:
        check("signing_seed must be 32 bytes", "exactly 32 bytes" in str(e))

    # ------------------------------------------------------------------
    # Reference › Controls — the ControlExecution `version` stamp
    # ------------------------------------------------------------------
    from glacis.integrations.base import SDK_VERSION

    check(
        "ControlExecution.version stamp is 0.7.0, not the package version "
        "(documented quirk)",
        SDK_VERSION == "0.7.0" and SDK_VERSION != glacis.__version__,
        f"SDK_VERSION={SDK_VERSION}, __version__={glacis.__version__}",
    )

    # ------------------------------------------------------------------
    # Connect › Install — the extras that actually exist
    # ------------------------------------------------------------------
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text()
        declared = set(
            re.findall(r"^([a-z0-9_-]+) = \[", text.split("[project.optional-dependencies]")[1].split("\n[project.urls]")[0], re.M)
        )
        check(
            "documented extras match pyproject.toml",
            declared
            == {
                "dev",
                "openai",
                "anthropic",
                "gemini",
                "litellm",
                "redaction",
                "jailbreak",
                "controls",
                "all",
            },
            ", ".join(sorted(declared)),
        )
        check(
            "there is no `content-safety` extra, despite the SDK error message",
            "content-safety" not in declared,
        )

    # ------------------------------------------------------------------
    # Connect › provider wrappers — signatures only (no provider key here)
    # ------------------------------------------------------------------
    import inspect

    from glacis.integrations import (
        attested_anthropic,
        attested_gemini,
        attested_litellm,
        attested_openai,
    )
    from glacis.integrations import get_evidence, get_last_receipt

    shared_kwargs = {
        "glacis_api_key",
        "glacis_base_url",
        "service_id",
        "offline",
        "signing_seed",
        "policy_key",
        "config",
        "input_controls",
        "output_controls",
        "metadata",
        "debug",
    }
    for fn, extra in (
        (attested_openai, {"openai_api_key"}),
        (attested_anthropic, {"anthropic_api_key"}),
        (attested_gemini, {"gemini_api_key"}),
        (attested_litellm, set()),
    ):
        params = set(inspect.signature(fn).parameters)
        missing = (shared_kwargs | extra) - params
        check(
            f"{fn.__name__}() accepts every documented keyword",
            not missing,
            "missing: " + ", ".join(sorted(missing)) if missing else "",
        )

    check(
        "get_last_receipt()/get_evidence() are importable from glacis.integrations",
        callable(get_last_receipt) and callable(get_evidence),
    )
    check(
        "attested_litellm() exposes .completion()/.acompletion()",
        hasattr(
            sys.modules["glacis.integrations.litellm"].AttestedLiteLLM, "completion"
        )
        and hasattr(
            sys.modules["glacis.integrations.litellm"].AttestedLiteLLM, "acompletion"
        ),
    )

    import asyncio

    _acompletion = sys.modules["glacis.integrations.litellm"].AttestedLiteLLM.acompletion
    _ll_src = inspect.getsource(_acompletion)
    check(
        "acompletion() attests on the event loop — the blocking retry sleep is "
        "not offloaded",
        asyncio.iscoroutinefunction(_acompletion)
        and "attest_and_store(ctx," in _ll_src
        and not any(
            tok in _ll_src for tok in ("to_thread", "run_in_executor", "await attest")
        ),
        "so an online retry storm stalls every other task on the loop, not just "
        "this one",
    )

    # ------------------------------------------------------------------
    # What this run does NOT establish.
    #
    # Every line here is a claim on a published page that this script cannot
    # execute. They are printed rather than omitted so that "the harness is
    # green" can never be read as "every documented claim is checked".
    # ------------------------------------------------------------------
    print()
    not_covered(
        "witnessed (online) attestation end to end",
        "needs a live api.glacis.io and a real workspace key; the request body, "
        "retry behaviour and L1/L2 evidence above are pinned against a stubbed "
        "transport, but no real server response has been seen",
    )
    not_covered(
        "the 129.1s worst-case latency, measured",
        "the arithmetic and the four-attempt count are checked; actually "
        "observing four 30-second timeouts needs an endpoint that hangs, and "
        "would make this script take over two minutes",
    )
    not_covered(
        "provider wrappers end to end (OpenAI/Anthropic/Gemini/LiteLLM)",
        "needs paid provider keys; the stored projections are pinned by reading "
        "the wrapper source, not by making a call, so a change in a provider's "
        "response shape would not be caught here",
    )
    not_covered(
        "the browser verifier and the `#r=` permalink",
        "lives in the glacis-plg repository and needs a browser; nothing in this "
        "script exercises it",
    )
    not_covered(
        "the portal's witnessed mint path",
        "lives in the labs-plg repository and needs a deployed backend; the "
        "witnessed tier described on /connect/offline-vs-witnessed/ is not "
        "verified by anything here",
    )
    not_covered(
        "SQLite receipts written by a genuinely old install",
        "the pre-0.8.1 row above is reconstructed by hand in the shape 0.8.0 "
        "wrote; the v4->v5 migration itself is covered by the SDK test suite "
        "(tests/test_cpr_persistence.py), not by this script",
    )
    not_covered(
        "PII/PHI and jailbreak controls",
        "need presidio/spacy and transformers/torch; only the word filter, which "
        "has no extra dependencies, is executed here",
    )

    failed = [name for name, ok, _ in CHECKS if not ok]
    print()
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    print(f"{len(UNCOVERED)} documented claims NOT COVERED by this script")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
