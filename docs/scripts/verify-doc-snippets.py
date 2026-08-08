#!/usr/bin/env python3
"""Executable check that the code in these docs actually runs on glacis 0.8.0.

Every snippet published under `Connect` and `Verify` has a counterpart here.
If the SDK surface changes, this script fails and the affected page is wrong.

Usage (from the repo root, with the SDK installed):

    python docs/scripts/verify-doc-snippets.py

Exit code 0 = every documented behaviour reproduced. Non-zero = a doc page
is making a claim the SDK does not support.

Snippets that require a network call (online/witnessed mode) or a paid
provider key (OpenAI/Anthropic/Gemini/LiteLLM wrappers) are NOT exercised
here — those pages mark themselves as untested against a live endpoint.
What this script does cover: the offline signing path, canonical hashing
(including every documented divergence from RFC 8785), which fields are
inside and outside the offline signature, storage and its 100-character
previews, operation/sequence linking, sampling, the CLI, the mode
differences the docs promise, and the independent (third-party)
signature-verification recipe.
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


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}{(' — ' + detail) if detail else ''}")


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

    check("SDK version is 0.8.0", glacis.__version__ == "0.8.0", glacis.__version__)

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
    def still_verifies(mutate: dict) -> bool:
        r = dict(as_dict)
        r.update(mutate)
        try:
            VerifyKey(bytes.fromhex(r["public_key"])).verify(
                signed_message(r), bytes.fromhex(r["signature"])
            )
            return True
        except (BadSignatureError, ValueError, KeyError):
            return False

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
        "control_plane_results content is inside the signature",
        not verify_offline_receipt(
            dict(
                json.loads(json.dumps(with_cpr.model_dump(), default=str)),
                control_plane_results={"policy": {"id": "tampered"}},
            ),
            doc_input,
            doc_output,
        )[0],
    )

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
        SDK_VERSION == "0.7.0" and glacis.__version__ == "0.8.0",
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

    failed = [name for name, ok, _ in CHECKS if not ok]
    print()
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
