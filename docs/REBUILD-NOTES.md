# docs.glacis.io rebuild — 2026-08-08

Task C1 of the PLG zero-to-receipt build plan
(`glacis-know/2026-08-08_plg-zero-to-receipt-build-plan.md`, §4 WS-C).

**Verdict acted on:** the docs were stale, so they were rebuilt rather than
patched. The Astro/Starlight scaffold and the deploy pipeline are untouched —
replatforming is a post-Monday question.

**Shape of the change:** the information architecture now follows the activation
ladder from plan §1 (Start → Connect → Verify → OVERT → Reference) instead of
"here is a Python package". Everything that survives has been checked against
the `glacis` 0.8.0 source in this worktree, and most of it is checked by a
script that runs the code.

---

## How snippets were verified

`docs/scripts/verify-doc-snippets.py` executes the documented behaviour against
the SDK in this repo. **141 checks, all passing** as of this commit
(53 from the original rebuild, 24 added in round 2, 41 in round 3, 21 in
round 4, 3 in round 5 less one removed), plus **8 `NOT COVERED` lines** — claims
the script cannot execute, printed with a reason and counted separately so that
a green run is never read as complete coverage.

```
pip install -e .          # from the repo root
pip install pyyaml pynacl
python docs/scripts/verify-doc-snippets.py
```

It covers: offline attestation, canonical-JSON hashing and every documented
divergence from RFC 8785, `verify()` — the real Ed25519 check it performs from
0.8.1, *and* the structural-only behaviour 0.8.0 had, each asserted on the
version that claims it — the published independent verification routine across
every tamper case plus the `control_plane_results` and `supersedes` variants,
all eighteen rows of the signed/unsigned field tables — including the four
unsigned ones that decide the outcome, two of which (`is_offline`, `id`) are
checked through public `Glacis.verify()` dispatch because that is where they
act — the persisted-CPR round
trip and the pre-0.8.1 loss of it, the request and response projections of all
four provider wrappers, the online path's retry and latency behaviour and
request body (against a stubbed transport), L1/L2 evidence retention, operation
linking, `decompose()`, `supersedes`, `should_review()` determinism, both
storage backends, the two storage path gotchas, the CLI on good and tampered
receipts, `glacis.yaml` loading, the word filter and `ControlsRunner`,
constructor validation, the declared extras in `pyproject.toml`, and the
keyword signature of all four provider wrappers.

It cannot cover: witnessed (online) mode against a live endpoint, or an
end-to-end provider call. Those need a live endpoint and paid keys, and the
script prints them as NOT COVERED rather than leaving them out. **Every page
carrying such a snippet says so
explicitly** rather than implying it was tested.

---

## Page-by-page

### Deleted

| Page | Why |
| --- | --- |
| `sdk/python/quickstart.mdx` | Carried the **request-access dead end** ("Request&#32;access on glacis.io") this task exists to kill, and a broken `<Steps>` list (numbered 1, 2, 4, 5). Replaced by `connect/quickstart.mdx`, written offline-first so a reader gets a receipt before being asked for anything. |
| `sdk/python/installation.mdx` | Told readers to "Visit glacis.io and sign up" — signup is at **app.glacis.dev**, not glacis.io. Also printed `glacis.__version__  # 0.5.0` against a 0.8.0 package, and omitted the `litellm` extra entirely. Replaced by `connect/install.mdx`. |
| `sdk/python/offline.mdx` | Framed offline mode as a feature tier ("upgrade to online when you need third-party verifiability") rather than as a different claim. Also asserted that `glacis.verify()` performs "full Ed25519 signature verification" — it does not (see Corrections). Replaced by `connect/offline-vs-witnessed.mdx`, built on the diary-vs-evidence framing. |
| `sdk/python/openai.mdx` | Rewritten as `connect/openai.mdx`. The original never mentioned that attestation is fail-open, that only `chat.completions.create` is wrapped, or that a failed attestation leaves `get_last_receipt()` stale. |
| `sdk/python/anthropic.mdx` | Rewritten as `connect/anthropic.mdx`, same reasons. |
| `sdk/python/gemini.mdx` | Rewritten as `connect/gemini.mdx`, same reasons. |
| `sdk/python/litellm.mdx` | Rewritten as `connect/litellm.mdx` with the config-first path the plan asked for, and an honest statement of what "config-only" does and does not mean here (see Residuals). |
| `sdk/python/pipelines.mdx` | **Orphan** — never listed in the sidebar, so it was reachable only by direct URL. Content overlapped `batch.mdx` heavily and contained a snippet that cannot run: `Glacis(offline=True, signing_seed=seed)` — the parameter is `mode="offline"`. Merged into `reference/operations.mdx` with the snippet fixed. |

### Rewritten

| Page | Now | What changed |
| --- | --- | --- |
| `index.mdx` | `index.mdx` | Splash rebuilt around the two entry points (browser / SDK). Primary CTA is now **Start — no code, no install**. Added the three-state badge legend and an explicit "what a receipt is not". Dropped the old "Provider Integrations" card that listed OpenAI/Anthropic/Gemini and silently omitted LiteLLM. |
| `sdk/python/cli.mdx` | `verify/cli.mdx` | The old page said the CLI does structural validation only, then told readers to "use `Glacis.verify()` for full cryptographic verification" — which is also not a signature check. Rewritten with the real behaviour and a tested ten-line PyNaCl routine that actually verifies. |

### New

| Page | Why it exists |
| --- | --- |
| `start/index.mdx` | The ladder, phone-legible, with the three-state legend front and centre. |
| `start/create-a-workspace.mdx` | Signup → verification email → sign in → system profile. Includes the free-tier position: receipts are never metered; rate limits are abuse control, not billing. |
| `start/sample-workspace.mdx` | Reading a receipt before producing one. Every synthetic row is `SAMPLE`, everywhere, including exports. |
| `start/mint-a-receipt.mdx` | The L1 rung. Documents the flag-gated honest state: when witnessing is not live the portal says so or returns a `SELF-SIGNED` receipt — it never shows a green check nothing backs. |
| `connect/index.mdx` | Three paths (wrap a provider / config-first / attest directly) and the one decision that matters: offline or witnessed. |
| `connect/install.mdx` | Extras and what each pulls in, plus two things the old page missed: `glacis[controls]` is a multi-gigabyte install, and `glacis.yaml` needs `pyyaml`, which is not a core dependency. |
| `connect/quickstart.mdx` | Offline-first. Every snippet on it is executed by the verification script. |
| `connect/offline-vs-witnessed.mdx` | The diary-vs-evidence framing, what is actually transmitted, and the constructor's exact failure modes. |
| `verify/index.mdx` | In-browser verification, the `#r=` fragment permalink and why a fragment rather than a query string, and the per-check breakdown. |
| `verify/what-a-check-proves.mdx` | The core honesty page: what a green check establishes, the six things it does not, and how to read someone else's receipt. |
| `overt.mdx` | OVERT raised to a top-level section, as the announcement cites it. Facts taken from overt.is directly (v1.1, published 11 June 2026, `overt.is/1.1`, royalty-free covenant). |

### Kept, re-homed, corrected

All reference pages survived — they were the most accurate part of the old site.
Each was moved into `reference/` (or `connect/`), had its internal links
rewritten, and had specific false claims fixed.

| Was | Now | Corrections |
| --- | --- | --- |
| `sdk/python/api.mdx` | `reference/api.mdx` | "For offline attestations, verifies the Ed25519 signature locally" → replaced with what it does. `OfflineVerifyResult.signature_valid` re-described. `__version__ # "0.5.0"` → `"0.8.0"`. |
| `sdk/python/configuration.mdx` | `connect/configuration.mdx` | `pip install glacis[content-safety]` → that extra does not exist. `path: "~/.glacis/glacis.db"` removed from the example — `~` is not expanded. |
| `sdk/python/controls.mdx` | `reference/controls.mdx` | Same `content-safety` fix. `ControlExecution.version` re-described (it is stamped `"0.7.0"` in a 0.8.0 package). GitHub org corrected from `glacisai` to `Glacis-io` in demo links. |
| `sdk/python/sampling.mdx` | `reference/sampling-and-evidence.mdx` | The Evidence Model section showed an offline example reading `receipt.evidence`, which is always `None` offline. `should_review() requires offline mode` → it requires an HMAC key. `from glacis.integrations.base import get_evidence` → the public path is `glacis.integrations`. |
| `sdk/python/storage.mdx` | `reference/storage.mdx` | `storage_path="/path/to/storage"` crashes — it must be a `Path`. `~` expansion documented. "This ensures a complete audit trail" softened to what append-only storage actually buys. |
| `sdk/python/judges.mdx` | `reference/judges.mdx` | Links and title only; content was accurate. |
| `sdk/python/batch.mdx` | `reference/operations.mdx` | Merged `pipelines.mdx` in, fixed its unrunnable constructor, and added the fail-open warning about gaps in an operation chain. |

### Infrastructure

| File | Note |
| --- | --- |
| `astro.config.mjs` | New sidebar; **17 redirects** from every old `/sdk/python/*` URL. `pyproject.toml` advertises `https://docs.glacis.io/sdk/python` as the package documentation URL, so those URLs had to keep resolving — they now land on the new pages instead of 404ing. |
| `src/components/ReceiptBadge.astro` | The three states, rendered identically everywhere. Each badge carries a glyph *and* a word, so the distinction survives greyscale, colour-blindness and screenshots. |
| `src/components/Screenshot.astro` | Obvious placeholder for captures that do not exist yet — never a mock-up passed off as a screenshot. |
| `scripts/verify-doc-snippets.py` | New. See above. |
| `src/styles/custom.css` | Unchanged theme, one addition: wide tables now scroll inside themselves below 50rem instead of pushing the page sideways. The Start section has to work at 375px. |
| `README.md` | Rewritten: IA, the five editorial rules, how to run the verification script. |

---

## Corrections worth calling out

These are cases where the previous docs stated something the SDK does not do.
All are now pinned by checks so they cannot silently reappear.

1. **Offline `verify()` does not verify a signature.** `Glacis._verify_offline`
   derives a public key from the client's own `signing_seed` and compares it to
   `attestation.public_key`; the `signature` field is never read. With no seed
   available, `signature_valid` is hardcoded `True`. `Ed25519Runtime` has no
   `verify()` method at all in 0.8.0. Demonstrated by zeroing a signature and
   watching `verify()` still return `valid=True`.
   → documented on `/reference/api/`, `/verify/cli/`, `/connect/quickstart/`,
   with a working PyNaCl replacement.

2. **`python -m glacis verify` is structural for offline receipts.** Same
   demonstration through the CLI: a receipt with 128 zeroes for a signature
   still prints `Status: VALID` / `Signature: PASS` and exits 0.

3. **Wrapper attestation is fail-open.** `attest_and_store()` catches every
   exception and prints only in debug mode. The model response is always
   returned; the receipt may simply not exist, leaving `get_last_receipt()`
   `None` or holding the previous call's receipt. Not mentioned anywhere in the
   old docs; now on every wrapper page and in `reference/operations.mdx`.

4. **`glacis[content-safety]` does not exist.** The SDK's own `ImportError`
   names it. The declared extras are `dev`, `openai`, `anthropic`, `gemini`,
   `litellm`, `redaction`, `jailbreak`, `controls`, `all` — asserted against
   `pyproject.toml` by the check script.

5. **`~` is not expanded in storage paths.** An explicit
   `path: "~/.glacis/glacis.db"` — which the old configuration and storage pages
   both recommended — creates a directory literally named `~` beside the process
   working directory. The defaults are already resolved against `Path.home()`,
   so the fix is to omit the key.

6. **`storage_path` / `db_path` must be `Path` objects.** A string raises
   `AttributeError: 'str' object has no attribute 'mkdir'` at construction. The
   old storage page passed a string.

7. **`ControlExecution.version` is stamped `"0.7.0"`** from a constant in
   `glacis/integrations/base.py` that was not bumped with the package. It is not
   the installed version and should not be read as one.

8. **`Attestation.evidence` is always `None` offline.** Only the online path
   attaches it, driven by the server's sampling decision.

9. **~~The browser verifier does not read SDK offline receipts.~~**
   **Superseded — see Corrections round 2, finding 10.** True of the build
   observed on 2026-08-08 (`glacis.io/verify` detected only `v2`,
   `v1-gateway` and `v1-scanner`, and returned "Unrecognized receipt format"
   for a flat `oatt_…` receipt). The launch build on the announcement branch
   adds an `sdk-offline` format with a real Ed25519 check, so `/verify/` now
   documents that capability with explicit "as of the launch build" scoping and
   keeps the Python routine as the fallback.

---

## The request-access dead end — grep proof

```
$ grep -rniE "request[[:space:]]+access" . \
    --exclude-dir=node_modules --exclude-dir=dist --exclude-dir=.git
$ echo $?
1
```

(The pattern is written as a regex so that this file does not match its own
grep. Run it verbatim from the repo root.)

**Zero occurrences** across the whole worktree. The single instance was in
`sdk/python/quickstart.mdx`; the call to action is now **"Create a workspace"** →
<https://app.glacis.dev>, which is where signup actually lives (the live
`/login` page carries the *Create account* form — confirmed by GET on
2026-08-08).

---

## Verification run for this rebuild

| Check | Result |
| --- | --- |
| `npm run build` | Green — 25 pages, 42 HTML files including redirects |
| `python docs/scripts/verify-doc-snippets.py` | 53/53 checks pass (round 1) · **77/77 (round 2)** |
| Internal link check over `dist/` | 42 pages, **0 broken links** |
| Cross-page anchor check over `dist/` | **0 missing anchors** |
| `grep -rniE "request[[:space:]]+access"` | **0 occurrences** |
| `grep -rniE "compliant\|protected\|certified"` | Only inside explicit negations |

Live surfaces probed read-only (GET) on 2026-08-08 to keep claims true:
`app.glacis.dev/login` (200, carries *Create account*), `glacis.io/verify` (200,
client-side verifier, currently a `?receipt=` share link), `overt.is` (200,
OVERT 1.1), `docs.glacis.io` (200), `api.glacis.io/v1/root` (404 JSON).

---

## Corrections — round 2 (Codex launch-gate review, 2026-08-08)

The external Codex launch-gate review
(`glacis-know/2026-08-08_codex-launch-gate-review.md`, session
`019fe30b-b67d-7db2-b9f8-72cff2793709`) returned **NO-LAUNCH** for this branch:

> The rebuilt docs contain launch-blocking receipt-tier conflation, label any
> online SDK attestation WITNESSED without evidence, and make several
> primary-path claims the SDK source does not support.

Eleven findings were raised against Branch 3. Every one is listed below with
what was actually done. Every corrected claim was re-verified against the
`glacis` 0.8.0 source in this worktree, and each is now pinned by an executable
check (53 → **77**).

### 1 · SAMPLE modelled as a third cryptographic tier — **blocker** — fixed

*`src/components/ReceiptBadge.astro:3-14,21-35` modelled SAMPLE, SELF-SIGNED and
WITNESSED as three mutually exclusive states, and SAMPLE said "nothing was
signed for real" — while `start/sample-workspace.mdx:39-68` said a sample may
carry and pass a valid signature.*

SAMPLE is now an **orthogonal data-provenance flag**, not a tier. The component
carries two independent axes — tier (`self-signed` | `witnessed`) and provenance
(`sample`) — and `<ReceiptBadge kind="self-signed" sample />` renders both chips,
because that is exactly what a signed sample receipt is. `kind="sample"` alone
still renders the flag with the tier deliberately unstated, for legend rows; it
never means "unsigned". The gloss now reads *"Synthetic subject matter — the
operation it describes never ran. The signature over it is still a real
signature."*

`start/sample-workspace.mdx` says plainly that the signature over fabricated
content is real and verifies, and scopes the caution to what a passing check on
a sample receipt does establish (unedited bytes) versus what it does not (any
event at all). The three-state legends on `index.mdx`, `start/index.mdx`,
`verify/index.mdx` and `verify/what-a-check-proves.mdx` are now two tiers plus
the flag. `docs/README.md` editorial rule 2 was rewritten to match, so the next
author cannot reintroduce the conflation from the style guide.

### 2 · Online mode equated with a witnessed receipt — **blocker** — fixed

*`connect/offline-vs-witnessed.mdx:48-62` equated online mode with a
countersigned, transparency-logged WITNESSED receipt. `witness_status` is
computed solely as `"WITNESSED"` whenever `is_offline` is false
(`glacis/models.py:304-306`), and the returned `Attestation` exposes only one
public key/signature.*

The page was rewritten and retitled **"Offline vs online"** — calling it
"offline vs witnessed" was itself the conflation. The URL is unchanged
(`/connect/offline-vs-witnessed/`); inbound link text was updated. It now states:

- **`witness_status` is a transport-mode label.** The property's only input is
  `is_offline`; the source is quoted inline. The `Attestation` model has no field
  for a second signature, an inclusion proof or a tree head, and
  `_normalize_server_response` keeps a fixed field list — so a transparency proof
  in a server response would not survive into the returned object.
- **What online mode returns today:** a record signed by a party other than you
  (the service). That is a real step up from a diary and is *not* the witnessed
  artifact. `verify(id)` can return a `proof` and `tree_head`, but that is the
  server answering about its own record; there is no Merkle verification code
  anywhere in the package.
- **Where the witnessed tier comes from:** the launch build's portal mint path,
  whose response carries the witness's inclusion envelope — with two caveats
  kept in the text, that independence is a property of the deployment rather
  than the format, and that a witnessed artifact is not an SDK object.
- A direct instruction for anyone building a UI over SDK output: render
  `SELF-SIGNED` for offline, and for online render what you actually verified —
  a `WITNESSED` string is not that verification.

The diary-vs-evidence framing is kept, but scoped to what each **artifact**
proves rather than to a mode name.

### 3 · "Altering any character breaks it" — high — fixed

*The offline signed payload (`glacis/client.py:417-460`) excludes the receipt
id, `cpr_hash`, `public_key`, `is_offline` and other returned fields.*

`verify/what-a-check-proves.mdx` now carries two tables, both measured:

| Inside the signature | `version`, `mode`, `service_id`, `operation_type`, `evidence_hash`, `timestamp_ms` (the `str()` of `timestamp`), `operation_id`, `operation_sequence`, and — when present — `control_plane_results` (the whole structure) and `supersedes` |
| Outside it | `id`, `cpr_hash`, `public_key`, `is_offline`, `evidence` / `review` / `sampling_decision`, and any extra key added to the file |

with the sentence *"'Altering any character breaks the signature' is false"*
written down, and the true version — altering any **signed** field breaks it —
in its place. Ten tamper checks in the harness assert both halves.

### 4 · "Prove what your AI system did" — high — fixed

*`index.mdx:1-6` and `start/index.mdx:34-42`. `Glacis.attest()` accepts
caller-supplied input, output, operation labels and control results
(`client.py:256-320`) — it proves a caller generated a claim.*

The hero, its description and the Start "What a receipt is" section now mirror
the approved Utah press-release articulation: **a receipt is cryptographic
evidence that the specific claims stated in it were generated for a given
event**, at a point in time, by the holder of a particular key — and explicitly
not a compliance certificate, an audit result, a guarantee that a control was
effective, or a statement that the model was right. `start/mint-a-receipt.mdx`
was brought into line, and `what-a-check-proves.mdx` no longer says a receipt
"records that the exchange happened".

### 5 · "Every call through it is attested" — high — fixed

*`connect/index.mdx:23-45` and the provider-page descriptions.
`attest_and_store()` (`glacis/integrations/base.py:932-984`) swallows every
attestation/storage exception by design.*

The headline is now *the wrapper attempts a receipt on every wrapped call —
best effort, never blocking*, and a new **"Attestation is best effort, never a
gate"** section on `connect/index.mdx` spells out the consequences: the provider
response is always returned, no receipt exists on failure, `get_last_receipt()`
returns `None` **or the previous call's receipt from the same context**, nothing
retries, and nothing surfaces the failure to your code. The one case that does
raise — `GlacisBlockedError` from a blocking control — is named so it is not
confused with an attestation failure.

The four provider pages' frontmatter descriptions no longer say "every call
produces a receipt", and each page now opens with the best-effort sentence
instead of burying it two-thirds down.

### 6 · Undefined `seed`, and a config-first path that cannot sign — high — fixed

*`connect/index.mdx:51-77` and `connect/litellm.mdx:137-153` referenced an
undefined `seed`; `connect/configuration.mdx:310-326` omitted a seed entirely
even though offline is the default and the factory raises
(`integrations/base.py:297-305,358-360`).*

All three snippets now define the seed from the environment. `connect/index.mdx`
and `connect/configuration.mdx` additionally say **why** the YAML cannot supply
it — a signing key does not belong in a file you commit — and quote the exact
failure, `ValueError: signing_seed is required for offline mode`. The harness
asserts that failure.

### 7 · RFC 8785 — medium — fixed

*`glacis/crypto.py:56-83` uses Python `json.dumps`, Python number formatting and
Unicode-code-point key sorting.*

A new Quickstart section, **"The canonicalisation actually used"**, tabulates
every divergence, each measured against the installed 0.8.0:

| | `glacis.crypto` | RFC 8785 |
|---|---|---|
| keys | Python `sorted()` — code point | UTF-16 code unit |
| non-ASCII | escaped `\uXXXX` | literal UTF-8 |
| `1.0` | `1.0` | `1` |
| `1e16` | `1e+16` | `10000000000000000` |
| integers | arbitrary precision | IEEE-754 doubles |

…with the honest scoping that ASCII strings and integers **do** agree, which is
why the browser verifier can re-derive an offline receipt's signed bytes, and a
note that the signature path and the hash path are two different serialisers.
The five RFC 8785 claims in `reference/api.mdx` were corrected to the same
scoping, and the note asserting RFC 8785 guarantees cross-language agreement is
now a caution saying it does not.

### 8 · Transmission table and local retention — medium — fixed

*The table claimed a timestamp is sent (`client.py:338-359` has none), and "full
input/output retained locally" is too broad for direct offline `attest()`, which
stores 100-character previews (`client.py:463-470`).*

The timestamp row is gone, from `connect/offline-vs-witnessed.mdx` and from
`index.mdx`'s summary table, with a sentence saying the SDK never sent one. The
retention claim is now a per-call-path table:

| `attest()`, offline | attestation row + **100-character previews** of `str(input)` / `str(output)` |
| `attest()`, online | **nothing** — the online path never writes to the local receipt store |
| provider wrappers, either mode | full request and response, via `store_evidence()` |

### 9 · "Everything else is identical" — medium — fixed

*`connect/quickstart.mdx:143-163`. `_attest_online` (`client.py:322-393`) does
not write to the client's receipt store.*

The section is now **"Make it online"** with a table of the six things that
actually change (signer, id prefix, `witness_status`, local storage,
`get_last_receipt()` raising, `verify()` becoming a server round-trip, `evidence`
population), and a paragraph calling out the storage difference as the one that
surprises people.

### 10 · Browser verifier capability was stale — medium — fixed

*`verify/index.mdx:37-47` said the browser verifier cannot parse SDK offline
receipts; the announcement branch's `verify.html` now detects and verifies them.*

Rewritten with explicit **"as of the launch build"** scoping: the page
recognises `oatt_…` receipts (`is_offline`, or the id prefix, plus a `signature`
and `public_key`), re-derives the exact bytes the SDK signed, runs a real
Ed25519 check with WebCrypto, and prints a *signed field coverage* line naming
what is and is not inside the signature. The `#r=` fragment is verified on load,
with the legacy `?receipt=` form still opening. The old advice survives as a
fallback rather than being deleted, in case a reader hits an earlier build.

One caution was **added**, which the previous version did not have: for an SDK
offline receipt the signature is checked against the public key carried in that
same receipt, which establishes internal consistency and nothing about who holds
the key.

### 11 · OVERT conformance claim — medium — fixed

*`overt.mdx:10-13,64-77` said the receipts in these docs implement OVERT 1.1,
then admitted the Python offline receipt is a different flat SDK-native shape.*

The opening now says GLACIS publishes OVERT and builds its receipt format
against it, followed by a caution naming which artifact is which: the platform /
gateway receipt is the artifact written against OVERT 1.1; **the SDK's flat
`oatt_…` offline receipt is not an OVERT document** — no OVERT envelope, no
per-signal recomputable/operator-dependent marking, no version identifier naming
the standard. These docs claim no conformance for it. The relationship is
described as conceptual (same content-safe commitments, same operator-statement
assurance level, different container) rather than structural, and the verifier's
ability to read it is described as a fact about that verifier.

### What the round-2 checks pin

24 new executable checks, 53 → **77**, so none of the above can silently drift
back:

| Finding | Checks added |
| --- | --- |
| 3 — signed-field boundary | 11 (`id`, `cpr_hash`, `is_offline`, an arbitrary extra key outside; `service_id`, `operation_type`, `evidence_hash`, `timestamp`, `operation_id`, `operation_sequence`, CPR content inside) |
| 7 — canonicalisation | 5 (non-ASCII escaping, whole floats, `1e16`, code-point vs UTF-16 key order, NaN/Infinity) |
| 2 — witness_status | 3 (computed from `is_offline` alone; no countersignature field on the model; the normaliser drops a transparency proof and a second signature) |
| 8, 9 — mode differences | 4 (`get_last_receipt()` raises online; `_attest_online` has no `store_receipt`; the offline store keeps 100-character previews, asserted against a 500-character payload) |
| 6 — config-first seed | 1 (`create_glacis_client(offline=True, signing_seed=None)` raises `ValueError`) |

### Round-2 verification run

| Check | Result |
| --- | --- |
| `cd docs && npm run build` | Green — 25 pages, 42 HTML files |
| `python docs/scripts/verify-doc-snippets.py` | **77/77 checks pass** |
| Internal link check over `dist/` | 42 pages, **0 broken links** |
| Cross-page anchor check over `dist/` | **0 missing anchors** |
| `grep -rniE "request[[:space:]]+access"` | **0 occurrences** |
| `grep -rniE "compliant\|protected\|certified"` | 4 hits, all inside explicit negations |

## Corrections — round 3 (Codex re-review, 2026-08-08)

Second Codex launch-gate pass
(`glacis-know/2026-08-08_codex-rereview-2.md`). It confirmed the round-2 fixes
as real and found four things still wrong on this branch — plus one **SDK
defect** the rebuilt docs and harness had both missed.

This round is the first that changes SDK source. Those changes are staged as
**0.8.1.dev0 and are NOT published**: `pip install glacis` still serves 0.8.0.
Every page that describes a behaviour 0.8.1 changes says which version it is
talking about.

### 1 · Persisted offline receipts lost signed CPR content — **SDK defect** — fixed in source

*Codex finding 4. `control_plane_results` is inside the offline signature
(`client.py`), but SQLite `store_receipt` (`storage.py:336`) and
`_row_to_attestation` (`storage.py:468`) both omitted it. A receipt that
verified before storage failed independent Ed25519 verification after reload,
and offline `verify()` did not notice because it only compares the locally
derived public key.*

Confirmed and reproduced here, on **both** backends — the JSONL line dropped it
too, which the finding did not name. Fixed in the SDK, not documented around:

- SQLite schema **v5** adds `offline_receipts.control_plane_json`, written whole
  and never truncated, with a v4→v5 migration; the JSONL receipt line gains
  `control_plane_results`; both backends reconstruct it on read.
- Fixing the migration required fixing migrations: `version` is the primary key
  of `schema_version`, so `INSERT OR REPLACE` **appended** a row instead of
  replacing one and the reader took the oldest. The schema version never
  advanced past the first migration and every migration re-ran on every open.
  Now `MAX(version)` on read, one row on write.
- **Legacy rows degrade by name.** The content was never written, so it cannot
  be recovered. A row carrying a `cpr_hash` with no stored content is *not*
  reconstructed as "this receipt had no control-plane results":
  `Attestation.cpr_recovery_error` states the reason, and `verify()` fails
  closed with that reason as its `error`. Receipts that genuinely had none are
  untouched and still verify.
- 17 new SDK tests in `tests/test_cpr_persistence.py`: round trip on both
  backends, a hand-built schema-v4 database that migrates and then reads back as
  a named degradation, a legacy JSONL line, and the proof that the marker never
  reaches the signed payload.

Documented on `/reference/storage/` (including a rule for anyone writing a
custom backend), `/verify/what-a-check-proves/` and the retention table on
`/connect/offline-vs-witnessed/`.

### 2 · The wrapper published its receipt before storing evidence — **SDK defect** — fixed in source

*Codex finding 2, second half. `attest_and_store()` called `set_last_receipt()`
before `store_evidence()` (`base.py:957`), so a storage failure left a current
receipt for a call the docs say produced none.*

Order swapped: the receipt is published only after storage returns. The
attestation may still exist after a storage failure — `attest()` writes the
offline receipt row itself, and an online attestation is already on the server —
so the honest statement is that the wrapper no longer presents it as complete.
`connect/index.mdx` carries the 0.8.0 gap explicitly, because that is what
readers have installed.

### 3 · "Never blocks, delays" and "nothing anywhere retries" — **blocker** — fixed

*Codex finding 2. `connect/index.mdx:117`. Both false: `attest_and_store()` runs
synchronously after the provider response, and online attestation makes up to
four attempts with 30-second timeouts and blocking exponential backoff.*

Rewritten as arithmetic rather than reassurance. The page now states that
attestation runs on the caller's thread between the provider response and the
return; the four defaults from `client.py`; the 1s/2s/4s backoff with up to 30%
jitter; the **129.1-second** worst case (4 × 30s + 1.3 + 2.6 + 5.2); and the
measured common case — a probe against a closed local port makes exactly four
attempts and returns after 8.4 seconds, all of it sleeping. It also says which
failures do *not* retry: a 4xx raises `GlacisApiError` and a 429 raises
`GlacisRateLimitError` on the first attempt.

Bounding it is where the honesty matters. `Glacis(timeout=…, max_retries=…)`
exists, but no wrapper factory accepts or forwards it and `glacis.yaml` has no
such setting, so at 0.8.0 there are exactly two answers: run the wrapper
offline, or construct your own client and call `attest()` yourself. The page
says that instead of pointing at knobs the wrapper does not expose.

### 4 · The corrected data-boundary tables still overclaimed — **blocker** — fixed

*Codex finding 3. Three claims on `/connect/offline-vs-witnessed/`.*

- **Direct online `attest()` retains "Nothing"** — false. It writes nothing to
  disk, but when the server assigns L1 or L2, `_attest_online` attaches the raw
  `input` and `output` to `attestation.evidence.data`. The table now separates
  "written to disk" from "on the returned object", and a caution names the
  consequence the docs themselves create: Quickstart shows
  `json.dump(receipt.model_dump())` and `/verify/` shows sharing a receipt as a
  `#r=` link, so an online L1/L2 receipt can publish a prompt by accident.
  Remedy on the page: `model_dump(exclude={"evidence", "review"})`.
- **Wrappers store "the complete request and response"** — false. Each stores
  its own projection. OpenAI's is enumerated field by field alongside what is
  dropped (every kwarg but `model`/`messages`; `id`, `created`,
  `system_fingerprint`, `tool_calls`, `logprobs`, token-detail breakdowns), with
  the consequence that matters: `evidence_hash` commits to the **projection**,
  so recomputing over the provider's raw JSON will not reproduce it.
- **"Prompts, documents, patient data are not transmitted"** — true of `input=`
  and `output=`, which are hashed; false of `control_plane_results`, which
  online mode puts in the body verbatim. A danger box now says so: it is an
  arbitrary `dict[str, Any]`, nothing inspects or redacts it, `cpr_hash` is sent
  *alongside* it rather than instead of it, and anything the caller puts there —
  including PHI — leaves the machine. The wrapper-built structure is safe by
  construction and its contents are enumerated, so that safety is not mistaken
  for a property of the field.

The same exception is now on the `connect/index.mdx` aside and on
`/reference/api/`.

### 5 · The snippet harness overstated its coverage — medium — fixed

*Codex finding 5. The docstring claimed every documented behaviour and every
canonicalisation divergence; `what-a-check-proves.mdx:48` said all twelve rows
were tamper-tested. The tables have sixteen rows, and the omitted cases included
`supersedes`, several unsigned fields, persisted CPR and the
arbitrary-precision integer divergence.*

All sixteen rows are now tamper-tested — ten signed fields that must break, six
unsigned that must not — and the page says sixteen. The `public_key` row was
also imprecise and is now split into the two things that are true of it:
swapping it alone **breaks** the check, and a full re-sign under another key
verifies perfectly, which is what "not bound to an identity" means.

Added beyond the named gaps: the persisted-CPR round trip pinned in *both*
directions (so the 0.8.0 loss stays pinned even when the fixed build is under
test), the arbitrary-precision integer divergence and its hash consequence past
2⁵³, the retry/latency claims, the online request body including verbatim CPR
transmission, L1/L2 evidence retention, and the wrapper projections. Everything
network-shaped runs against a stubbed transport — no socket is opened and no
host is contacted.

And the structural fix: **the script now prints `NOT COVERED` lines.** Where a
documented claim cannot be executed, it is named with a reason and counted
separately from passes. Seven today (live witnessed mode, the measured 129.1s
worst case, provider wrappers end to end, the browser verifier, the portal mint
path, receipts from a genuinely old install, the dependency-heavy controls). A
green run means every executed check passed; the NOT COVERED block is the list
of what it does not establish. `docs/README.md` says the same, and says never to
describe a green run as "everything is verified".

### What the round-3 checks pin

41 new executable checks, 77 → **118**, plus 7 NOT COVERED lines:

| Finding | Checks added |
| --- | --- |
| 5 — the sixteen-row boundary | 8 (`version`, `mode`, `supersedes` edited, `supersedes` added, CPR removed, `evidence`/`review`/`sampling_decision`, `public_key` swapped, `public_key` re-signed) |
| 4 — persisted CPR | 7 (verifies before storage; carries CPR after reload; verifies after reload; a pre-0.8.1 row loses it; that row fails independent verification; the loss is named; `verify()` fails closed with the reason) |
| 2 — retry and latency | 12 (the four defaults; the backoff sequence; the 129.1s arithmetic; four attempts on connect failure and on 5xx; one attempt on 4xx and 429; both exception types; no thread/queue in `attest_and_store`; ordering inside the wrapper; no timeout knob on any factory; `acompletion()` attesting on the event loop) |
| 3 — the data boundary | 11 (body carries only hashes; CPR verbatim including PHI-shaped fields; `cpr_hash` alongside not instead; key in the header; no timestamp; L1/L2 raw I/O on the object; `model_dump` leaks it and `exclude=` fixes it; offline never populates `evidence`; three wrapper-projection checks) |
| 5 — canonicalisation | 2 (arbitrary-precision integers; an int and its nearest double hash differently past 2⁵³) |
| version scope | 1 (the SDK under test is 0.8.0 or the unpublished 0.8.1) |

### Round-3 verification run

| Check | Result |
| --- | --- |
| `python -m pytest` (SDK) | **499 passed, 63 skipped** (481 before; +18) |
| `python docs/scripts/verify-doc-snippets.py` | **118/118 checks pass, 7 NOT COVERED** |
| `cd docs && npm run build` | Green — 25 pages, 42 HTML files |
| Internal link check over `dist/` | **0 broken links** |
| Cross-page anchor check over `dist/` | **0 missing anchors** |
| `ruff check` on every file this round touched | Clean |
| `grep -rniE "request[[:space:]]+access"` | **0 occurrences** |
| `grep -rniE "compliant\|protected\|certified"` | 8 hits, all inside explicit negations |

## Corrections — round 4

*Third Codex launch-gate pass (`glacis-know/2026-08-08_codex-pass3-final-gate.md`,
session 019fe3db). `glacis-plg` and `labs-plg` passed; this branch was blocked on
five findings. All five are closed below.*

### 1 · Stored CPR tampering was not fail-closed — the load-bearing one — fixed

*Codex finding 1. `_recover_cpr()` accepts any JSON object without checking it
against `cpr_hash`, and `_verify_offline()` checks only `cpr_recovery_error` and
the public key — not the signature. A modified CPR, or a removal of both the CPR
and the unsigned `cpr_hash`, could be reported valid although independent
verification failed.*

The real problem was never `_recover_cpr()`. It was that **offline verification
verified nothing**: `Glacis._verify_offline()` derived a public key from the
client's own seed and compared it to the receipt's, and the CLI checked string
lengths. Both printed `Signature: PASS` over 128 zeroes. A hash-check inside the
storage layer would have patched one symptom of that.

So the check is now real, and there is one of it:

- `Ed25519Runtime.verify()` — `nacl.signing.VerifyKey` over the exact signed
  bytes. **No new dependency**: the SDK already signs with PyNaCl. It returns
  `False` for a well-formed wrong signature and raises `CryptoError` when the
  key or signature cannot be decoded, because "could not check" is a different
  answer from "wrong".
- `crypto.offline_signed_payload()` — the single definition of the signed bytes.
  `_attest_offline()` signs with it and `verify_offline()` rebuilds with it, so a
  change that broke verification would break signing in the same commit.
- `glacis.verify.verify_offline()` is the one implementation. `Glacis.verify()`
  delegates to it and `python -m glacis verify` calls it, so the library and the
  command line cannot disagree about a receipt.

Verification needs **no signing seed** — the public key on the receipt is the
verifier. That is what makes a receipt checkable by the party it was handed to,
and it means verifying a third party's receipt now succeeds when their signature
is good, where 0.8.0 rejected it for not being ours.

`error` names which failure it was: `signature_invalid`, `structural`,
`cpr_unrecoverable`. The hash cross-check Codex asked for is there as well, but
deliberately **not** as the authority: `cpr_hash` is unsigned, so a mismatch is
reported by name (`cpr_hash_mismatch` / `cpr_hash_orphaned`) and the signature
still decides `valid`. Making an unsigned field able to fail a receipt would
have contradicted the boundary page — and been wrong.

Tests tamper with every signed field *after storage*, including editing
`control_plane_json` directly in the SQLite row and the CPR in the JSONL line,
and stripping CPR and `cpr_hash` together so nothing structural is left to
notice. All fail on the signature. Honest receipts still pass.

Five pages then lost caveats that had become false. **A claim weaker than the
code is a truth gap too** — a reader who believes the SDK never checks
signatures writes verification they did not need. What stays on every one of
those pages is the version split: 0.8.1 is unpublished, so `pip install glacis`
still gives you the version that does not check.

### 2 · The timeout claim was still false — fixed

*Codex finding 2. `connect/index.mdx:181` calls 129.1 seconds a wall-clock
ceiling. `httpx.Client(timeout=30)` sets operation timeouts; its read timeout is
per received chunk, not a total request deadline.*

Correct, and the fix is not a hedge. The arithmetic is valid for exactly one
scenario — four attempts that each hang and time out once — and a
slow-dripping response never trips a 30-second read timeout at all. The pages
now say that, say plainly that **there is no default bound**, and say where a
bound can come from: offline mode, a `ThreadPoolExecutor` with
`future.result(timeout=…)` (which bounds your latency, not the process's work),
or `asyncio.wait_for()` around `AsyncGlacis`, which does cancel the request but
is unreachable through the provider wrappers. The four wrapper pages that
repeated "a ceiling of 129.1 seconds" were corrected too.

### 3 · The boundary page contradicted its own `public_key` row — fixed

*Codex finding 3. Line 43 correctly says swapping `public_key` breaks
verification; lines 48–50 then say all six unsigned rows "must not" break. The
new unsigned, verifier-controlling `cpr_recovery_error` is also absent.*

Resolved by splitting the table rather than softening either half, because both
halves were true and the framing was wrong. Five unsigned fields are inert. Two
are unsigned **and** load-bearing — and the reason is stated exactly: they are
not inputs to the signed payload, they are inputs to *the verifier*. Swapping
`public_key` breaks the check because the key **is** the verifier, not because
it is signed content.

`cpr_recovery_error` joins the boundary as the second such row, described
precisely: never signed, never transmitted by the SDK, set by the reader's own
store, and able to turn a good receipt into a refusal but **never a bad receipt
into a pass**. Both directions are pinned.

The tamper table was re-run after the finding-1 fix and re-pinned: seventeen
rows — ten signed that must break, five unsigned that must not, two unsigned
that must.

### 4 · "Safe by construction", and a harness that checked one wrapper — fixed

*Codex finding 4. `offline-vs-witnessed.mdx:196` calls wrapper CPR "safe by
construction" although policy IDs/tags/model/provider metadata are transmitted
verbatim. The harness's NOT COVERED section says all provider projections are
source-pinned, but its assertions inspect only OpenAI's.*

The accurate claim is narrower than "safe": the wrapper never puts the *content
of the exchange* in there, and everything else in it is what you named. Policy
id, policy version, environment and every tag are read off `policy:` in your
`glacis.yaml` and transmitted verbatim; a tag reading `patient-88231` is egress
and nothing inspects it. "Safe by construction" invited a reader to stop
thinking about a field whose contents they choose.

The harness now pins **all four** projections — openai, anthropic, gemini,
litellm — request shape and retained response keys, plus that no wrapper keeps
`system_fingerprint` / `tool_calls` / `logprobs`, that only anthropic and gemini
project a separate system prompt, and that every wrapper puts the *hash* of the
system prompt in the control plane. The NOT COVERED line was rewritten to say
what reading source cannot catch, rather than implying more than it did.

### 5 · A failed migration stamped itself as successful — fixed

*Codex finding 5. `_run_migrations()` swallows every `OperationalError` as
"column already exists", then unconditionally writes version 5.*

Only `duplicate column name` means that, and it is now the only tolerated
failure — it is what re-running a partly applied migration looks like. Anything
else raises `StorageMigrationError` naming the version pair, and the version
stamp is written only after every step has actually been applied, so the
database keeps saying the version it really is and the next open retries.
`_migrate_v3_to_v4()` got the same treatment: it reads the columns before
deciding whether to rename, a failed add-and-copy fallback is a failure rather
than "already fine", and the indexes are built after both columns are known to
exist. A connection whose migrations failed is closed and discarded instead of
being handed to the caller half-opened.

Tested with three crafted databases — a v4 with no `offline_receipts` table, a
v3 with neither hash column, a v2 with no `evidence` table. Each fails by name
with the recorded version untouched. A healthy v4 still reaches v5, and a
partly applied one still re-runs cleanly.

### What the round-4 checks pin

21 new executable checks, 118 → **139**, plus the same 7 NOT COVERED lines:

| Finding | Checks added |
| --- | --- |
| 1 — real verification | 6 (`verify()` rejects a zeroed signature and names it; verification without a seed; a foreign key's receipt verifies; an undecodable key is `structural`; the CLI exits 1 and names `signature_invalid`; the CLI and the library return the same error) |
| 3 — the seventeen-row boundary | 5 (`cpr_recovery_error` is outside the signature; it makes the SDK refuse, by name; it cannot rescue a broken signature; an edited `cpr_hash` still verifies and is named; the CLI reports that note too) |
| 4 — all four projections | 10 (request projection, retained response keys and hashed-projection binding for each of openai/anthropic/gemini/litellm, less OpenAI's three that already existed; plus no `system_fingerprint`/`tool_calls`/`logprobs` anywhere, the system-prompt split, and the system-prompt hash) |

Findings 2 and 5 added no harness checks: the timeout correction is prose about
a scenario the harness already declines to execute, and the migration fix is
covered by the SDK test suite (`tests/test_storage_migrations.py`), which is
where a database-corruption test belongs.

### Round-4 verification run

| Check | Result |
| --- | --- |
| `python -m pytest` (SDK) | **547 passed, 63 skipped** (499 before; +48) |
| `python docs/scripts/verify-doc-snippets.py` | **139/139 checks pass, 7 NOT COVERED** |
| `cd docs && npm run build` | Green — 25 pages, 42 HTML files |
| `ruff check glacis/` and the two new test files | Clean |
| `ruff check docs/scripts/verify-doc-snippets.py` | **3 findings, all pre-existing** — two import-order, one long line, none introduced or removed this round |

### The 0.8.1 changes, in one place

Staged, unpublished, Joe-gated. `CHANGELOG.md` carries the full entry.

| File | Change |
| --- | --- |
| `glacis/crypto.py` | `Ed25519Runtime.verify()`; `offline_signed_payload()` / `offline_signed_payload_for()` — one definition of the signed bytes, used by signer and verifier |
| `glacis/verify.py` | `verify_offline()` is a real Ed25519 check with named failures; the CLI prints a note when a passing receipt disagrees with itself |
| `glacis/client.py` | `_attest_offline` signs the shared payload; `_verify_offline` delegates to `glacis.verify.verify_offline()` |
| `glacis/storage.py` | Schema v5 + v4→v5 migration; persist and reconstruct `control_plane_results` on both backends; `_recover_cpr()` names the loss on legacy rows; `MAX(version)` schema-version read; `StorageMigrationError` instead of stamping a version a failed migration never reached |
| `glacis/models.py` | `Attestation.cpr_recovery_error` — SDK convenience, never signed, never transmitted |
| `glacis/integrations/base.py` | `attest_and_store()` publishes the receipt after storage, not before |
| `pyproject.toml`, `glacis/__init__.py` | `0.8.0` → `0.8.1.dev0` (unchanged in round 4) |
| `tests/test_cpr_persistence.py` | New in round 3 — 17 tests |
| `tests/test_offline_signature_verification.py` | New in round 4 — 36 tests: every signed field tampered after storage, the unsigned rows that must not break, tampering with the store itself, and the CLI |
| `tests/test_storage_migrations.py` | New in round 4 — 12 tests: healthy and partly applied migrations, and three crafted-corrupt databases that fail by name |
| `tests/test_integration_base.py` | One test: a storage failure leaves no current receipt |

---

## Corrections — round 5

*Fourth Codex launch-gate pass (`glacis-know/2026-08-08_codex-pass4-sdk.md`,
session 019fe40d). `glacis-plg` and `labs-plg` passed at round 3; this branch was
blocked on two findings, with three further precision notes. All five are closed
below. Version stays `0.8.1.dev0` — still unpublished.*

### 1 · An unsigned field chose whether a signature got checked — fixed

*Codex finding 2. `Glacis.verify(Attestation)` trusts the unsigned `is_offline`
to select the verifier. Changing `is_offline=False` and `id` to a valid online
attestation routes to `_verify_online(id)` and never examines the supplied
object's bad signature. The CLI has the equivalent reclassification.*

Round 4 made `verify_offline()` a real Ed25519 check and Codex confirms it is
fail-closed. This is the layer above it: **the dispatch**. Two unsigned fields
chose which verifier ran, so an attacker holding a receipt could route around
the check entirely. The signature was never wrong; it was never consulted.

The distinction matters for what the fix had to be. Hardening the offline
verifier further would have done nothing. What was needed was a dispatch where
`is_offline` cannot subtract a check:

- **The offline signature check runs on every supplied `Attestation` object,
  always.** Nothing skips it.
- `is_offline=False` **adds** a lookup of `receipt.id`. That answer is about an
  id, and anybody can put any id on any object, so it is applied to the object
  only if the two **bind**: matching `signature` and `evidence_hash` — the
  Arbiter's signature and the commitment to the exchange — plus `service_id` and
  `operation_type` when both sides carry them.
- **Bound**, the server's `VerifyResult` is returned with `error` naming the
  binding *and* listing what the log entry carries nothing about
  (`control_plane_results`, `cpr_hash`, `evidence`, `review`, `timestamp`,
  `operation_id`, `operation_sequence`, `supersedes`). A verdict that covers
  four fields should not be read as covering twelve.
- **Unbound**, none of the server's answer is returned — not the org, not the
  proof, not the tree head. The result is the object's own `OfflineVerifyResult`,
  `valid` reflects the bytes that were actually checked, and `error` says the
  lookup happened and was not applied. An object that is not the record cannot
  borrow the record's verdict, and cannot be dressed in its evidence either.

`glacis.verify.verify_attestation()` is the one implementation, called by
`Glacis.verify()` and by `python -m glacis verify`, so the library and the
command line cannot disagree — the same rule round 4 applied to `verify_offline`.
The CLI now prints the type of check it *actually ran* rather than the one the
file asked for: a receipt claiming to be witnessed that does not bind comes back
labelled `Offline`.

One consequence worth stating plainly, because it is a behaviour change:
`verify()` on an `Attestation` object with `is_offline=False` now returns an
`OfflineVerifyResult` when the object does not bind, where it previously
returned whatever the server said about the id. That is the fix, not a
side-effect.

Eight adversarial tests, all **through public dispatch** rather than raw
internals: the pass-4 attack exactly as reported; a zeroed signature
reclassified the same way; a server saying `valid=True` about an id that does
not describe the bytes; an honest receipt with the flag flipped (still valid,
route named); an object that binds (server's verdict, binding named); a bound
object whose record is invalid; the flip in the other direction (narrows the
check, never reaches the network); and the CLI path.

The no-op the review found — `tests/…:257` setting `is_offline` to `True` on a
receipt whose `is_offline` was already `True` — is replaced by a real flip.

### 2 · A tolerated ALTER error still stamped a corrupt schema v5 — fixed

*Codex finding 1. `_is_duplicate_column()` treats the duplicate-column message
as sufficient proof that the migration is complete. Reproduced in memory: a
database declaring v4 with only `offline_receipts(control_plane_json TEXT)`
swallowed the duplicate-column error and finished with `version 5`, `columns
['control_plane_json']`.*

Round 4 made every step fail loudly and that was genuinely not enough, for a
reason worth writing down: **"duplicate column name" is evidence about one
column.** It says the column this step adds is already there. It says nothing
about the other fifteen, the second table, or the indexes. Treating it as
"migration complete" is inferring a schema from a single error string.

So the required shape is now stated as data — `REQUIRED_SCHEMA`, cumulative, one
entry per version target — and `_validate_schema()` checks it against the live
database **after every step set**, before anything is stamped. Every missing
table, column and index is collected and named in one message rather than
surfacing one per attempt. Nothing is stamped until the declared schema, the
same one a freshly created database gets, validates in full.

Two details that keep it from being decorative:

- **Fresh databases are validated too.** That is where `SCHEMA` and
  `REQUIRED_SCHEMA` would silently drift apart; now a disagreement fails every
  new store, loudly, in the test suite.
- **`ENSURE_DECLARED_INDEXES`** brings across the three convenience indexes on
  `offline_receipts` that no version step ever created. A database migrated from
  v4 was missing them permanently, so requiring them without creating them would
  have been a check nothing could pass.

Codex's exact reproduction now raises `StorageMigrationError` naming the fifteen
missing columns, with the recorded version still 4 and the table still holding
its one column. Tested per version target, each fixture built so that every
ALTER for that target *succeeds* and the result is still not what the version
means: an `evidence` table that is not one (v2), a missing `offline_receipts`
(v3), an `evidence` left at the v2 shape (v4), and Codex's own database (v5).

**The v4 fixtures in the test suite were not v4.** Three test files hand-built a
"v4" database with no `evidence` table — which `v1→v2` adds — and the new
postcondition rejected all of them. They now build the database they claimed to
be. That is the check working on the tests themselves, and it is the reason nine
tests had to change.

### 3 · `is_offline` was documented as inert — fixed

*Codex: the documentation incorrectly calls `is_offline` inert at
`what-a-check-proves.mdx:37`; the harness changes it to false but checks raw
signature bytes rather than public dispatch.*

It moves to the load-bearing table beside `public_key` and `cpr_recovery_error`:
unsigned, not an input to the signed payload, and an input to **the verifier**.
`id` joins it, because the fix gave `id` the same character — when `is_offline`
is false the identifier chooses which log entry is fetched. Four inert rows, four
load-bearing, **eighteen** in all, and the prose count says so.

The page also carries a caution naming what it used to say and why that was
worse than merely wrong.

In the harness, `is_offline` leaves the raw-bytes loop — true of it and beside
the point — for three checks through public `Glacis.verify()` dispatch against a
stubbed transport that answers per id, because answering per id is what makes
binding necessary in the first place.

### 4 · The `cpr_hash_mismatch` advisory was documented on the wrong result — fixed

*Codex: `valid=True` plus `cpr_hash_mismatch` is acceptable — the signature
authenticates CPR content and the unsigned hash is advisory. The API reference
documents it under online `VerifyResult`, while `OfflineVerifyResult.error`
still says "if invalid".*

It is on `OfflineVerifyResult.error`, which now reads: set when verification
failed, **or** carries the advisory `cpr_hash_mismatch` on a receipt whose
`valid` follows the signature. `VerifyResult.error` describes the binding note it
does carry instead.

### 5 · The harness still called 129.1 seconds a ceiling — fixed

*Codex: the timeout product documentation now correctly describes one scenario;
the harness still labels 129.1 seconds a "ceiling" and "worst case".*

The pages were corrected in round 4 and the harness was not, which is the
failure mode a snippet harness exists to prevent — the executable record
disagreeing with the page it pins. The check now says what it checks: the
arithmetic of one scenario, four attempts that each hang and time out once. Its
detail line and the NOT COVERED entry both name what nothing here reaches — that
`httpx` read timeouts cap the gap between chunks, so there is no default bound
on a call at all.

### What the round-5 checks pin

3 new executable checks, 138 → **141** (the `is_offline` raw-bytes row was
removed as part of finding 3), and an eighth NOT COVERED line — that a real
log entry binds to the attestation object it describes, which needs one call
against a live endpoint:

| Finding | Checks added |
| --- | --- |
| 1 / 3 — dispatch and the boundary table | 3 (a bad signature reclassified as online still fails, naming the signature, through public dispatch; an honest receipt with the flag flipped still verifies with its route named; `id` chooses which entry is fetched and a receipt that does not match it cannot borrow its verdict) |

Findings 2, 4 and 5 added none: the migration fix is covered by the SDK test
suite where a database-corruption test belongs, and the other two are wording on
pages the harness already pins.

### Round-5 verification run

| Check | Result |
| --- | --- |
| `python -m pytest` (SDK) | **562 passed, 63 skipped** (547 before; +15) |
| `python docs/scripts/verify-doc-snippets.py` | **141/141 checks pass, 8 NOT COVERED** |
| `cd docs && npm run build` | Green — 25 pages, 42 HTML files |
| `ruff check` on every file this round touched | Clean — including `docs/scripts/verify-doc-snippets.py`, whose three pre-existing findings were fixed rather than carried forward again |

### The 0.8.1 changes added in round 5

| File | Change |
| --- | --- |
| `glacis/verify.py` | `verify_attestation()` — the one dispatch for a supplied object, shared with the CLI; `bind_to_log_entry()`; the CLI reports the check it ran, not the one the file asked for |
| `glacis/client.py` | `Glacis.verify()` delegates object dispatch to `verify_attestation()`; `is_offline` no longer selects the verifier |
| `glacis/storage.py` | `REQUIRED_SCHEMA`, `DECLARED_INDEXES`, `_validate_schema()`, `ENSURE_DECLARED_INDEXES` — postconditions checked after every step set and before any version is stamped, on fresh databases as well as migrated ones |
| `tests/test_offline_signature_verification.py` | 36 → **43 tests**: eight adversarial dispatch tests through the public surface, replacing a no-op |
| `tests/test_storage_migrations.py` | 12 → **20 tests**: a postcondition test per version target, including Codex's exact reproduction |
| `tests/conftest.py` | `V4_EVIDENCE_TABLE` — what a database that really reached v4 has; the `sample_verify_response` log entry now describes `sample_attestation_data`, so a supplied object can bind to it |

## Residuals

1. **The LiteLLM "config-only path" is config-*first*, not code-free.** The plan
   asked for a config-only LiteLLM path. The genuinely code-free version is a
   callback dropped into a LiteLLM **proxy** config, and the component that does
   that (`glacis_callback.py` in the gateway) is marked *PRIVATE, UNPUBLISHED* and
   is not on PyPI. Documenting it as available would have been false. What
   shipped is the honest version — everything in `glacis.yaml` plus one factory
   call — with an explicit note saying why the proxy path is not a `pip install`.
   If a public proxy callback ships, `connect/litellm.mdx` gains a section.

2. **Start-section screenshots are placeholders.** `Screenshot.astro` renders an
   obvious *SCREENSHOT PENDING* frame. The portal first-run screens (A2) were
   still being built in a parallel worktree, so capturing them would have meant
   photographing something unfinished. Swap each component call for an `<Image>`
   once the flow is deployed.

3. **The Start pages describe the launch build of app.glacis.dev.** Account
   creation and email verification are live and were confirmed by GET. The
   system profile, the sample-receipt workspace and mint-from-browser are A2's
   deliverables landing in the same release. `start/index.mdx` carries one
   explicit scope note saying so, and the mint page documents the flag-gated
   "witnessing is not available yet" state rather than assuming it is on.

4. **The `#r=` fragment permalink is documented against B2's contract, not a
   deployed page.** `glacis.io/verify` today builds share links with a
   `?receipt=` **query parameter**, which puts the payload where server logs can
   see it. B2 replaces it with the fragment form. `/verify/` documents `#r=`,
   explains why the fragment matters, and tells readers to paste the receipt
   instead if the build they hit has not picked it up.

5. **Witnessed-mode and provider-wrapper snippets are unexecuted.** They need a
   live `api.glacis.io` and paid provider keys. Shapes come from the SDK source
   and the wrapper signatures are asserted, but every such page carries an
   explicit *"untested against a live endpoint"* note. Once A1's keys and the
   backend are up, the verification script should grow an opt-in online section.

6. **Some SDK bugs were documented rather than fixed — fewer now.** Items 1–8
   in *Corrections* are defects in `glacis` 0.8.0, not in the prose, and the
   original round treated changing behaviour under a released version number as
   out of scope. Rounds 3 and 4 changed that: persisted CPR, receipt ordering in
   the wrapper, real offline verification and honest migrations are all fixed in
   `0.8.1.dev0`. What is *not* fixed remains documented — the `witness_status`
   string is still computed from `is_offline` alone, and the wrapper factories
   still cannot forward `timeout` / `max_retries` (residual 17).

   **The fix that closes the gap is a release.** Everything above is staged and
   unpublished, so every user on `pip install glacis` still has the SDK the docs
   describe as defective.

7. **`pyproject.toml` still advertises `https://docs.glacis.io/sdk/python`.**
   Left alone deliberately: it is package metadata for a published release, and
   the redirect resolves it correctly. Worth updating to `https://docs.glacis.io/`
   at the next version bump.

8. **The binding pair has never met a real server.** Round 5's dispatch fix
   compares a supplied object's `signature` and `evidence_hash` against the log
   entry `GET /v1/verify/{id}` returns, and nothing in this repo can reach a live
   `api.glacis.io`. Every binding test — the SDK suite and the snippet harness
   alike — runs against a stub whose entry shape we chose from
   `models.AttestationEntry`.

   If a live endpoint returns the signature in some other form, a genuine
   witnessed `Attestation` object passed to `verify()` would be reported
   **unbound**: `valid=False`, with the disagreeing field named in `error`. That
   is the fail-closed direction and it is diagnosable in one read, but it is a
   false negative and it would be a regression for anyone verifying objects
   rather than ids. `verify("att_…")` by id is unaffected — no binding is
   involved.

   One call against a live endpoint settles it, and it should happen before
   0.8.1 is published. It is printed as a NOT COVERED line on every harness run
   so it cannot be lost.

### Added in round 2

8. **The witnessed tier is documented against another branch's launch build.**
   `/connect/offline-vs-witnessed/` says the witnessed artifact comes from the
   portal's mint path, because the SDK demonstrably does not produce one. That
   description was read off the portal branch's source, not off a deployed
   service, and the Codex review raised separate blockers against that branch's
   witness verification. The docs are therefore written so that **nothing here
   depends on the portal being right**: the SDK claims stand on their own, the
   witnessed tier is described as an artifact you can inspect rather than a
   badge we assert, and the caveat that a countersignature is only worth the
   countersigner's independence is on the page. If the portal's witness path
   changes shape, that one section changes with it.

9. **"WITNESSED" as an SDK string is now documented as a defect, not a
   feature.** Adding `witness_status` inputs beyond `is_offline` — or renaming
   the value — is an SDK change under a published version number, which is out
   of scope for a documentation task, exactly as with the round-1 SDK bugs. The
   docs quote the property and say what it does and does not mean. The clean
   fix is upstream: either expose the countersignature and inclusion proof on
   the returned model, or stop calling the online mode "witnessed".

10. **The online request/response shape is still unexecuted.** Everything
    asserted about online mode in round 2 — no timestamp in the body, no local
    receipt store write, the normaliser dropping proofs — is read off the source
    and, where testable without a network, pinned by a check
    (`inspect.getsource`, the normaliser called directly). The end-to-end
    behaviour of a real `api.glacis.io` response is still untested, and the
    pages carrying online snippets still say so.

11. **`glacis.crypto`'s docstrings still claim RFC 8785.** The module header and
    three function docstrings in the published package assert conformance the
    implementation does not have. The docs now describe the real behaviour and
    say the docstrings are wrong; correcting the docstrings is an SDK change.
    Round-2 checks pin each divergence, so the docs cannot drift back to the
    docstring's claim.

### Added in round 3

12. **0.8.1 is staged, not published.** Publishing is Joe's decision, so
    everything in the table above sits in this worktree at `0.8.1.dev0` while
    PyPI still serves 0.8.0. Every page that describes a behaviour 0.8.1 changes
    names the version it is talking about, and describes 0.8.0 as what the
    reader has installed. **Until it ships, the persisted-CPR defect is live for
    every user who attaches control-plane results and relies on the local store.
    Nothing in the docs can fix that; only a release can.**

13. **Receipts already written under 0.8.0 cannot be repaired.** The
    control-plane content was never stored, so there is nothing to migrate. The
    fix is forward-looking: old rows read back with `cpr_recovery_error` set and
    fail verification honestly. If anyone shipped 0.8.0 receipts to a
    counterparty as verifiable evidence, that is a support conversation, not a
    code change.

14. ~~**`glacis.verify()` still does not check a signature.**~~ **Closed in
    round 4.** `Ed25519Runtime.verify()` exists, `glacis.verify.verify_offline()`
    is a real Ed25519 check over the rebuilt signed payload, and both
    `Glacis.verify()` and the CLI call it. Five pages got shorter. What remains
    is the version gap: 0.8.1 is unpublished, so **`pip install glacis` still
    serves the version that does not check signatures**, and every page that
    mentions the check names both versions.

15. **The retry and request-body claims are pinned against a stub, not a
    server.** Round 3 executes the retry loop and captures the online request
    body by replacing the transport, which proves what the SDK *sends* and how
    long it waits. It does not prove what `api.glacis.io` does with it. The
    harness prints that as a NOT COVERED line rather than implying otherwise.

16. **The 129.1-second figure is arithmetic, not a measurement — and not a
    ceiling.** The constants, the backoff sequence and the four-attempt count
    are each executed; the number is computed from them. Round 4 corrected the
    framing: `timeout=30` is per-operation, and its read timeout bounds the gap
    between chunks rather than the response, so a slow-dripping endpoint can
    exceed 129.1s and **there is no default wall-clock bound at all**. Observing
    even the four-timeout case needs an endpoint that hangs and would add two
    minutes to every harness run, so the harness still names it NOT COVERED.

17. **The wrapper still cannot bound its own latency.** The honest workaround is
    on the page — go offline, or construct your own client — but the real fix is
    an SDK change: let the factories forward `timeout` / `max_retries`, or move
    attestation off the request path. Out of scope for this round, which was
    already stretching a documentation task into SDK source.

18. **`docs/dist` ships a `<link rel="icon" href="/favicon.svg">` for a file
    that does not exist.** Pre-existing, cosmetic, and untouched here: it comes
    from the Starlight default and `docs/public/` has no `favicon.svg`. Noted
    because a link check will keep reporting it.

### Added in round 4

19. **`Glacis.verify()` now succeeds on receipts it used to reject.** The old
    offline branch compared the receipt's `public_key` against the one derived
    from your own seed, so a receipt signed by anybody else came back
    `valid=False`. That is not what a signature check answers, and 0.8.1
    verifies under the key on the receipt — so a third party's good receipt now
    passes. Anyone who was reading `valid=False` as "not mine" was reading a
    coincidence, and the boundary page has always said a key is not an identity.
    Named here because it is a behaviour change in the permissive direction.

20. **The `cpr_hash` cross-check is deliberately not authoritative.** A receipt
    whose unsigned `cpr_hash` disagrees with its signed control-plane content
    still returns `valid=True`, with `error` naming `cpr_hash_mismatch`. Making
    an unsigned field able to fail a receipt would contradict the signed/unsigned
    boundary; leaving the inconsistency silent would hide it. So it is reported
    and does not decide. A caller that only reads `valid` will not see it.

21. **`StorageMigrationError` is a new failure mode for existing callers.**
    Opening a receipt store whose schema cannot be migrated now raises where it
    previously mislabelled the database and carried on. That is the point, but
    it means a corrupt store fails at `_get_connection()` rather than returning
    wrong rows — code that opened a store defensively should expect it.

22. **The docs now describe two SDK versions on every verification page.** Until
    0.8.1 ships, `pip install glacis` serves 0.8.0, whose `verify()` and CLI do
    not check signatures. Each page states both. That is honest and it is also
    clutter; the paragraphs collapse to one version the day a release goes out.
