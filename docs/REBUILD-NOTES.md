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
the SDK in this repo. **77 checks, all passing** as of this commit
(53 from the original rebuild, 24 added in Corrections round 2 below).

```
pip install -e .          # from the repo root
pip install pyyaml pynacl
python docs/scripts/verify-doc-snippets.py
```

It covers: offline attestation, canonical-JSON hashing and its key-order
independence, `verify()` (including its gap), the published independent
verification routine across four tamper cases plus the `control_plane_results`
and `supersedes` variants, operation linking, `decompose()`, `supersedes`,
`should_review()` determinism, both storage backends, the two storage path
gotchas, the CLI on good and tampered receipts, `glacis.yaml` loading, the word
filter and `ControlsRunner`, constructor validation, the declared extras in
`pyproject.toml`, and the keyword signature of all four provider wrappers.

It cannot cover: witnessed (online) mode, or an end-to-end provider call. Those
need a live endpoint and paid keys. **Every page carrying such a snippet says so
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

6. **Nine SDK bugs are documented rather than fixed.** Items 1–8 in *Corrections*
   are defects in `glacis` 0.8.0, not in the prose. C1 is a documentation task
   and the SDK is published; changing behaviour under a released version number
   was out of scope. The highest-value fix is an `Ed25519Runtime.verify()` plus a
   real offline `verify()` — at which point three pages get shorter and happier.

7. **`pyproject.toml` still advertises `https://docs.glacis.io/sdk/python`.**
   Left alone deliberately: it is package metadata for a published release, and
   the redirect resolves it correctly. Worth updating to `https://docs.glacis.io/`
   at the next version bump.

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
