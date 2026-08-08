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
the SDK in this repo. **53 checks, all passing** as of this commit.

```
pip install -e .          # from the repo root
pip install pyyaml
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

9. **The browser verifier does not read SDK offline receipts.** `glacis.io/verify`
   detects `v2`, `v1-gateway` and `v1-scanner` receipt shapes; a flat `oatt_…`
   receipt matches none of them and returns "Unrecognized receipt format".
   Telling readers to paste an SDK receipt there would have been a false
   instruction, so `/verify/` says plainly which receipts it reads and points
   SDK users at the Python routine instead.

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
| `python docs/scripts/verify-doc-snippets.py` | 53/53 checks pass |
| Internal link check over `dist/` | 42 pages, **0 broken links** |
| Cross-page anchor check over `dist/` | **0 missing anchors** |
| `grep -rniE "request[[:space:]]+access"` | **0 occurrences** |
| `grep -rniE "compliant\|protected\|certified"` | Only inside explicit negations |

Live surfaces probed read-only (GET) on 2026-08-08 to keep claims true:
`app.glacis.dev/login` (200, carries *Create account*), `glacis.io/verify` (200,
client-side verifier, currently a `?receipt=` share link), `overt.is` (200,
OVERT 1.1), `docs.glacis.io` (200), `api.glacis.io/v1/root` (404 JSON).

---

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
