# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.1.dev0] - Unreleased

Not published. `pip install glacis` still serves 0.8.0, which has the defect
described below.

### Fixed

- **Persisted offline receipts lost signed control-plane content.**
  `control_plane_results` is *inside* the offline signature —
  `Glacis._attest_offline` puts the whole structure into the payload it signs —
  but neither storage backend persisted it. A receipt that verified perfectly
  when returned no longer carried the content its own signature covered once it
  had been through `store_receipt()` / `get_receipt()`, so rebuilding the signed
  payload from a reloaded receipt produced different bytes and **independent
  Ed25519 verification failed**. The SDK's own offline `verify()` did not notice,
  because it only compares the locally derived public key and never checks a
  signature.

  Present in every release that had `control_plane_results`, up to and including
  0.8.0. Anyone who attached control-plane results and relied on the local store
  as their copy of record has receipts that a third party cannot verify. The
  content was never written, so it cannot be recovered from the store; the fix is
  forward-looking only.

  - SQLite: schema v5 adds `offline_receipts.control_plane_json`, written whole
    and never truncated, with a v4→v5 migration.
  - JSONL: the receipt line now carries `control_plane_results`.
  - Both backends reconstruct it on read.
  - **Rows written before this release stay honest rather than convenient.** A
    row with a `cpr_hash` but no stored content is *not* reconstructed as "this
    receipt had no control-plane results". `Attestation.cpr_recovery_error`
    carries the reason in words, and `Glacis.verify()` fails closed on such a
    receipt with that reason as its `error`. Receipts that genuinely had no
    control-plane results are unaffected and still verify.

- **Schema version never advanced past the first migration.** `version` is the
  primary key of `schema_version`, so `INSERT OR REPLACE` appended a row instead
  of replacing one, and the reader took the *first* row — the oldest version.
  Every migration re-ran on every connection open. The reader now takes
  `MAX(version)` and the writer collapses the table to a single row.

- **The wrapper published its receipt before storing evidence.**
  `attest_and_store()` called `set_last_receipt()` before `store_evidence()`, so
  a storage failure left `get_last_receipt()` handing back a receipt for a call
  the docs say produced none. The receipt is now published only after storage
  returns. When storage fails, `get_last_receipt()` keeps returning whatever it
  returned before — nothing, or the previous call's receipt — which is what the
  documentation describes. The attestation itself may still exist (an offline
  receipt row is written by `attest()`, an online one is on the server); what
  changed is that the wrapper no longer presents it as complete.

### Added

- `Attestation.cpr_recovery_error` — SDK convenience, never signed and never
  transmitted. Set only by storage reconstruction, to name a loss instead of
  hiding it.

## [0.5.0] - 2025-02-24

### Breaking Changes
- **Vocabulary rename**: `action` field → `if_detected` in all control configs
- **Action values**: `"pass"` → `"forward"`, `output_block_action`: `"suppress"` → `"block"`, `"flag"` → `"forward"`
- **Determination values**: `determination.action` is now `"forwarded"` / `"blocked"`
- **Config format**: v1.3 nested structure (`controls.input.pii_phi` instead of `controls.pii_phi`)
- **Streaming removed**: Deferred to future release
- **Redaction removed**: PII control now detects but does not rewrite text
- **Storage**: Added JSONL backend as alternative to SQLite (default unchanged)

### Added
- Word filter control (case-insensitive term matching, configurable per-stage)
- LLM Judge framework (`BaseJudge`, `JudgeRunner`, `JudgesConfig`)
- Sampling config with `l1_rate` / `l2_rate` (nested L2⊂L1)
- JSONL storage backend option (append-only `receipts.jsonl` + `evidence.jsonl`)
- `decompose()` for batch → per-item attestations with shared `operation_id`
- `should_review()` deterministic HMAC-SHA256 sampling gate
- Custom controls interface (`BaseControl`)
- Google Gemini integration (`attested_gemini()`)

### Fixed
- Removed unconditional `print()` in jailbreak control
- Added validation: `l2_rate` must be <= `l1_rate`
- Cleaned up dead code in `_canonicalize_value()`
- Fixed `get_blocking_control()` return type annotation
- Fixed control ID collision for same control on input+output stages
- Fixed `sample_probability` hardcoded to 0.0 — now uses actual sampling rate
- Fixed `verify.py` using `evidence_hash` instead of attestation `id`
- Fixed broken examples (`basic_offline.py`, `online_openai.py`)
- Various docstring and test fixes

## [0.3.0] - 2025-01-15

### Added
- Anthropic integration (`attested_anthropic()`)
- PII/PHI detection control
- Jailbreak detection control
- YAML configuration file support
- Control pipeline with staged input/output execution

## [0.2.0] - 2025-01-02

Initial public release.

### Features

- **Online attestation** with Merkle tree inclusion proofs and signed tree heads
- **Offline mode** with local Ed25519 signing (no API key required)
- **OpenAI integration** - auto-attesting wrapper for chat completions
- **Anthropic integration** - auto-attesting wrapper for messages
- **Streaming sessions** - chunk-by-chunk attestation for streaming responses
- **SQLite storage** - local receipt persistence at `~/.glacis/receipts.db`
- **Cross-runtime hashing** - RFC 8785 canonical JSON, compatible with Rust/TypeScript
- **CLI verification** - `python -m glacis verify receipt.json`

### Security

- Zero-egress design: only SHA-256 hashes are transmitted, never payloads
- Ed25519 signatures via PyNaCl (libsodium) or WASM runtime
- Offline receipts clearly marked as "UNVERIFIED"

### Notes

- Offline receipts show `witness_status: "UNVERIFIED"` - this is by design
- For witnessed attestations with Merkle proofs, use online mode with an API key
- Get your API key at https://glacis.io
