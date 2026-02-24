# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-02-24

### Breaking Changes
- **Vocabulary rename**: `action` field → `if_detected` in all control configs
- **Action values**: `"pass"` → `"forward"`, `output_block_action`: `"suppress"` → `"block"`, `"flag"` → `"forward"`
- **Determination values**: `determination.action` is now `"forwarded"` / `"blocked"`
- **Config format**: v1.3 nested structure (`controls.input.pii_phi` instead of `controls.pii_phi`)
- **Streaming removed**: Deferred to future release
- **Redaction removed**: PII control now detects but does not rewrite text
- **Storage format**: JSONL (`receipts.jsonl` + `evidence.jsonl`) replaces SQLite

### Added
- Word filter control (case-insensitive term matching, configurable per-stage)
- LLM Judge framework (`BaseJudge`, `JudgeRunner`, `JudgesConfig`)
- Sampling config with `l1_rate` / `l2_rate` (nested L2⊂L1)
- JSONL evidence storage (append-only `receipts.jsonl` + `evidence.jsonl`)
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
