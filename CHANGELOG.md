# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
