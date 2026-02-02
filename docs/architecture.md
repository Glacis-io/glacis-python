# GLACIS Python SDK Architecture

## Philosophy: Hash Locally, Prove Globally

The GLACIS SDK is built on a single core principle: **zero data egress**. Your AI inputs and outputs never leave your infrastructure. Instead, only cryptographic hashes are sent to the GLACIS transparency log, creating a tamper-evident audit trail without exposing sensitive data.

### The Problem

AI systems handling sensitive data (healthcare, finance, legal) face a compliance paradox:
- Regulations require audit trails proving what AI systems did
- But sending data to third-party logging services creates new compliance risks
- Traditional logging solutions force a choice: compliance OR privacy

### The Solution

GLACIS resolves this by separating **proof** from **data**:

```
Your Infrastructure          GLACIS Log
┌─────────────────────┐      ┌─────────────────────┐
│ Input + Output      │      │ Hash (64 chars)     │
│ (full data stays)   │  →   │ Timestamp           │
│                     │      │ Merkle Proof        │
│ Local Evidence DB   │      │ Witness Signature   │
└─────────────────────┘      └─────────────────────┘
```

## Core Concepts

### 1. RFC 8785 Canonical JSON

Before hashing, data is serialized using RFC 8785 canonical JSON:
- Keys sorted lexicographically
- No whitespace
- Deterministic number formatting

This ensures identical hashes across Python, TypeScript, and Rust runtimes:

```python
from glacis.crypto import canonical_json, hash_payload

# Key order doesn't matter - same hash
hash1 = hash_payload({"b": 2, "a": 1})
hash2 = hash_payload({"a": 1, "b": 2})
assert hash1 == hash2
```

### 2. Two Attestation Modes

**Online Mode** (server-witnessed):
- Hash is sent to `api.glacis.io`
- Server witnesses the hash and returns a signed receipt
- Hash is included in a Merkle tree for inclusion proofs
- Receipt includes `verify_url` for third-party verification

**Offline Mode** (locally-signed):
- No network calls required
- Signs attestations locally using Ed25519
- Receipts stored in local SQLite database
- Can be verified later without server connectivity

### 3. Control Plane (Optional)

Before attestation, the SDK can run configurable controls:

**PII/PHI Redaction**:
- Detects and redacts sensitive data (SSN, email, phone, medical terms)
- Two modes: `fast` (regex) or `full` (Presidio NER models)
- Redacted text format: `[ENTITY_TYPE]` (e.g., `[US_SSN]`)

**Jailbreak Detection**:
- Scores prompts for potential prompt injection attacks
- Configurable threshold (0.0-1.0)
- Can warn or block based on score

### 4. Evidence Storage

Full payloads are stored locally in SQLite (`~/.glacis/receipts.db`):
- Never sent to GLACIS servers
- Queryable by attestation_id, service_id, time range
- Includes control plane results for audit

## Code Architecture

```
glacis/
├── __init__.py          # Public API exports
├── client.py            # Glacis/AsyncGlacis clients
├── crypto.py            # RFC 8785 + SHA-256 + Ed25519
├── models.py            # Pydantic models (receipts, config)
├── storage.py           # SQLite evidence storage
├── config.py            # YAML config loading
├── verify.py            # CLI verification tool
├── streaming.py         # Streaming session support
├── controls/            # Control plane modules
│   ├── base.py          # BaseControl, ControlResult
│   ├── pii.py           # PIIControl (Presidio)
│   └── jailbreak.py     # JailbreakControl (PromptGuard)
└── integrations/        # Provider wrappers
    ├── base.py          # Shared integration code
    ├── openai.py        # attested_openai()
    └── anthropic.py     # attested_anthropic()
```

## Data Flow Summary

1. **User calls** `client.chat.completions.create(messages=[...])`
2. **Controls run** (if enabled): PII redacted, jailbreak checked
3. **Payload hashed** using RFC 8785 canonical JSON + SHA-256
4. **Hash attested**:
   - Online: POST to `api.glacis.io/v1/attest`
   - Offline: Sign with Ed25519, store locally
5. **Evidence stored** locally in SQLite
6. **Receipt returned** with attestation_id, verify_url
7. **Original request** forwarded to AI provider
8. **Response** processed and returned to user

## Key Files

| File | Purpose |
|------|---------|
| `crypto.py` | Hash-local logic (RFC 8785 + Ed25519) |
| `client.py` | Prove-global logic (API calls, verification) |
| `storage.py` | Local evidence retention |
| `controls/` | Pre-attestation safety controls |
| `integrations/` | Drop-in provider wrappers |

## Security Properties

- **Tamper Evidence**: Merkle tree inclusion proofs
- **Non-Repudiation**: Server or Ed25519 signatures
- **Data Sovereignty**: Full payloads never leave your infrastructure
- **Auditability**: Local evidence + global proofs = complete audit trail
