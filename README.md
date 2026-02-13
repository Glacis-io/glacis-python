<p align="center">
  <img src="assets/glacis-logo.png" alt="Glacis" width="200">
</p>

# Glacis Python SDK

**Tamper-proof audit logs for AI systems - without exposing sensitive data.**

## The Problem

You need to prove what your AI did for compliance, audits, or legal discovery. But sending prompts and responses to a logging service exposes sensitive data (PII, PHI, trade secrets).

## The Solution

Glacis creates cryptographic proofs of AI operations. Your data stays local - only a SHA-256 hash is sent for witnessing.

```
Your Infrastructure              Glacis Log
┌─────────────────────┐         ┌─────────────────────┐
│ "Pt. Frodo Baggins  │         │ 7a3f8b2c...         │
│  has diabetes"      │  ──→    │ (64-char hash)      │
│                     │         │ + timestamp         │
│ (data stays here)   │         │ + Merkle proof      │
└─────────────────────┘         └─────────────────────┘
```

Later, you can prove the hash matches your local records without revealing the data itself.

## Installation

```bash
pip install glacis[openai]      # For OpenAI
pip install glacis[anthropic]   # For Anthropic
pip install glacis[gemini]      # For Google Gemini
pip install glacis[controls]    # Add PII redaction + jailbreak detection
pip install glacis[all]         # Everything
```

## Quick Start

### Option 1: Drop-in Wrapper (Recommended)

Replace your OpenAI/Anthropic/Gemini client with a wrapped version. Every API call is automatically attested.

```python
import os
from glacis.integrations.openai import attested_openai, get_last_receipt

# Create wrapped client (offline mode - no Glacis account needed)
client = attested_openai(
    openai_api_key="sk-...",
    offline=True,
    signing_seed=os.urandom(32),
)

# Use exactly like the normal OpenAI client
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Get the attestation receipt
receipt = get_last_receipt()
print(f"Attestation ID: {receipt.id}")
```

Works the same for Anthropic:

```python
from glacis.integrations.anthropic import attested_anthropic, get_last_receipt

client = attested_anthropic(
    anthropic_api_key="sk-ant-...",
    offline=True,
    signing_seed=os.urandom(32),
)
```

And for Google Gemini:

```python
from glacis.integrations.gemini import attested_gemini, get_last_receipt

client = attested_gemini(
    gemini_api_key="...",
    offline=True,
    signing_seed=os.urandom(32),
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello!"
)

receipt = get_last_receipt()
```

### Option 2: Direct API

For custom attestations (non-OpenAI/Anthropic/Gemini, or manual control):

```python
import os
from glacis import Glacis

glacis = Glacis(mode="offline", signing_seed=os.urandom(32))

receipt = glacis.attest(
    service_id="my-ai-app",
    operation_type="inference",
    input={"prompt": "Summarize this..."},
    output={"response": "The document..."},
)
```

## Adding PII Redaction

Automatically detect and redact sensitive data before it's hashed:

```python
client = attested_openai(
    openai_api_key="sk-...",
    offline=True,
    signing_seed=os.urandom(32),
    redaction="fast",  # Regex-based, or "full" for ML models
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "My SSN is 123-45-6789"}]
)

# The attestation hash is computed on: "My SSN is [US_SSN]"
# Original text still sent to OpenAI, but redacted version is attested
```

## Configuration File

For persistent settings, create `glacis.yaml`:

```yaml
attestation:
  offline: true
  service_id: my-ai-service

controls:
  pii_phi:
    enabled: true
    mode: fast          # "fast" (regex) or "full" (Presidio NER)

  jailbreak:
    enabled: true
    threshold: 0.5      # Block prompt injection attempts
    action: block       # "warn" or "block"
```

Then:

```python
client = attested_openai(
    openai_api_key="sk-...",
    config_path="glacis.yaml",
)
```

## Retrieving Evidence

Full payloads are stored locally for audits:

```python
from glacis.integrations.openai import get_last_receipt, get_evidence

receipt = get_last_receipt()
evidence = get_evidence(receipt.id)

print(evidence["input"])                  # Original input
print(evidence["output"])                 # Original output
print(evidence["control_plane_results"])  # PII/jailbreak results
```

Evidence is stored in `~/.glacis/receipts.db` (SQLite).

## Online vs Offline Mode

| Feature | Offline | Online |
|---------|---------|--------|
| Requires Glacis account | No | Yes |
| Signing | Local Ed25519 | Glacis witness |
| Third-party verifiable | No | Yes (Merkle proofs) |
| Use case | Development, air-gapped | Production, audits |

To use online mode:

```python
client = attested_openai(
    openai_api_key="sk-...",
    glacis_api_key="glsk_live_...",  # Get at glacis.io
)
```

## What Gets Sent to Glacis?

| Data | Sent? |
|------|-------|
| Your prompts | No (hash only) |
| Model responses | No (hash only) |
| API keys | No |
| service_id, operation_type | Yes |
| Timestamps | Yes |

## CLI

Verify a receipt:

```bash
python -m glacis verify receipt.json
```

## Security

- **Hashing**: SHA-256 with RFC 8785 canonical JSON (cross-runtime compatible)
- **Signing**: Ed25519 via PyNaCl (libsodium)
- **Online mode**: Merkle tree inclusion proofs (RFC 6962)

## License

Apache 2.0
