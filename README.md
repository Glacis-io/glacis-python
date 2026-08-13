<p align="center">
  <img src="https://raw.githubusercontent.com/Glacis-io/glacis-python/main/assets/glacis-logo.png" alt="Glacis" width="200">
</p>

# Glacis Python SDK

**Tamper-proof audit logs for AI systems - without exposing sensitive data.**

Two modes ship today:

- **Hosted (server-attested)** - every attestation is minted into the Glacis
  transparency log and comes back with an RFC 6962 inclusion proof and a
  signed tree head, which the SDK verifies locally before labeling anything.
- **Offline (self-signed)** - local Ed25519 signing, no account needed.

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
pip install glacis[controls]    # Add PII detection + jailbreak detection
pip install glacis[all]         # Everything
```

## Quick Start

### Option 1: Hosted (server-attested) receipts

```bash
pip install glacis
export GLACIS_API_KEY=glsk_live_...           # from glacis.io
export GLACIS_LOG_PUBLIC_KEY_HEX=...          # the Glacis log key, published at launch
```

```python
from glacis import Glacis

glacis = Glacis(mode="hosted")
artifact = glacis.attest(service_id="my-ai-app", operation_type="inference",
                         input={"prompt": "..."}, output={"response": "..."})
artifact.save("receipt.json")  # paste the file at glacis.io/verify
```

`artifact.witness_status` tells you exactly what you hold:

- `WITNESSED` - the SDK recomputed the RFC 6962 inclusion proof from the
  receipt's own identifier to the signed tree head, and the tree head's
  Ed25519 signature verified under the log key you configured. This is what
  "server-attested" means here: the receipt is logged, and its inclusion is
  verified under the Glacis log. The tree head also carries a witness
  countersignature; that countersigner is not independently attested yet, so
  the SDK does not count it as verification.
- `LOGGED_UNVERIFIED` - the mint succeeded but the SDK could not verify the
  record (most commonly: no `GLACIS_LOG_PUBLIC_KEY_HEX` configured). The
  reason is in `artifact.verification.reason`. Never silently upgraded.
- `SELF_SIGNED` - offline receipts. Your own key, your own word.

The SDK ships no baked-in log key: verification only happens under a key you
configure, so a receipt can never vouch for itself.

### Option 2: Drop-in Wrapper

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

### Option 3: Direct API (offline)

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

## Adding Controls

Detect PII/PHI and prompt injection attempts in your AI calls. Enable controls via a YAML config file:

```python
client = attested_openai(
    openai_api_key="sk-...",
    offline=True,
    signing_seed=os.urandom(32),
    config_path="glacis.yaml",  # Enable controls via config
)
```

Control results (detections, scores, latencies) are included in the attestation record.

## Configuration File

For persistent settings, create `glacis.yaml`:

```yaml
version: "1.3"

attestation:
  offline: true
  service_id: my-ai-service

controls:
  input:
    pii_phi:
      enabled: true
      mode: fast            # "fast" (regex) or "full" (Presidio NER)
      if_detected: flag     # "forward", "flag", or "block"

    jailbreak:
      enabled: true
      threshold: 0.5
      if_detected: block

sampling:
  l1_rate: 1.0   # Evidence collection rate (0.0-1.0)
  l2_rate: 0.0   # Deep inspection rate (must be <= l1_rate)
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

Evidence is stored locally using SQLite (default) or JSONL backends.

## Hosted vs Offline Mode

| Feature | Offline | Hosted |
|---------|---------|--------|
| Requires Glacis account | No | Yes (`GLACIS_API_KEY`) |
| Signing | Local Ed25519 | Local Ed25519 + transparency-log inclusion |
| Third-party verifiable | Signature only | Yes (RFC 6962 proof at glacis.io/verify) |
| witness_status | `SELF_SIGNED` | `WITNESSED` after local verification |
| Use case | Development | Audits, regulatory |

Hosted mode never sends payload text - the gateway receives only a task-class
label and `request_sha256`, the SHA-256 of the attestation's signed bytes.

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
