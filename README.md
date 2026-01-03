# Glacis Python SDK

[![PyPI version](https://badge.fury.io/py/glacis.svg)](https://badge.fury.io/py/glacis)
[![Tests](https://github.com/Glacis-io/glacis-python/actions/workflows/test.yml/badge.svg)](https://github.com/Glacis-io/glacis-python/actions/workflows/test.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Cryptographic attestation for AI systems.** Prove what your AI did, what data it saw, and what controls were active - without sensitive data leaving your environment.

## Installation

```bash
pip install glacis
```

With provider integrations:
```bash
pip install glacis[openai]      # OpenAI auto-attestation
pip install glacis[anthropic]   # Anthropic auto-attestation
pip install glacis[all]         # Everything
```

## Quick Start

### Offline Mode (No API Key Required)

Works immediately. Receipts are self-signed and marked "UNVERIFIED."

```python
from glacis import Glacis

glacis = Glacis(mode="offline")

# Attest an AI interaction
receipt = glacis.attest(
    service_id="my-ai-app",
    operation_type="inference",
    input={"prompt": "Summarize this document..."},   # Hashed locally
    output={"response": "The document discusses..."}, # Never sent
)

print(f"Receipt ID: {receipt.attestation_id}")
print(f"Witness status: {receipt.witness_status}")  # "UNVERIFIED"

# Verify locally
result = glacis.verify(receipt)
print(f"Signature valid: {result.signature_valid}")
```

### Online Mode (Witnessed Attestation)

Add an API key for cryptographically witnessed receipts with Merkle proofs.

```python
from glacis import Glacis

glacis = Glacis(api_key="glsk_live_...")  # Get yours at glacis.io

receipt = glacis.attest(
    service_id="my-ai-app",
    operation_type="inference",
    input={"prompt": "..."},
    output={"response": "..."},
)

print(f"Leaf index: {receipt.leaf_index}")
print(f"Merkle root: {receipt.signed_tree_head.root_hash}")
print(f"Badge URL: {receipt.badge_url}")  # Shareable verification link
```

### Auto-Attesting OpenAI

```python
from glacis.integrations.openai import attested_openai, get_last_receipt

client = attested_openai(
    glacis_api_key="glsk_live_...",
    openai_api_key="sk-..."
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Receipt is automatically created
receipt = get_last_receipt()
print(f"Attested: {receipt.badge_url}")
```

### Auto-Attesting Anthropic

```python
from glacis.integrations.anthropic import attested_anthropic, get_last_receipt

client = attested_anthropic(
    glacis_api_key="glsk_live_...",
    anthropic_api_key="sk-..."
)

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

receipt = get_last_receipt()
```

## What Data Leaves Your Environment?

**Only hashes. Never payloads.**

| What | Sent to Glacis? |
|------|-----------------|
| Your prompts | No - SHA-256 hash only |
| Model responses | No - SHA-256 hash only |
| API keys | No |
| Metadata (service_id, operation_type) | Yes |
| Timestamps | Yes |

This is the "zero-egress" design. Your sensitive data stays local; only cryptographic commitments are transmitted for witnessing.

## Offline vs Online Mode

| Feature | Offline | Online |
|---------|---------|--------|
| API key required | No | Yes |
| Signing | Local Ed25519 | Glacis witness network |
| Merkle proofs | No | Yes |
| Transparency log | No | Yes |
| Verification URL | No | Yes |
| Witness status | "UNVERIFIED" | "VERIFIED" |

**Offline mode is fully functional** - correct crypto, local verification, production-grade Ed25519. The only difference is the absence of an independent witness anchor.

Use offline for development. Upgrade to online when you need third-party verifiability (audits, papers, customer due diligence).

## CLI

Verify a receipt:
```bash
python -m glacis verify receipt.json
```

## Async Support

```python
from glacis import AsyncGlacis

async with AsyncGlacis(api_key="glsk_live_...") as glacis:
    receipt = await glacis.attest(
        service_id="my-service",
        operation_type="inference",
        input={"prompt": "Hello"},
        output={"response": "Hi!"},
    )
```

## Query the Transparency Log

```python
# Browse public attestations (no auth required)
result = glacis.query_log(
    org_id="org_xxx",
    service_id="my-service",
    start="2024-01-01T00:00:00Z",
    limit=100,
)

for entry in result.entries:
    print(f"{entry.timestamp}: {entry.operation_type}")

# Get current tree state
tree = glacis.get_tree_head()
print(f"Tree size: {tree.size}")
print(f"Root hash: {tree.root_hash}")
```

## Cross-Runtime Hash Compatibility

The Python SDK produces identical hashes to the TypeScript and Rust SDKs:

```python
from glacis.crypto import hash_payload

# These all produce the same hash across Python, TypeScript, and Rust
hash1 = hash_payload({"b": 2, "a": 1})
hash2 = hash_payload({"a": 1, "b": 2})
assert hash1 == hash2  # Keys are sorted per RFC 8785
```

## Security & Trust

- **Cryptography**: Ed25519 signatures via PyNaCl (libsodium) or WASM
- **Hashing**: SHA-256 with RFC 8785 canonical JSON (cross-runtime compatible)
- **Transparency**: Online receipts are included in an append-only Merkle tree (RFC 6962)

### Threat Model

Glacis provides cryptographic evidence that a specific operation occurred on a specific input/output pair at a specific time. It does not:
- Prevent AI from misbehaving (it attests, not enforces)
- Hide that an AI system exists (receipts are evidence of operation)
- Guarantee the AI output was correct (only that it was attested)

### Security Disclosure

Report vulnerabilities to security@glacis.io

## API Reference

### Glacis

```python
from glacis import Glacis

# Online mode (default)
glacis = Glacis(api_key="glsk_live_...")

# Offline mode
glacis = Glacis(mode="offline")
glacis = Glacis(mode="offline", signing_seed=os.urandom(32))
```

### glacis.attest()

```python
receipt = glacis.attest(
    service_id="my-service",           # Required
    operation_type="inference",         # Required
    input={"prompt": "..."},           # Hashed, not sent
    output={"response": "..."},        # Hashed, not sent
    metadata={"model": "gpt-4"},       # Optional, sent as-is
)
```

### glacis.verify()

```python
result = glacis.verify(receipt)
print(result.valid)            # Overall validity
print(result.signature_valid)  # Signature check
print(result.proof_valid)      # Merkle proof check (online only)
```

### AttestReceipt

| Field | Description |
|-------|-------------|
| `attestation_id` | Unique attestation ID |
| `timestamp` | ISO 8601 timestamp |
| `leaf_index` | Merkle tree leaf index |
| `merkle_proof` | Inclusion proof |
| `signed_tree_head` | Signed tree state |
| `badge_url` | Verification badge URL |
| `verify_url` | Verification endpoint |

## Pricing

| Tier | Price | Includes |
|------|-------|----------|
| Offline | Free | Local signing, "UNVERIFIED" status |
| Witnessed | Free, then usage-based | Independent witness, Merkle proofs, verification URLs |
| Enterprise | Custom | SLA, compliance exports, dedicated support |

Get started at [docs.glacis.io](https://docs.glacis.io/sdk/python/quickstart)

## License

Apache 2.0. See [LICENSE](LICENSE).

## Links

- [Documentation](https://docs.glacis.io)
- [Changelog](CHANGELOG.md)
