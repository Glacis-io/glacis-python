#!/usr/bin/env python3
"""
Glacis Offline Mode Example

No API key required. Receipts are self-signed and marked "UNVERIFIED."
This is fully functional for development - upgrade to online mode
when you need third-party verifiability.
"""

from glacis import Glacis
import os

def main():
    # Create client in offline mode
    glacis = Glacis(mode="offline", signing_seed=os.urandom(32))

    # Simulate an AI interaction
    prompt = "What is the capital of France?"
    response = "The capital of France is Paris."

    # Create attestation
    # Note: input and output are hashed locally, never sent anywhere
    receipt = glacis.attest(
        service_id="example-app",
        operation_type="inference",
        input={"prompt": prompt},
        output={"response": response},
        metadata={
            "model": "gpt-4",
            "temperature": 0.7,
        },
    )

    print("Attestation created!")
    print(f"  Receipt ID: {receipt.id}")
    print(f"  Evidence hash: {receipt.evidence_hash}")
    print(f"  Witness status: {receipt.witness_status}")
    print()

    # Verify the receipt
    result = glacis.verify(receipt)
    print("Verification result:")
    print(f"  Signature valid: {result.signature_valid}")
    print(f"  Overall valid: {result.valid}")
    print()

    # Receipts are stored locally
    print("Receipts are stored locally (SQLite by default, JSONL optional)")
    print()
    print("To get witnessed attestations with Merkle proofs,")
    print("use online mode with an API key from https://glacis.io")


if __name__ == "__main__":
    main()
