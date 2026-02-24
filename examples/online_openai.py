#!/usr/bin/env python3
"""
Glacis + OpenAI Auto-Attestation Example

Every API call is automatically attested with Merkle proofs.
Requires: pip install glacis[openai]
"""

import os

from glacis.integrations.openai import attested_openai, get_last_receipt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    # Get API keys from environment
    glacis_api_key = os.environ.get("GLACIS_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not glacis_api_key:
        print("Set GLACIS_API_KEY environment variable")
        print("Get your key at https://glacis.io")
        return

    if not openai_api_key:
        print("Set OPENAI_API_KEY environment variable")
        return

    # Create attested OpenAI client
    client = attested_openai(
        glacis_api_key=glacis_api_key,
        openai_api_key=openai_api_key,
    )

    # Make a normal OpenAI call - attestation happens automatically
    print("Calling OpenAI...")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
    )

    print(f"Response: {response.choices[0].message.content}")
    print()

    # Get the attestation receipt
    receipt = get_last_receipt()

    print("Attestation:")
    print(f"  Receipt ID: {receipt.id}")
    print(f"  Evidence hash: {receipt.evidence_hash}")
    print(f"  Witness status: {receipt.witness_status}")
    print()
    print("Use receipt.id for third-party verification!")


if __name__ == "__main__":
    main()
