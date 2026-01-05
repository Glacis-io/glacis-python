#!/usr/bin/env python3
"""
Glacis + Azure AI Inference Offline Attestation Example

No Glacis API key required. Receipts are self-signed locally.
Requires: pip install glacis[azure] python-dotenv

Supports any model deployed on Azure AI Foundry (Llama, DeepSeek, Mistral, etc.)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from glacis.integrations.azure import attested_azure_inference, get_last_receipt

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")


def main():
    # Get Azure credentials from environment
    azure_endpoint = os.environ.get("AZURE_AI_ENDPOINT")
    azure_api_key = os.environ.get("AZURE_AI_API_KEY")
    azure_model = os.environ.get("AZURE_AI_MODEL", "gpt-4o")

    if not azure_endpoint:
        print("Set AZURE_AI_ENDPOINT environment variable")
        print("Example: https://your-project.services.ai.azure.com/models")
        return

    if not azure_api_key:
        print("Set AZURE_AI_API_KEY environment variable")
        return

    # Generate a signing seed (in production, persist this securely)
    signing_seed = os.urandom(32)

    # Create attested Azure AI client in OFFLINE mode
    client = attested_azure_inference(
        endpoint=azure_endpoint,
        credential=AzureKeyCredential(azure_api_key),
        offline=True,
        signing_seed=signing_seed,
        service_id="azure-ai",
        debug=True,
    )

    # Make a normal Azure AI call - attestation happens automatically
    print(f"Calling Azure AI ({azure_model})...")
    response = client.complete(
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is 2 + 2?"),
        ],
        model=azure_model,
        max_tokens=100,
    )

    print(f"Response: {response.choices[0].message.content}")
    print()

    # Get the attestation receipt
    receipt = get_last_receipt()

    if receipt:
        print("Attestation (offline):")
        print(f"  Receipt ID: {receipt.attestation_id}")
        print(f"  Payload Hash: {receipt.payload_hash}")
        print(f"  Signature: {receipt.signature[:40]}...")
        print(f"  Witness Status: {receipt.witness_status}")
        print()
        print("Receipts stored in ~/.glacis/receipts.db")
        print("Upgrade to online mode for third-party verifiability.")
    else:
        print("Attestation failed - check debug output above.")


if __name__ == "__main__":
    main()
