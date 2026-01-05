#!/usr/bin/env python3
"""
Glacis + Azure AI Inference Auto-Attestation Example

Every API call is automatically attested with Merkle proofs.
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
    # Get credentials from environment
    glacis_api_key = os.environ.get("GLACIS_API_KEY")
    azure_endpoint = os.environ.get("AZURE_AI_ENDPOINT")
    azure_api_key = os.environ.get("AZURE_AI_API_KEY")
    azure_model = os.environ.get("AZURE_AI_MODEL", "gpt-4o")

    if not glacis_api_key:
        print("Set GLACIS_API_KEY environment variable")
        print("Get your key at https://glacis.io")
        return

    if not azure_endpoint:
        print("Set AZURE_AI_ENDPOINT environment variable")
        print("Example: https://your-project.services.ai.azure.com/models")
        return

    if not azure_api_key:
        print("Set AZURE_AI_API_KEY environment variable")
        return

    # Create attested Azure AI client
    client = attested_azure_inference(
        glacis_api_key=glacis_api_key,
        endpoint=azure_endpoint,
        credential=AzureKeyCredential(azure_api_key),
        service_id="azure-ai",
        debug=True,  # Show attestation errors
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
        print("Attestation:")
        print(f"  Receipt ID: {receipt.attestation_id}")
        print(f"  Leaf index: {receipt.leaf_index}")
        print(f"  Merkle root: {receipt.signed_tree_head.root_hash[:16]}...")
        print(f"  Badge URL: {receipt.badge_url}")
        print()
        print("Share the badge URL for third-party verification!")
    else:
        print("Attestation failed - check debug output above.")
        print("Verify your GLACIS_API_KEY is valid.")


if __name__ == "__main__":
    main()