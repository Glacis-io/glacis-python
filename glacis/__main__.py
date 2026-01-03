"""
Glacis CLI

Usage:
    python -m glacis verify <receipt.json>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Union

from glacis import Glacis
from glacis.models import AttestReceipt, OfflineAttestReceipt, OfflineVerifyResult, VerifyResult


def verify_command(args: argparse.Namespace) -> None:
    """Verify a receipt file."""
    receipt_path = Path(args.receipt)

    if not receipt_path.exists():
        print(f"Error: File not found: {receipt_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(receipt_path) as f:
            data: dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine receipt type and verify
    receipt: Union[AttestReceipt, OfflineAttestReceipt]
    result: Union[VerifyResult, OfflineVerifyResult]

    if data.get("attestation_id", "").startswith("oatt_"):
        offline_receipt = OfflineAttestReceipt(**data)
        # For verification, we need a signing_seed but it's not used
        # since we verify using the public key from the receipt
        dummy_seed = bytes(32)  # All zeros - only used to satisfy constructor
        glacis = Glacis(mode="offline", signing_seed=dummy_seed)
        result = glacis.verify(offline_receipt)
        receipt = offline_receipt
    else:
        online_receipt = AttestReceipt(**data)
        # Online verification doesn't require API key for public receipts
        glacis = Glacis(api_key="verify_only")
        result = glacis.verify(online_receipt)
        receipt = online_receipt

    # Output
    print(f"Receipt: {receipt.attestation_id}")
    print(f"Type: {'Offline' if isinstance(receipt, OfflineAttestReceipt) else 'Online'}")
    print()

    if result.valid:
        print("Status: VALID")
        # Get signature validity based on result type
        if isinstance(result, OfflineVerifyResult):
            sig_valid = result.signature_valid
        else:
            sig_valid = result.verification.signature_valid
        print(f"  Signature: {'PASS' if sig_valid else 'FAIL'}")
        if isinstance(result, VerifyResult):
            print(f"  Merkle proof: {'PASS' if result.verification.proof_valid else 'FAIL'}")
    else:
        print("Status: INVALID")
        if result.error:
            print(f"  Error: {result.error}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="glacis", description="Glacis CLI - Cryptographic attestation for AI systems"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a receipt")
    verify_parser.add_argument("receipt", help="Path to receipt JSON file")
    verify_parser.set_defaults(func=verify_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
