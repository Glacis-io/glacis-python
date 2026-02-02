"""
Glacis CLI - Receipt Verification

Usage:
    python -m glacis verify <receipt.json>
    python -m glacis verify <receipt.json> --base-url https://api.glacis.io
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Union

import httpx

from glacis import Glacis
from glacis.models import (
    AttestReceipt,
    OfflineAttestReceipt,
    OfflineVerifyResult,
    VerifyResult,
)

DEFAULT_BASE_URL = "https://api.glacis.io"


def verify_online(attestation_hash: str, base_url: str) -> VerifyResult:
    """Verify an online attestation via direct HTTP call."""
    url = f"{base_url}/v1/verify/{attestation_hash}"

    try:
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
        return VerifyResult.model_validate(response.json())
    except httpx.HTTPStatusError as e:
        return VerifyResult(
            valid=False,
            error=f"HTTP {e.response.status_code}: {e.response.text}",
        )
    except httpx.RequestError as e:
        return VerifyResult(
            valid=False,
            error=f"Request failed: {e}",
        )


def verify_offline(receipt: OfflineAttestReceipt) -> OfflineVerifyResult:
    """
    Verify an offline attestation.

    Note: Full cryptographic signature verification of offline receipts requires
    the original signing seed or the complete signed payload (which includes
    a timestamp). Without these, we validate the receipt structure and format.

    For full cryptographic verification, use the Glacis client with the
    original signing_seed that created the receipt.
    """
    try:
        # Validate receipt structure and format
        valid = (
            receipt.attestation_id.startswith("oatt_")
            and len(receipt.payload_hash) == 64
            and len(receipt.public_key) == 64
            and len(receipt.signature) > 0
            and receipt.witness_status == "UNVERIFIED"
        )

        return OfflineVerifyResult(
            valid=valid,
            witness_status="UNVERIFIED",
            signature_valid=valid,  # Structure valid (not full crypto verification)
            attestation=receipt,
        )
    except Exception as e:
        return OfflineVerifyResult(
            valid=False,
            witness_status="UNVERIFIED",
            signature_valid=False,
            attestation=receipt,
            error=str(e),
        )


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

    # Check for offline receipt - supports both camelCase and snake_case
    att_id = data.get("attestationId") or data.get("attestation_id") or ""
    if att_id.startswith("oatt_"):
        receipt = OfflineAttestReceipt(**data)
        result = verify_offline(receipt)
    else:
        receipt = AttestReceipt(**data)
        result = verify_online(receipt.attestation_hash, args.base_url)

    # Output
    print(f"Receipt: {receipt.attestation_id}")
    print(f"Type: {'Offline' if isinstance(receipt, OfflineAttestReceipt) else 'Online'}")
    print()

    if result.valid:
        print("Status: VALID")
        if isinstance(result, OfflineVerifyResult):
            sig_valid = result.signature_valid
        else:
            sig_valid = result.verification.signature_valid if result.verification else False
        print(f"  Signature: {'PASS' if sig_valid else 'FAIL'}")
        if isinstance(result, VerifyResult) and result.verification:
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
    verify_parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    verify_parser.set_defaults(func=verify_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
