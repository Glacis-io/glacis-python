"""
Glacis receipt verification — the CLI, and the offline check behind it.

Usage:
    python -m glacis verify <receipt.json>
    python -m glacis verify <receipt.json> --base-url https://api.glacis.io

``verify_offline()`` here is the one offline verification in the SDK:
``Glacis.verify()`` calls it too, so the library and the command line can never
give different answers about the same receipt.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Union

import httpx

from glacis.crypto import (
    CryptoError,
    get_ed25519_runtime,
    hash_payload,
    offline_signed_payload_for,
)
from glacis.models import (
    Attestation,
    OfflineVerifyResult,
    VerifyResult,
)

DEFAULT_BASE_URL = "https://api.glacis.io"


def verify_online(attestation_id: str, base_url: str) -> VerifyResult:
    """Verify an online attestation via direct HTTP call."""
    url = f"{base_url}/v1/verify/{attestation_id}"

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


def _cpr_degradation(receipt: Attestation) -> Optional[str]:
    """Name a control-plane inconsistency, if there is one.

    This is a fast SHA-256 cross-check, and it is *not* the authority.
    ``cpr_hash`` is **outside** the signature — a receipt with an edited
    ``cpr_hash`` still verifies, because the signature covers the
    control-plane *content*, not this convenience hash of it. So a mismatch
    here never decides ``valid``; it is reported by name alongside whatever
    the signature says, because a receipt that disagrees with itself is worth
    telling the caller about either way.
    """
    if not receipt.cpr_hash:
        return None

    if receipt.control_plane_results is None:
        return (
            "cpr_hash_orphaned: the receipt carries a cpr_hash but no "
            "control_plane_results. The content is inside the signature, so "
            "the signed payload cannot be rebuilt from this receipt."
        )

    recomputed = hash_payload(receipt.control_plane_results)
    if recomputed != receipt.cpr_hash:
        return (
            "cpr_hash_mismatch: the receipt's cpr_hash is not a SHA-256 over "
            "the control_plane_results it carries. cpr_hash is unsigned, so "
            "the signature — not this hash — decides whether the receipt is "
            f"intact (expected {recomputed}, receipt says {receipt.cpr_hash})."
        )

    return None


def verify_offline(receipt: Attestation) -> OfflineVerifyResult:
    """Verify an offline receipt's Ed25519 signature.

    This is a real cryptographic check: the payload is rebuilt from the
    receipt's own signed fields (``glacis.crypto.offline_signed_payload``) and
    the signature is verified against it under the public key **on the
    receipt**. No signing seed is needed — a third party holding nothing but
    the receipt runs exactly this check.

    Tampering with any signed field — ``service_id``, ``evidence_hash``, the
    timestamp, or the ``control_plane_results`` content, including losing it in
    storage — makes the check fail.

    What it does *not* establish is who holds the key. Verifying against a
    public key found in the same document establishes internal consistency and
    nothing more. See /verify/what-a-check-proves/.

    ``error`` names what happened:

    * ``cpr_unrecoverable`` — the receipt came back from a store that could not
      return its signed control-plane content. No check is possible.
    * ``structural`` — the receipt cannot be turned into verifiable form at all
      (undecodable key or signature, no timestamp).
    * ``signature_invalid`` — the payload rebuilt, and the signature does not
      verify over it.
    * ``cpr_hash_mismatch`` / ``cpr_hash_orphaned`` — a named degradation of an
      unsigned field. It can accompany ``valid=True``, because the signature is
      the authority.
    """

    def failed(reason: str) -> OfflineVerifyResult:
        return OfflineVerifyResult(
            valid=False,
            witness_status="UNVERIFIED",
            signature_valid=False,
            attestation=receipt,
            error=reason,
        )

    # A store that dropped signed control-plane content leaves nothing to
    # check: the signed bytes cannot be rebuilt by us or by anyone else.
    if receipt.cpr_recovery_error:
        return failed(f"cpr_unrecoverable: {receipt.cpr_recovery_error}")

    degradation = _cpr_degradation(receipt)

    try:
        message = offline_signed_payload_for(receipt)
    except (ValueError, AttributeError) as e:
        return failed(f"structural: {e}")

    try:
        signature_valid = get_ed25519_runtime().verify(
            receipt.public_key, message, receipt.signature
        )
    except CryptoError as e:
        return failed(f"structural: {e}")

    if not signature_valid:
        reason = (
            "signature_invalid: the Ed25519 signature does not verify over the "
            "payload rebuilt from this receipt's signed fields, under the "
            "public_key the receipt carries. A signed field has been altered "
            "since signing, or the signature does not belong to this receipt."
        )
        if degradation:
            reason = f"{reason} Also: {degradation}"
        return failed(reason)

    return OfflineVerifyResult(
        valid=True,
        witness_status="UNVERIFIED",
        signature_valid=True,
        attestation=receipt,
        error=degradation,
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
    result: Union[VerifyResult, OfflineVerifyResult]

    # Check for offline receipt - supports both camelCase and snake_case
    att_id = data.get("attestationId") or data.get("id") or ""
    is_offline = att_id.startswith("oatt_") or data.get("is_offline", False)

    if is_offline:
        receipt = Attestation.model_validate(data)
        receipt.is_offline = True
        result = verify_offline(receipt)
        receipt_type = "Offline"
    else:
        receipt = Attestation.model_validate(data)
        result = verify_online(receipt.id, args.base_url)
        receipt_type = "Online"

    # Output
    print(f"Receipt: {receipt.id}")
    print(f"Type: {receipt_type}")
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
        if result.error:
            # A passing signature with something still wrong — an unsigned
            # field that disagrees with the signed content. Say so; do not let
            # "VALID" swallow it.
            print(f"  Note: {result.error}")
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
