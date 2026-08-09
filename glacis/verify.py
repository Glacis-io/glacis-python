"""
Glacis receipt verification — the CLI, and the offline check behind it.

Usage:
    python -m glacis verify <receipt.json>
    python -m glacis verify <receipt.json> --base-url https://api.glacis.io

``verify_offline()`` here is the one offline verification in the SDK:
``Glacis.verify()`` calls it too, so the library and the command line can never
give different answers about the same receipt.

``verify_attestation()`` is the one *dispatch* for a supplied ``Attestation``
object, shared for the same reason. It never lets the unsigned ``is_offline``
flag decide whether a signature gets looked at.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Union

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

    # cpr_hash is unsigned, so computing its match is a degradation note, never
    # the verdict — and hashing hand-crafted control_plane_results can raise
    # (e.g. a NaN/Infinity a third party put in the JSON). A crash here would
    # turn `verify(some-file-someone-sent-you)` into an unhandled exception
    # instead of an honest valid=False. Catch it: the signed-payload rebuild
    # below fails structurally on the same content, which is the real answer.
    try:
        degradation = _cpr_degradation(receipt)
    except ValueError as e:
        degradation = f"cpr_uncanonicalizable: {e}"

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


# =============================================================================
# Dispatch for a supplied Attestation object
# =============================================================================

#: The fields of a supplied ``Attestation`` that a transparency-log entry
#: carries, and therefore the only ones a server lookup can be said to cover.
#: ``signature`` and ``evidence_hash`` are the binding pair — the first is the
#: Arbiter's, the second commits to the exchange. ``service_id`` and
#: ``operation_type`` are compared as well, but only when both sides carry a
#: value, so an object that simply omits them is reported as unbound rather
#: than mistaken for a match.
BINDABLE_FIELDS = ("signature", "evidence_hash", "service_id", "operation_type")

#: Everything on the object that no log entry carries. A bound result says
#: nothing about these, and ``bind_to_log_entry()`` says so in its report.
UNBINDABLE_FIELDS = (
    "control_plane_results",
    "cpr_hash",
    "evidence",
    "review",
    "timestamp",
    "operation_id",
    "operation_sequence",
    "supersedes",
)


def _hex_matches(left: str, right: str) -> bool:
    """Compare two hex fields, with neither side allowed to be empty.

    Two empty strings are not a match. Binding on a field that neither side
    filled in would let an object with no signature "match" a record with no
    signature, which is the opposite of what binding is for.
    """
    return bool(left) and bool(right) and left.strip().lower() == right.strip().lower()


def bind_to_log_entry(receipt: Attestation, record: VerifyResult) -> tuple[bool, str]:
    """Decide whether ``record`` is the log entry for *these* bytes.

    A server lookup answers a question about an **id**. The id is unsigned and
    anybody can put any id on any object, so the answer only describes the
    supplied object if the object and the entry agree on the fields they both
    carry. That agreement is what "bound" means here, and it is checked rather
    than assumed.

    Returns ``(bound, report)``. The report names what was compared either way,
    including what a bound entry still does not cover.
    """
    entry = record.attestation
    if entry is None:
        return False, (
            f"the lookup of {receipt.id} came back without a log entry, so there "
            "is nothing to compare the supplied attestation against"
        )

    mismatched = []
    if not _hex_matches(receipt.signature, entry.signature):
        mismatched.append("signature")
    if not _hex_matches(receipt.evidence_hash, entry.evidence_hash):
        mismatched.append("evidence_hash")
    if receipt.service_id and entry.service_id and receipt.service_id != entry.service_id:
        mismatched.append("service_id")
    if (
        receipt.operation_type
        and entry.operation_type
        and receipt.operation_type != entry.operation_type
    ):
        mismatched.append("operation_type")

    if mismatched:
        return False, (
            f"the log entry for {receipt.id} does not describe the supplied "
            f"attestation — they disagree on {', '.join(mismatched)}"
        )

    return True, (
        f"the supplied attestation matches the log entry for {receipt.id} on "
        "signature and evidence_hash. The entry carries nothing about "
        f"{', '.join(UNBINDABLE_FIELDS)}, so this verdict does not cover them"
    )


def _with_note(existing: Optional[str], note: str) -> str:
    """Append a routing note without discarding what the check already said."""
    return f"{existing} Also: {note}" if existing else note


def verify_attestation(
    receipt: Attestation,
    online_lookup: Callable[[str], VerifyResult],
) -> Union[VerifyResult, OfflineVerifyResult]:
    """Verify a supplied ``Attestation`` **object**, fail-closed.

    ``is_offline`` and ``id`` are unsigned: an attacker holding a receipt can
    set them to anything. Up to 0.8.1.dev0 they chose the verifier outright, so
    flipping ``is_offline`` to ``False`` and pointing ``id`` at some valid
    online attestation routed the call to a server lookup and the supplied
    object's own bad signature was never examined — ``valid=True`` for bytes
    nothing had checked.

    Here ``is_offline`` selects **additional** verification, never a bypass:

    * The offline Ed25519 check runs on every supplied object, always, over the
      payload rebuilt from that object's own signed fields. Nothing skips it.
    * ``is_offline=False`` adds a lookup of ``receipt.id``. The answer is about
      an id, so it is applied to the object only if the object **binds** to the
      returned log entry (see :func:`bind_to_log_entry`).
    * **Bound, local check passed** — the server's :class:`VerifyResult` is
      returned, with the binding named in ``error`` along with what the entry
      cannot cover.
    * **Bound, local check failed** — fail closed. Binding compares the
      *strings* of ``signature`` and ``evidence_hash``; a string-equal
      signature is not a verified one, and an object whose own Ed25519 check
      failed has not been authenticated by anything. The result is the failed
      local check, with the binding reported in ``error`` as
      ``bound-but-unverified:`` — never the server's ``valid``.
    * **Unbound** — the server's answer described some other record, so none of
      it is returned. The result is the object's own offline check, with
      ``error`` naming that the lookup happened and was not applied. ``valid``
      then reflects the supplied object, which is the only thing that was
      actually verified.

    A receipt whose own signature fails is therefore ``valid=False`` whatever
    the server says about the id it claims — bound or not.
    """
    local = verify_offline(receipt)

    # is_offline=True asks for the local check and nothing more. It is already
    # the strictest route, so there is no bypass to close here.
    if receipt.is_offline:
        return local

    record = online_lookup(receipt.id)
    bound, report = bind_to_log_entry(receipt, record)

    if bound:
        if local.valid:
            return record.model_copy(
                update={"error": _with_note(record.error, f"bound: {report}")}
            )
        # Fail closed: the binding matched string-for-string, but a
        # string-equal signature is not a verified one. The supplied bytes
        # failed their own Ed25519 check, so the server's verdict about the
        # id must not be laundered onto them — valid follows the local check.
        return local.model_copy(
            update={
                "error": _with_note(
                    local.error,
                    f"bound-but-unverified: {report}. The log entry matches "
                    "this object's binding fields, but the supplied bytes "
                    "failed their own Ed25519 check, and valid follows that "
                    "check — not the server's answer about the id",
                )
            }
        )

    return local.model_copy(
        update={
            "error": _with_note(
                local.error,
                f"unbound: {report}. is_offline said this was a witnessed "
                "attestation, so the id was looked up; nothing from that lookup "
                "is reported here and it did not contribute to valid. Only the "
                "supplied attestation's own Ed25519 signature was checked",
            )
        }
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

    # Classify, then verify. The classification only ever *adds* a check:
    # `verify_attestation` runs the offline signature check on any receipt that
    # carries one, whatever these two unsigned fields say. Flipping
    # `is_offline` to false and pointing `id` at a valid online attestation
    # used to route straight past that check — see `verify_attestation`.
    result: Union[VerifyResult, OfflineVerifyResult]

    # Supports both camelCase and snake_case
    att_id = data.get("attestationId") or data.get("id") or ""
    receipt = Attestation.model_validate(data)
    if att_id.startswith("oatt_"):
        # An `oatt_` id is a local receipt whatever the flag claims. This only
        # ever narrows the routing to the local check.
        receipt.is_offline = True

    result = verify_attestation(
        receipt, lambda attestation_id: verify_online(attestation_id, args.base_url)
    )

    # What was actually checked, not what the file asked for: an object that
    # claimed to be witnessed and did not bind to the log entry for its id
    # comes back as the offline check it really got.
    receipt_type = "Offline" if isinstance(result, OfflineVerifyResult) else "Online"

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
