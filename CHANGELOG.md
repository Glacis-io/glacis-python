# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-08-13

### Added

- **Hosted (server-attested) minting.** `Glacis(mode="hosted")` computes the
  local attestation exactly as offline mode does, then mints it into the
  Glacis transparency log via `POST /v1/govern?sync_anchor=true` on
  api.glacis.io (`X-Glacis-Key` auth). The gateway receives only
  `{task_class, request_sha256}` — never payload text — where
  `request_sha256` is SHA-256 over the attestation's exact signed bytes.
  Returns a `HostedArtifact`: the local attestation, the gateway's
  `{v:1, receipt, inclusion}` permalink envelope verbatim, the binding, and
  the SDK's own verification result, serializable with `.save()` to one JSON
  file that parses at glacis.io/verify.
- **Local verification before labeling** (`glacis.witness`): RFC 6962
  inclusion recompute from the receipt's own identifier, plus Ed25519
  verification of the signed tree head under a log key configured via
  `GLACIS_LOG_PUBLIC_KEY_HEX` or `log_public_keys=`. No baked-in production
  key ships in the SDK; the tree head's witness countersignature is not
  pinned and not counted as verification.
- Environment configuration: `GLACIS_API_KEY`, `GLACIS_WITNESS_API_BASE`,
  `GLACIS_LOG_PUBLIC_KEY_HEX`, `GLACIS_SIGNING_SEED_HEX`.
- Hosted mints run under one 8-second deadline (POST plus anchor polling),
  matching the portal's mint client. `/v1/govern` is not idempotent, so a
  request that may have been processed (timeout after send, 5xx) is never
  retried; only a connect error — request never sent — is retried, once.
- Version-sync test: `pyproject.toml` and `glacis.__version__` must agree
  (0.7.0 shipped desynced once).

### Changed

- **`witness_status` no longer overclaims.** 0.8.1 returned `WITNESSED` for
  any `is_offline=False` attestation with zero verification. The label set is
  now: `SELF_SIGNED` (offline/locally signed), `LOGGED_UNVERIFIED` (a server
  response exists but nothing was verified locally), and `WITNESSED` — issued
  only by `glacis.witness.classify_envelope` after the inclusion proof
  recomputes to a tree head signed under a configured log key.
  `OfflineVerifyResult.witness_status` is now `SELF_SIGNED` (was
  `UNVERIFIED`).

## [0.8.1] - 2026-08-09

Not published. `pip install glacis` still serves 0.8.0, which has the defect
described below.

### Fixed

- **Offline `verify()` did not verify anything.** `Glacis._verify_offline()`
  derived a public key from the client's own `signing_seed` and compared it to
  `attestation.public_key`; if they matched it returned `valid=True` and
  `signature_valid=True` **without looking at `signature`**, and with no seed at
  all `signature_valid` was hardcoded `True`. The CLI's `verify_offline()` was
  weaker still: `id` starts `oatt_`, two fields are 64 characters, `signature`
  is non-empty. Both reported `Signature: PASS` over 128 zeroes.

  The consequence was not theoretical. The receipt store is a file the SDK
  itself writes: edit `control_plane_json` in the SQLite row, or strip both the
  CPR and the unsigned `cpr_hash`, and the SDK called the reloaded receipt valid
  while an independent Ed25519 check failed. Same for any receipt handed to you
  by someone else.

  Offline verification is now a real Ed25519 check, and there is exactly one
  implementation of it — `glacis.verify.verify_offline()` — which both
  `Glacis.verify()` and `python -m glacis verify` call:

  - `crypto.offline_signed_payload()` is the single definition of the signed
    bytes. The signer builds them with it and the verifier rebuilds them with
    it, so the two cannot drift.
  - `Ed25519Runtime.verify()` checks the signature under the public key **on
    the receipt**. No signing seed is required, which is what makes a receipt
    checkable by the party it was handed to.
  - `error` names the failure: `signature_invalid` (rebuilt, does not verify),
    `structural` (undecodable key or signature, or no timestamp — "could not
    check" is not "wrong"), `cpr_unrecoverable` (the store could not return
    signed content, so no verdict is possible).
  - `cpr_hash` is **unsigned**, so a mismatch against the recovered
    control-plane results never overrides the signature. It is reported by name
    (`cpr_hash_mismatch`, `cpr_hash_orphaned`) alongside `valid=True`, because a
    receipt that disagrees with itself is worth saying out loud.

  Return shape is unchanged. Verifying a receipt signed by a key that is not
  yours now succeeds when the signature is good, where 0.8.0 rejected it —
  correctly, since a signature is checked against the key on the receipt, and
  that establishes internal consistency and not identity.

- **An unsigned field chose whether a signature was checked.**
  `Glacis.verify()` selected its verifier from `Attestation.is_offline`, which
  is not inside the signature and which anyone holding a receipt can set. With
  `is_offline=False` and `id` pointing at a valid online attestation, the call
  routed to a server lookup, the supplied object's own signature was never
  examined, and the caller read `valid=True` as an answer about the bytes they
  held. It was an answer about an id. `python -m glacis verify` reclassified on
  the same two unsigned fields.

  The signature check itself was never wrong — it was not consulted. So the
  dispatch is what changed, and `is_offline` can no longer subtract a check:

  - The offline Ed25519 check runs on **every** supplied `Attestation` object,
    always.
  - `is_offline=False` **adds** a lookup of `receipt.id`. Because that answer
    describes an id and not your bytes, it is applied to the object only when
    the two **bind**: matching `signature` and `evidence_hash`, plus
    `service_id` and `operation_type` when both sides carry them.
  - Bound **and the object's own signature verified**, you get the server's
    `VerifyResult`, with `error` naming the binding and listing what the log
    entry carries nothing about — `control_plane_results`, `cpr_hash`,
    `evidence`, `review`, `timestamp`, `operation_id`, `operation_sequence`,
    `supersedes`.
  - Bound but the object's own check failed, you get the failed
    `OfflineVerifyResult` with `error` carrying `bound-but-unverified:`.
    Binding compares the *strings* of `signature` and `evidence_hash`; a
    string-equal signature is not a verified one, and the server's `valid` is
    never applied to bytes that failed their own Ed25519 check. (Closed after
    an external review found the bound path returning the server's `valid=True`
    over a failed local check.)
  - Unbound, none of the server's answer is returned — not the org, not the
    proof, not the tree head. You get the object's own `OfflineVerifyResult`,
    `valid` reflecting the bytes that were checked, and `error` saying the
    lookup happened and was not applied.

  `glacis.verify.verify_attestation()` is that one dispatch, called by the
  library and the CLI alike.

- **A failed schema migration stamped itself as successful.**
  `_run_migrations()` caught every `sqlite3.OperationalError` as "column already
  exists" and then wrote `schema_version = 5` regardless. A database claiming v4
  without an `offline_receipts` table — crafted, truncated, half-restored — came
  out recorded as a migrated v5 and was read from then on through a schema it
  did not have. Only `duplicate column name` now means "already exists";
  anything else raises `StorageMigrationError`, naming the version pair, and the
  version stamp is written only after every step has actually been applied. The
  same applies to the v3→v4 step, which now reads the columns before deciding
  whether to rename. A connection whose migrations failed is closed and
  discarded rather than handed to the caller.

  **That was still not enough.** `duplicate column name` is evidence about one
  column — it says the column this step adds is already there, and nothing about
  the other fifteen, the second table, or the indexes. A database declaring v4
  whose `offline_receipts` held only `control_plane_json` therefore swallowed
  the error legitimately and finished stamped v5 with one column to its name.
  The required schema is now stated per version target and validated against the
  live database after every migration step set, before any version is stamped;
  freshly created databases are validated the same way. Every missing table,
  column and index is named in the error.

- **Persisted offline receipts lost signed control-plane content.**
  `control_plane_results` is *inside* the offline signature —
  `Glacis._attest_offline` puts the whole structure into the payload it signs —
  but neither storage backend persisted it. A receipt that verified perfectly
  when returned no longer carried the content its own signature covered once it
  had been through `store_receipt()` / `get_receipt()`, so rebuilding the signed
  payload from a reloaded receipt produced different bytes and **independent
  Ed25519 verification failed**. The SDK's own offline `verify()` did not notice,
  because it only compared the locally derived public key and never checked a
  signature — fixed above, so this class of loss now fails locally too.

  Present in every release that had `control_plane_results`, up to and including
  0.8.0. Anyone who attached control-plane results and relied on the local store
  as their copy of record has receipts that a third party cannot verify. The
  content was never written, so it cannot be recovered from the store; the fix is
  forward-looking only.

  - SQLite: schema v5 adds `offline_receipts.control_plane_json`, written whole
    and never truncated, with a v4→v5 migration.
  - JSONL: the receipt line now carries `control_plane_results`.
  - Both backends reconstruct it on read.
  - **Rows written before this release stay honest rather than convenient.** A
    row with a `cpr_hash` but no stored content is *not* reconstructed as "this
    receipt had no control-plane results". `Attestation.cpr_recovery_error`
    carries the reason in words, and `Glacis.verify()` fails closed on such a
    receipt with that reason as its `error`. Receipts that genuinely had no
    control-plane results are unaffected and still verify.

- **Schema version never advanced past the first migration.** `version` is the
  primary key of `schema_version`, so `INSERT OR REPLACE` appended a row instead
  of replacing one, and the reader took the *first* row — the oldest version.
  Every migration re-ran on every connection open. The reader now takes
  `MAX(version)` and the writer collapses the table to a single row.

- **The wrapper published its receipt before storing evidence.**
  `attest_and_store()` called `set_last_receipt()` before `store_evidence()`, so
  a storage failure left `get_last_receipt()` handing back a receipt for a call
  the docs say produced none. The receipt is now published only after storage
  returns. When storage fails, `get_last_receipt()` keeps returning whatever it
  returned before — nothing, or the previous call's receipt — which is what the
  documentation describes. The attestation itself may still exist (an offline
  receipt row is written by `attest()`, an online one is on the server); what
  changed is that the wrapper no longer presents it as complete.

### Added

- `Ed25519Runtime.verify(public_key_hex, message, signature_hex)` — Ed25519
  verification over exact bytes. Returns `False` for a well-formed signature
  that does not verify; raises `CryptoError` when the key or the signature
  cannot be decoded, because "could not check" is a different answer.
- `crypto.offline_signed_payload()` / `offline_signed_payload_for()` — the
  signed bytes of an offline attestation, in one place, used by the signer and
  the verifier. Read this one function to write a compatible verifier.
- `glacis.verify.verify_offline(receipt)` is now the SDK's offline
  verification, importable and usable on its own. `Glacis.verify()` delegates
  to it.
- `glacis.verify.verify_attestation(receipt, online_lookup)` — the one dispatch
  for a supplied `Attestation` object, used by `Glacis.verify()` and by
  `python -m glacis verify`, and `glacis.verify.bind_to_log_entry()`, which
  decides whether a transparency-log entry describes the object in hand.
- `storage.REQUIRED_SCHEMA` / `storage.DECLARED_INDEXES` — the tables, columns
  and indexes each schema version means, checked before that version is stamped.
- `storage.StorageMigrationError` — raised when a schema migration cannot be
  applied, instead of stamping a version the database has not reached.
- `Attestation.cpr_recovery_error` — SDK convenience, never signed and never
  transmitted. Set only by storage reconstruction, to name a loss instead of
  hiding it.

### Changed

- `python -m glacis verify` exits `1` on an offline receipt whose signature does
  not verify, and prints the named reason. On 0.8.0 it exited `0`. A CI gate on
  this command means something for the first time; pin `glacis>=0.8.1` if you
  are relying on that.
- `OfflineVerifyResult.error` is now set on some receipts that **passed**, when
  an unsigned field disagrees with the signed content. `valid` still follows the
  signature alone.
- `Glacis.verify(attestation)` on an object with `is_offline=False` returns an
  `OfflineVerifyResult` when the object does not bind to the log entry for its
  id, where 0.8.0 returned whatever the server said about that id. Code that
  assumed the return type followed `is_offline` should read the type, or pass an
  id string when it wants a lookup. `verify("att_…")` is unchanged.
- `python -m glacis verify` prints the check it actually ran: a receipt claiming
  to be witnessed that does not bind to its log entry is reported `Offline`.

## [0.5.0] - 2025-02-24

### Breaking Changes
- **Vocabulary rename**: `action` field → `if_detected` in all control configs
- **Action values**: `"pass"` → `"forward"`, `output_block_action`: `"suppress"` → `"block"`, `"flag"` → `"forward"`
- **Determination values**: `determination.action` is now `"forwarded"` / `"blocked"`
- **Config format**: v1.3 nested structure (`controls.input.pii_phi` instead of `controls.pii_phi`)
- **Streaming removed**: Deferred to future release
- **Redaction removed**: PII control now detects but does not rewrite text
- **Storage**: Added JSONL backend as alternative to SQLite (default unchanged)

### Added
- Word filter control (case-insensitive term matching, configurable per-stage)
- LLM Judge framework (`BaseJudge`, `JudgeRunner`, `JudgesConfig`)
- Sampling config with `l1_rate` / `l2_rate` (nested L2⊂L1)
- JSONL storage backend option (append-only `receipts.jsonl` + `evidence.jsonl`)
- `decompose()` for batch → per-item attestations with shared `operation_id`
- `should_review()` deterministic HMAC-SHA256 sampling gate
- Custom controls interface (`BaseControl`)
- Google Gemini integration (`attested_gemini()`)

### Fixed
- Removed unconditional `print()` in jailbreak control
- Added validation: `l2_rate` must be <= `l1_rate`
- Cleaned up dead code in `_canonicalize_value()`
- Fixed `get_blocking_control()` return type annotation
- Fixed control ID collision for same control on input+output stages
- Fixed `sample_probability` hardcoded to 0.0 — now uses actual sampling rate
- Fixed `verify.py` using `evidence_hash` instead of attestation `id`
- Fixed broken examples (`basic_offline.py`, `online_openai.py`)
- Various docstring and test fixes

## [0.3.0] - 2025-01-15

### Added
- Anthropic integration (`attested_anthropic()`)
- PII/PHI detection control
- Jailbreak detection control
- YAML configuration file support
- Control pipeline with staged input/output execution

## [0.2.0] - 2025-01-02

Initial public release.

### Features

- **Online attestation** with Merkle tree inclusion proofs and signed tree heads
- **Offline mode** with local Ed25519 signing (no API key required)
- **OpenAI integration** - auto-attesting wrapper for chat completions
- **Anthropic integration** - auto-attesting wrapper for messages
- **Streaming sessions** - chunk-by-chunk attestation for streaming responses
- **SQLite storage** - local receipt persistence at `~/.glacis/receipts.db`
- **Cross-runtime hashing** - RFC 8785 canonical JSON, compatible with Rust/TypeScript
- **CLI verification** - `python -m glacis verify receipt.json`

### Security

- Zero-egress design: only SHA-256 hashes are transmitted, never payloads
- Ed25519 signatures via PyNaCl (libsodium) or WASM runtime
- Offline receipts clearly marked as "UNVERIFIED"

### Notes

- Offline receipts show `witness_status: "UNVERIFIED"` - this is by design
- For witnessed attestations with Merkle proofs, use online mode with an API key
- Get your API key at https://glacis.io
