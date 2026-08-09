"""
Storage backends for attestation receipts and evidence.

Supports two backends:
- SQLite (default): Stores in ~/.glacis/glacis.db
- JSON: Append-only JSONL files — ~/.glacis/receipts.jsonl and ~/.glacis/evidence.jsonl

Backend is selected via glacis.yaml configuration:
    evidence_storage:
      backend: "sqlite"  # or "json"
      path: "~/.glacis"

Evidence (input, output, control_plane_results) is stored locally for zero-egress
compliance - only hashes are sent to the GLACIS server.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from glacis.models import Attestation

DEFAULT_DB_PATH = Path.home() / ".glacis" / "glacis.db"
DEFAULT_STORAGE_DIR = Path.home() / ".glacis"


# ==============================================================================
# Storage Backend Protocol
# ==============================================================================


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol defining the storage backend interface.

    All storage backends (SQLite, JSON, etc.) must implement this interface.
    """

    def store_receipt(
        self,
        receipt: "Attestation",
        input_preview: Optional[str] = None,
        output_preview: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None: ...

    def get_receipt(self, attestation_id: str) -> Optional["Attestation"]: ...

    def get_last_receipt(self) -> Optional["Attestation"]: ...

    def query_receipts(
        self,
        service_id: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 50,
    ) -> list["Attestation"]: ...

    def count_receipts(self, service_id: Optional[str] = None) -> int: ...

    def delete_receipt(self, attestation_id: str) -> bool: ...

    def store_evidence(
        self,
        attestation_id: str,
        attestation_hash: str,
        mode: str,
        service_id: str,
        operation_type: str,
        timestamp: int,
        input_data: Any,
        output_data: Any,
        control_plane_results: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        sampling_level: str = "L0",
    ) -> None: ...

    def get_evidence(self, attestation_id: str) -> Optional[dict[str, Any]]: ...

    def get_evidence_by_hash(self, attestation_hash: str) -> Optional[dict[str, Any]]: ...

    def query_evidence(
        self,
        service_id: Optional[str] = None,
        mode: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]: ...

    def count_evidence(
        self, service_id: Optional[str] = None, mode: Optional[str] = None
    ) -> int: ...

    def close(self) -> None: ...


# ==============================================================================
# Factory
# ==============================================================================


CPR_LOST_IN_STORAGE = (
    "control_plane_results is missing from the stored row while cpr_hash is set. "
    "The content is inside the offline signature, so the signed payload cannot be "
    "rebuilt from this row and independent Ed25519 verification will fail. "
    "Receipts written by glacis <= 0.8.0 never persisted this field; it cannot be "
    "recovered from the receipt store."
)


def _recover_cpr(
    stored_cpr: Optional[Any],
    cpr_hash: Optional[str],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Reconstruct control_plane_results from a stored row, or name the loss.

    Returns ``(control_plane_results, cpr_recovery_error)``.

    Three cases, and none of them fabricate content:

    * The row carries the CPR: it is returned as stored.
    * The row carries neither CPR nor ``cpr_hash``: the receipt was signed
      without control-plane results. Absent stays absent, no error.
    * The row carries a ``cpr_hash`` but no CPR: signed content that this
      store cannot return. Absent stays absent **and** the reason is set, so
      the caller can see the degradation instead of inferring "no CPR".
    """
    if isinstance(stored_cpr, str):
        try:
            parsed: Optional[Any] = json.loads(stored_cpr)
        except json.JSONDecodeError:
            return None, (
                "control_plane_results is present in storage but is not valid JSON; "
                "the signed payload cannot be rebuilt from this row."
            )
    else:
        parsed = stored_cpr

    if isinstance(parsed, dict):
        return parsed, None
    if parsed is not None:
        return None, (
            "control_plane_results is present in storage but is not a JSON object; "
            "the signed payload cannot be rebuilt from this row."
        )
    if cpr_hash:
        return None, CPR_LOST_IN_STORAGE
    return None, None


def create_storage(
    backend: str = "sqlite",
    path: Optional[Path] = None,
) -> StorageBackend:
    """Create a storage backend instance.

    Args:
        backend: Backend type - "sqlite" or "json"
        path: Base path. For SQLite: full .db file path.
              For JSON: directory containing receipts.jsonl and evidence.jsonl.
              Defaults to ~/.glacis/glacis.db (sqlite) or ~/.glacis (json).

    Returns:
        A StorageBackend instance
    """
    if backend == "json":
        return JsonStorageBackend(base_dir=path or DEFAULT_STORAGE_DIR)
    # Default to SQLite
    return ReceiptStorage(db_path=path or DEFAULT_DB_PATH)


# ==============================================================================
# SQLite Backend
# ==============================================================================

SCHEMA_VERSION = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS offline_receipts (
    attestation_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    service_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    evidence_hash TEXT NOT NULL,
    signature TEXT NOT NULL,
    public_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    input_preview TEXT,
    output_preview TEXT,
    metadata_json TEXT,
    operation_id TEXT,
    operation_sequence INTEGER DEFAULT 0,
    supersedes TEXT,
    cpr_hash TEXT,
    -- control_plane_results is INSIDE the offline signature. It has to be
    -- stored verbatim or the signed payload cannot be rebuilt on reload.
    control_plane_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_service_id ON offline_receipts(service_id);
CREATE INDEX IF NOT EXISTS idx_timestamp ON offline_receipts(timestamp);
CREATE INDEX IF NOT EXISTS idx_evidence_hash ON offline_receipts(evidence_hash);
CREATE INDEX IF NOT EXISTS idx_created_at ON offline_receipts(created_at);
CREATE INDEX IF NOT EXISTS idx_operation_id ON offline_receipts(operation_id);

-- Evidence table for full audit trail (both online and offline modes)
CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attestation_id TEXT NOT NULL,
    attestation_hash TEXT NOT NULL,
    mode TEXT NOT NULL,  -- 'online' or 'offline'
    service_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL,
    input_json TEXT NOT NULL,
    output_json TEXT NOT NULL,
    control_plane_json TEXT,
    metadata_json TEXT,
    sampling_level TEXT NOT NULL DEFAULT 'L0',  -- 'L0', 'L1', or 'L2'
    UNIQUE(attestation_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_attestation_id ON evidence(attestation_id);
CREATE INDEX IF NOT EXISTS idx_evidence_attestation_hash ON evidence(attestation_hash);
CREATE INDEX IF NOT EXISTS idx_evidence_service_id ON evidence(service_id);
CREATE INDEX IF NOT EXISTS idx_evidence_timestamp ON evidence(timestamp);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""

# Migration from v1 to v2: Add evidence table
MIGRATION_V1_TO_V2 = """
CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attestation_id TEXT NOT NULL,
    attestation_hash TEXT NOT NULL,
    mode TEXT NOT NULL,
    service_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL,
    input_json TEXT NOT NULL,
    output_json TEXT NOT NULL,
    control_plane_json TEXT,
    metadata_json TEXT,
    UNIQUE(attestation_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_attestation_id ON evidence(attestation_id);
CREATE INDEX IF NOT EXISTS idx_evidence_attestation_hash ON evidence(attestation_hash);
CREATE INDEX IF NOT EXISTS idx_evidence_service_id ON evidence(service_id);
CREATE INDEX IF NOT EXISTS idx_evidence_timestamp ON evidence(timestamp);
"""

# Migration from v2 to v3: Add sampling_level column to evidence table
MIGRATION_V2_TO_V3 = """
ALTER TABLE evidence ADD COLUMN sampling_level TEXT NOT NULL DEFAULT 'L0';
"""

# Migration from v4 to v5: persist the signed control_plane_results on the
# receipt row. Rows written before this migration cannot be back-filled — the
# content was never stored — so they stay NULL and are reported as an explicit
# degradation on read rather than being silently reconstructed as "no CPR".
MIGRATION_V4_TO_V5 = """
ALTER TABLE offline_receipts ADD COLUMN control_plane_json TEXT;
"""

# The indexes SCHEMA declares, restated as idempotent statements so that a
# database arriving from any older version ends up with the set a fresh one
# gets. None of the migration steps above creates the three convenience indexes
# on offline_receipts, so without this a migrated database would be missing
# them for good. Every statement is IF NOT EXISTS, so this is a no-op on a
# database that already has them — and a genuine failure on one whose columns
# are not there to index.
ENSURE_DECLARED_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_service_id ON offline_receipts(service_id);
CREATE INDEX IF NOT EXISTS idx_timestamp ON offline_receipts(timestamp);
CREATE INDEX IF NOT EXISTS idx_evidence_hash ON offline_receipts(evidence_hash);
CREATE INDEX IF NOT EXISTS idx_created_at ON offline_receipts(created_at);
CREATE INDEX IF NOT EXISTS idx_operation_id ON offline_receipts(operation_id);
CREATE INDEX IF NOT EXISTS idx_evidence_attestation_id ON evidence(attestation_id);
CREATE INDEX IF NOT EXISTS idx_evidence_attestation_hash ON evidence(attestation_hash);
CREATE INDEX IF NOT EXISTS idx_evidence_service_id ON evidence(service_id);
CREATE INDEX IF NOT EXISTS idx_evidence_timestamp ON evidence(timestamp);
"""


# =============================================================================
# Required schema — the postcondition of a migration, stated as data
# =============================================================================
#
# A migration step reporting success is not evidence that the schema is now
# what the version number claims. `ALTER TABLE … ADD COLUMN` failing with
# "duplicate column name" means only that *that column* is present; it says
# nothing about the fifteen others, the second table, or the indexes. A v4
# database whose `offline_receipts` held nothing but `control_plane_json`
# therefore used to swallow the duplicate-column error and get stamped v5.
#
# So the required shape is written down per version target and checked against
# the live database before anything is stamped. Columns are the ones the
# storage code actually reads and writes; the hash column is deliberately
# absent below v4, because it is still named `payload_hash` there and the
# v3->v4 step checks it by name.

_OFFLINE_RECEIPTS_BASE = {
    "attestation_id",
    "timestamp",
    "service_id",
    "operation_type",
    "signature",
    "public_key",
    "created_at",
    "input_preview",
    "output_preview",
    "metadata_json",
}

_OFFLINE_RECEIPTS_V4 = _OFFLINE_RECEIPTS_BASE | {
    "evidence_hash",
    "operation_id",
    "operation_sequence",
    "supersedes",
    "cpr_hash",
}

_EVIDENCE_V2 = {
    "id",
    "attestation_id",
    "attestation_hash",
    "mode",
    "service_id",
    "operation_type",
    "timestamp",
    "created_at",
    "input_json",
    "output_json",
    "control_plane_json",
    "metadata_json",
}

_EVIDENCE_V3 = _EVIDENCE_V2 | {"sampling_level"}

_EVIDENCE_INDEXES = {
    "idx_evidence_attestation_id",
    "idx_evidence_attestation_hash",
    "idx_evidence_service_id",
    "idx_evidence_timestamp",
}

#: version -> (required tables and their columns, required indexes).
#: Cumulative: each entry is the whole shape a database must have to be called
#: that version, not just what the last step added. The index sets list only
#: what the migration path itself creates — the three convenience indexes on
#: offline_receipts are added by ENSURE_DECLARED_INDEXES and checked in
#: DECLARED_INDEXES below, after that step has run.
REQUIRED_SCHEMA: dict[int, tuple[dict[str, set[str]], set[str]]] = {
    2: (
        {
            "offline_receipts": _OFFLINE_RECEIPTS_BASE,
            "evidence": _EVIDENCE_V2,
            "schema_version": {"version"},
        },
        set(_EVIDENCE_INDEXES),
    ),
    3: (
        {
            "offline_receipts": _OFFLINE_RECEIPTS_BASE,
            "evidence": _EVIDENCE_V3,
            "schema_version": {"version"},
        },
        set(_EVIDENCE_INDEXES),
    ),
    4: (
        {
            "offline_receipts": _OFFLINE_RECEIPTS_V4,
            "evidence": _EVIDENCE_V3,
            "schema_version": {"version"},
        },
        _EVIDENCE_INDEXES | {"idx_operation_id", "idx_evidence_hash"},
    ),
    5: (
        {
            "offline_receipts": _OFFLINE_RECEIPTS_V4 | {"control_plane_json"},
            "evidence": _EVIDENCE_V3,
            "schema_version": {"version"},
        },
        _EVIDENCE_INDEXES | {"idx_operation_id", "idx_evidence_hash"},
    ),
}

#: Every index SCHEMA declares. Checked once, at the end, after
#: ENSURE_DECLARED_INDEXES has had its chance to create the missing ones.
DECLARED_INDEXES = _EVIDENCE_INDEXES | {
    "idx_service_id",
    "idx_timestamp",
    "idx_evidence_hash",
    "idx_created_at",
    "idx_operation_id",
}


class StorageMigrationError(RuntimeError):
    """A schema migration could not be applied.

    Raised instead of stamping a version the database has not reached. The
    caller gets a receipt store that refuses to open, which is the honest
    answer — the alternative is a database labelled v5 that is not v5, read
    through a schema that does not match it.
    """


def _is_duplicate_column(exc: sqlite3.OperationalError) -> bool:
    """True only for SQLite's "duplicate column name: x".

    That is the one failure a migration step may ignore: it means the column
    this step adds is already there, which is what re-running a partly applied
    migration looks like. Every other OperationalError — no such table, disk
    I/O error, malformed schema — means the step did not do its job.
    """
    return "duplicate column name" in str(exc).lower()


def _migrate_step(
    cursor: sqlite3.Cursor,
    script: str,
    from_version: int,
    to_version: int,
) -> None:
    """Run one migration script, tolerating only an already-present column."""
    try:
        cursor.executescript(script)
    except sqlite3.OperationalError as exc:
        if _is_duplicate_column(exc):
            return
        raise StorageMigrationError(
            f"migration v{from_version}->v{to_version} failed: {exc}. "
            "The database has been left at its previous schema version; "
            "nothing was stamped as migrated."
        ) from exc


def _table_columns(cursor: sqlite3.Cursor, table: str) -> set[str]:
    """Column names of ``table``, or an empty set when it does not exist."""
    return {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}


def _index_names(cursor: sqlite3.Cursor) -> set[str]:
    """Every index the database defines, by name."""
    return {
        row[0]
        for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }


def _validate_schema(
    cursor: sqlite3.Cursor,
    version: int,
    *,
    indexes: Optional[set[str]] = None,
) -> None:
    """Check the live schema against what ``version`` requires, or raise.

    This is the postcondition, and it is what decides whether a version may be
    stamped — not the fact that the ALTERs raised nothing worth stopping for.
    It runs after every step set, so a corrupt shape that survives the ALTERs
    is caught at the step that was supposed to produce it rather than being
    discovered later by a reader.

    Every missing thing is collected and named, so one open tells the caller
    the whole story instead of one item per attempt.
    """
    tables, required_indexes = REQUIRED_SCHEMA[version]
    if indexes is not None:
        required_indexes = indexes

    missing: list[str] = []

    for table, columns in tables.items():
        present = _table_columns(cursor, table)
        if not present:
            missing.append(f"table {table}")
            continue
        absent = columns - present
        if absent:
            missing.append(f"{table} columns {', '.join(sorted(absent))}")

    absent_indexes = required_indexes - _index_names(cursor)
    if absent_indexes:
        missing.append(f"indexes {', '.join(sorted(absent_indexes))}")

    if missing:
        raise StorageMigrationError(
            f"schema postcondition for v{version} failed: {'; '.join(missing)}. "
            "The migration steps raised nothing, but the database does not have "
            f"the shape v{version} requires — a duplicate-column error means one "
            "column was already there, not that the migration is complete. The "
            "database has been left at its previous schema version; nothing was "
            "stamped as migrated."
        )


class ReceiptStorage:
    """
    SQLite storage backend for attestation receipts and evidence.

    Default location: ~/.glacis/glacis.db
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            try:
                self._init_schema()
            except Exception:
                # A database whose schema could not be brought up to date is
                # not usable, and half-opening it would hand the caller a
                # connection whose migrations never ran. Drop it and let the
                # error out: the version on disk is untouched, so the next
                # open retries the same migration rather than reading rows
                # through a schema that does not match SCHEMA_VERSION.
                self._conn.close()
                self._conn = None
                raise
        return self._conn

    def _init_schema(self) -> None:
        """Initialize database schema if needed."""
        conn = self._conn
        if conn is None:
            return

        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        if cursor.fetchone() is None:
            cursor.executescript(SCHEMA)
            # The same postcondition a migrated database has to satisfy. A
            # fresh database is where SCHEMA and REQUIRED_SCHEMA would drift
            # apart unnoticed, so it is checked here too: if the two disagree,
            # every new store fails loudly instead of only the upgrade path.
            _validate_schema(cursor, SCHEMA_VERSION, indexes=DECLARED_INDEXES)
            cursor.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            conn.commit()
        else:
            # MAX(), not LIMIT 1: `version` is the primary key, so an upgrade
            # adds a row rather than replacing one. Reading the first row read
            # the *oldest* version back and re-ran every migration on every
            # open. _run_migrations now collapses the table to one row.
            cursor.execute("SELECT MAX(version) FROM schema_version")
            row = cursor.fetchone()
            current = row[0] if row and row[0] is not None else 0
            if current < SCHEMA_VERSION:
                self._run_migrations(current)

    def _run_migrations(self, from_version: int) -> None:
        """Run schema migrations, or fail without claiming they ran.

        Every step tolerates exactly one thing — the column it adds already
        being there, which is what a re-run of a partly applied migration
        looks like. Anything else propagates as ``StorageMigrationError`` and
        the version stamp is never written, so the database keeps saying the
        version it actually is.

        A step that raises nothing is not proof that it worked, so each step
        set is followed by ``_validate_schema()``: the required tables,
        columns and indexes for that version target, checked against the live
        database. The version is stamped only after the *declared* schema —
        everything a freshly created database has — validates in full.
        """
        conn = self._conn
        if conn is None:
            return

        cursor = conn.cursor()

        if from_version < 2:
            _migrate_step(cursor, MIGRATION_V1_TO_V2, from_version, 2)
            _validate_schema(cursor, 2)

        if from_version < 3:
            _migrate_step(cursor, MIGRATION_V2_TO_V3, from_version, 3)
            _validate_schema(cursor, 3)

        # Migration v3 to v4: rename payload_hash -> evidence_hash, add new columns
        if from_version < 4:
            self._migrate_v3_to_v4(cursor)
            _validate_schema(cursor, 4)

        # Migration v4 to v5: add control_plane_json to offline_receipts
        if from_version < 5:
            _migrate_step(cursor, MIGRATION_V4_TO_V5, from_version, 5)
            _validate_schema(cursor, 5)

        # The index set is the last thing to bring across, and it is the only
        # part of the declared schema no version step creates in full.
        _migrate_step(cursor, ENSURE_DECLARED_INDEXES, from_version, SCHEMA_VERSION)
        _validate_schema(cursor, SCHEMA_VERSION, indexes=DECLARED_INDEXES)

        # Only now, with every step applied and the resulting schema checked
        # against what SCHEMA_VERSION means.
        cursor.execute("DELETE FROM schema_version")
        cursor.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
        conn.commit()

    def _migrate_v3_to_v4(self, cursor: sqlite3.Cursor) -> None:
        """Migrate v3 to v4: rename payload_hash, add operation_id/sequence/supersedes/cpr_hash."""
        columns = _table_columns(cursor, "offline_receipts")
        if not columns:
            raise StorageMigrationError(
                "migration v3->v4 failed: there is no offline_receipts table to "
                "migrate. The database says it is v3 but does not have a v3 schema."
            )

        # payload_hash -> evidence_hash. Skipped outright when the column is
        # already named the new way, which is what a re-run looks like.
        if "evidence_hash" not in columns:
            if "payload_hash" not in columns:
                raise StorageMigrationError(
                    "migration v3->v4 failed: offline_receipts has neither "
                    "evidence_hash nor payload_hash, so the hash column cannot "
                    "be carried across."
                )
            try:
                cursor.execute(
                    "ALTER TABLE offline_receipts RENAME COLUMN payload_hash TO evidence_hash"
                )
            except sqlite3.OperationalError as rename_exc:
                # SQLite < 3.25 has no RENAME COLUMN. Add and copy instead —
                # and if *that* fails, the migration has failed. It is not
                # "already fine".
                try:
                    cursor.execute(
                        "ALTER TABLE offline_receipts ADD COLUMN evidence_hash TEXT"
                    )
                    cursor.execute(
                        "UPDATE offline_receipts SET evidence_hash = payload_hash"
                    )
                except sqlite3.OperationalError as copy_exc:
                    raise StorageMigrationError(
                        "migration v3->v4 failed: could not produce an "
                        f"evidence_hash column (rename: {rename_exc}; "
                        f"add-and-copy fallback: {copy_exc})."
                    ) from copy_exc

        # Add new v1.2 columns. Already present is fine; anything else is not.
        for col_sql in [
            "ALTER TABLE offline_receipts ADD COLUMN operation_id TEXT",
            "ALTER TABLE offline_receipts ADD COLUMN operation_sequence INTEGER DEFAULT 0",
            "ALTER TABLE offline_receipts ADD COLUMN supersedes TEXT",
            "ALTER TABLE offline_receipts ADD COLUMN cpr_hash TEXT",
        ]:
            _migrate_step(cursor, col_sql, 3, 4)

        # Both columns exist by now, so an index that will not build is a real
        # failure rather than the expected consequence of a skipped rename.
        for index_sql in [
            "CREATE INDEX IF NOT EXISTS idx_operation_id ON offline_receipts(operation_id)",
            "CREATE INDEX IF NOT EXISTS idx_evidence_hash ON offline_receipts(evidence_hash)",
        ]:
            _migrate_step(cursor, index_sql, 3, 4)

    def store_receipt(
        self,
        receipt: "Attestation",
        input_preview: Optional[str] = None,
        output_preview: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Store an offline attestation.

        Args:
            receipt: The Attestation to store
            input_preview: Optional preview of input (first 100 chars)
            output_preview: Optional preview of output (first 100 chars)
            metadata: Optional metadata dict
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO offline_receipts
            (attestation_id, timestamp, service_id, operation_type,
             evidence_hash, signature, public_key, created_at,
             input_preview, output_preview, metadata_json,
             operation_id, operation_sequence, supersedes, cpr_hash,
             control_plane_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.id,
                receipt.timestamp,
                receipt.service_id,
                receipt.operation_type,
                receipt.evidence_hash,
                receipt.signature,
                receipt.public_key,
                datetime.now(timezone.utc).isoformat(),
                input_preview[:100] if input_preview else None,
                output_preview[:100] if output_preview else None,
                json.dumps(metadata) if metadata else None,
                receipt.operation_id,
                receipt.operation_sequence,
                receipt.supersedes,
                receipt.cpr_hash,
                # Signed content, so it is stored whole and never truncated.
                json.dumps(receipt.control_plane_results, separators=(",", ":"))
                if receipt.control_plane_results is not None
                else None,
            ),
        )
        conn.commit()

    def get_receipt(self, attestation_id: str) -> Optional["Attestation"]:
        """
        Retrieve an attestation by ID.

        Args:
            attestation_id: The attestation ID (oatt_xxx)

        Returns:
            The Attestation if found, None otherwise
        """

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM offline_receipts WHERE attestation_id = ?",
            (attestation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return self._row_to_attestation(row)

    def get_last_receipt(self) -> Optional["Attestation"]:
        """
        Get the most recently created attestation.

        Returns:
            The most recent Attestation, or None if none exist
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM offline_receipts ORDER BY created_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return self._row_to_attestation(row)

    def query_receipts(
        self,
        service_id: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 50,
    ) -> list["Attestation"]:
        """
        Query attestations with optional filters.

        Args:
            service_id: Filter by service ID
            start: Filter by timestamp >= start
            end: Filter by timestamp <= end
            limit: Maximum number of results (default 50)

        Returns:
            List of matching Attestation objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM offline_receipts WHERE 1=1"
        params: list[Any] = []

        if service_id:
            query += " AND service_id = ?"
            params.append(service_id)
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [self._row_to_attestation(row) for row in rows]

    def _row_to_attestation(self, row: sqlite3.Row) -> "Attestation":
        """Convert a database row to an Attestation object."""
        from glacis.models import Attestation

        keys = row.keys()
        # Handle both old (payload_hash) and new (evidence_hash) column names
        evidence_hash = (
            row["evidence_hash"]
            if "evidence_hash" in keys and row["evidence_hash"]
            else row["payload_hash"] if "payload_hash" in keys else ""
        )

        cpr_hash = row["cpr_hash"] if "cpr_hash" in keys else None
        # Pre-v5 rows have no column at all; migrated rows have it as NULL.
        # Either way there is nothing to recover — _recover_cpr says so.
        control_plane_results, cpr_recovery_error = _recover_cpr(
            row["control_plane_json"] if "control_plane_json" in keys else None,
            cpr_hash,
        )

        return Attestation(
            id=row["attestation_id"],
            evidence_hash=evidence_hash,
            timestamp=row["timestamp"],
            service_id=row["service_id"],
            operation_type=row["operation_type"],
            signature=row["signature"],
            public_key=row["public_key"],
            is_offline=True,
            operation_id=(
                row["operation_id"]
                if "operation_id" in keys and row["operation_id"]
                else ""
            ),
            operation_sequence=(
                row["operation_sequence"]
                if "operation_sequence" in keys
                and row["operation_sequence"] is not None
                else 0
            ),
            supersedes=row["supersedes"] if "supersedes" in keys else None,
            cpr_hash=cpr_hash,
            control_plane_results=control_plane_results,
            cpr_recovery_error=cpr_recovery_error,
        )

    def count_receipts(self, service_id: Optional[str] = None) -> int:
        """Count receipts, optionally filtered by service ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if service_id:
            cursor.execute(
                "SELECT COUNT(*) FROM offline_receipts WHERE service_id = ?",
                (service_id,),
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM offline_receipts")

        row = cursor.fetchone()
        return row[0] if row else 0

    def delete_receipt(self, attestation_id: str) -> bool:
        """Delete a receipt by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM offline_receipts WHERE attestation_id = ?",
            (attestation_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ReceiptStorage":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.close()

    # =========================================================================
    # Evidence Storage (for full audit trail)
    # =========================================================================

    def store_evidence(
        self,
        attestation_id: str,
        attestation_hash: str,
        mode: str,
        service_id: str,
        operation_type: str,
        timestamp: int,
        input_data: Any,
        output_data: Any,
        control_plane_results: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        sampling_level: str = "L0",
    ) -> None:
        """
        Store full evidence for an attestation.

        Args:
            attestation_id: The attestation ID
            attestation_hash: The evidence_hash that was attested
            mode: 'online' or 'offline'
            service_id: Service identifier
            operation_type: Type of operation
            timestamp: Unix timestamp in milliseconds
            input_data: Full input data (will be JSON serialized)
            output_data: Full output data (will be JSON serialized)
            control_plane_results: Optional CPR (dict or Pydantic model)
            metadata: Optional metadata dict
            sampling_level: Sampling level ('L0', 'L1', or 'L2')
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        control_plane_json = None
        if control_plane_results:
            if hasattr(control_plane_results, "model_dump"):
                control_plane_json = json.dumps(control_plane_results.model_dump())
            elif isinstance(control_plane_results, dict):
                control_plane_json = json.dumps(control_plane_results)

        cursor.execute(
            """
            INSERT OR REPLACE INTO evidence
            (attestation_id, attestation_hash, mode, service_id, operation_type,
             timestamp, created_at, input_json, output_json, control_plane_json,
             metadata_json, sampling_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attestation_id,
                attestation_hash,
                mode,
                service_id,
                operation_type,
                str(timestamp),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(input_data),
                json.dumps(output_data),
                control_plane_json,
                json.dumps(metadata) if metadata else None,
                sampling_level,
            ),
        )
        conn.commit()

    def get_evidence(self, attestation_id: str) -> Optional[dict[str, Any]]:
        """Retrieve full evidence by attestation ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM evidence WHERE attestation_id = ?",
            (attestation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return self._row_to_evidence(row)

    def get_evidence_by_hash(self, attestation_hash: str) -> Optional[dict[str, Any]]:
        """Retrieve full evidence by attestation hash."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM evidence WHERE attestation_hash = ?",
            (attestation_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return self._row_to_evidence(row)

    def query_evidence(
        self,
        service_id: Optional[str] = None,
        mode: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query evidence with optional filters."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM evidence WHERE 1=1"
        params: list[Any] = []

        if service_id:
            query += " AND service_id = ?"
            params.append(service_id)
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [self._row_to_evidence(row) for row in rows]

    def count_evidence(
        self, service_id: Optional[str] = None, mode: Optional[str] = None
    ) -> int:
        """Count evidence records."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT COUNT(*) FROM evidence WHERE 1=1"
        params: list[Any] = []

        if service_id:
            query += " AND service_id = ?"
            params.append(service_id)
        if mode:
            query += " AND mode = ?"
            params.append(mode)

        cursor.execute(query, params)
        row = cursor.fetchone()
        return row[0] if row else 0

    def _row_to_evidence(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a database row to an evidence dict."""
        keys = row.keys()
        return {
            "attestation_id": row["attestation_id"],
            "attestation_hash": row["attestation_hash"],
            "mode": row["mode"],
            "service_id": row["service_id"],
            "operation_type": row["operation_type"],
            "timestamp": row["timestamp"],
            "created_at": row["created_at"],
            "input": json.loads(row["input_json"]),
            "output": json.loads(row["output_json"]),
            "control_plane_results": (
                json.loads(row["control_plane_json"])
                if row["control_plane_json"]
                else None
            ),
            "metadata": (
                json.loads(row["metadata_json"]) if row["metadata_json"] else None
            ),
            "sampling_level": row["sampling_level"] if "sampling_level" in keys else "L0",
        }

    # NOTE: Evidence is intentionally append-only (no delete method).


# ==============================================================================
# JSON Backend
# ==============================================================================


class JsonStorageBackend:
    """
    JSONL (JSON Lines) storage backend for attestation receipts and evidence.

    Stores records as append-only JSONL files (one JSON object per line):
        <base_dir>/receipts.jsonl
        <base_dir>/evidence.jsonl

    This is more efficient than one-file-per-attestation: only two files
    regardless of how many attestations are stored. Each line is a complete,
    self-contained JSON object that can be parsed independently.

    For duplicate attestation IDs (upsert), the last occurrence wins on read.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or DEFAULT_STORAGE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._receipts_path = self.base_dir / "receipts.jsonl"
        self._evidence_path = self.base_dir / "evidence.jsonl"

    def _append_line(self, path: Path, data: dict[str, Any]) -> None:
        """Append a single JSON line to a JSONL file."""
        line = json.dumps(data, separators=(",", ":"), default=str)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _read_lines(self, path: Path) -> list[dict[str, Any]]:
        """Read all JSON lines from a JSONL file."""
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _read_receipts_deduped(self) -> dict[str, dict[str, Any]]:
        """Read all receipts, deduplicating by attestation_id (last wins)."""
        by_id: dict[str, dict[str, Any]] = {}
        for record in self._read_lines(self._receipts_path):
            aid = record.get("attestation_id", "")
            by_id[aid] = record
        return by_id

    def _read_evidence_deduped(self) -> dict[str, dict[str, Any]]:
        """Read all evidence, deduplicating by attestation_id (last wins)."""
        by_id: dict[str, dict[str, Any]] = {}
        for record in self._read_lines(self._evidence_path):
            aid = record.get("attestation_id", "")
            by_id[aid] = record
        return by_id

    # =========================================================================
    # Receipt Storage
    # =========================================================================

    def store_receipt(
        self,
        receipt: "Attestation",
        input_preview: Optional[str] = None,
        output_preview: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append an attestation receipt as a JSONL line."""
        data = {
            "attestation_id": receipt.id,
            "timestamp": receipt.timestamp,
            "service_id": receipt.service_id,
            "operation_type": receipt.operation_type,
            "evidence_hash": receipt.evidence_hash,
            "signature": receipt.signature,
            "public_key": receipt.public_key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_preview": input_preview[:100] if input_preview else None,
            "output_preview": output_preview[:100] if output_preview else None,
            "metadata": metadata,
            "operation_id": receipt.operation_id,
            "operation_sequence": receipt.operation_sequence,
            "supersedes": receipt.supersedes,
            "cpr_hash": receipt.cpr_hash,
            # Signed content, so it is stored whole and never truncated.
            "control_plane_results": receipt.control_plane_results,
        }
        self._append_line(self._receipts_path, data)

    def get_receipt(self, attestation_id: str) -> Optional["Attestation"]:
        """Retrieve an attestation by ID (last occurrence wins)."""
        by_id = self._read_receipts_deduped()
        data = by_id.get(attestation_id)
        if data is None:
            return None
        return self._dict_to_attestation(data)

    def get_last_receipt(self) -> Optional["Attestation"]:
        """Get the most recently created attestation."""
        by_id = self._read_receipts_deduped()
        if not by_id:
            return None
        latest = max(by_id.values(), key=lambda d: d.get("created_at", ""))
        return self._dict_to_attestation(latest)

    def query_receipts(
        self,
        service_id: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 50,
    ) -> list["Attestation"]:
        """Query attestations with optional filters."""
        by_id = self._read_receipts_deduped()
        results: list[dict[str, Any]] = []

        for data in by_id.values():
            if service_id and data.get("service_id") != service_id:
                continue
            ts = str(data.get("timestamp", ""))
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            results.append(data)

        results.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return [self._dict_to_attestation(d) for d in results[:limit]]

    def count_receipts(self, service_id: Optional[str] = None) -> int:
        """Count unique receipts, optionally filtered by service ID."""
        by_id = self._read_receipts_deduped()
        if service_id is None:
            return len(by_id)
        return sum(1 for d in by_id.values() if d.get("service_id") == service_id)

    def delete_receipt(self, attestation_id: str) -> bool:
        """Delete a receipt by rewriting the file without it."""
        by_id = self._read_receipts_deduped()
        if attestation_id not in by_id:
            return False
        del by_id[attestation_id]
        # Rewrite the file without the deleted record
        with open(self._receipts_path, "w", encoding="utf-8") as f:
            for data in by_id.values():
                f.write(json.dumps(data, separators=(",", ":"), default=str) + "\n")
        return True

    def _dict_to_attestation(self, data: dict[str, Any]) -> "Attestation":
        """Convert a stored dict to an Attestation object."""
        from glacis.models import Attestation

        cpr_hash = data.get("cpr_hash")
        # Lines written before 0.8.1 have no control_plane_results key at all.
        control_plane_results, cpr_recovery_error = _recover_cpr(
            data.get("control_plane_results"), cpr_hash
        )

        return Attestation(
            id=data["attestation_id"],
            evidence_hash=data.get("evidence_hash", ""),
            timestamp=data.get("timestamp"),
            service_id=data.get("service_id", ""),
            operation_type=data.get("operation_type", ""),
            signature=data.get("signature", ""),
            public_key=data.get("public_key", ""),
            is_offline=True,
            operation_id=data.get("operation_id", ""),
            operation_sequence=data.get("operation_sequence", 0),
            supersedes=data.get("supersedes"),
            cpr_hash=cpr_hash,
            control_plane_results=control_plane_results,
            cpr_recovery_error=cpr_recovery_error,
        )

    # =========================================================================
    # Evidence Storage
    # =========================================================================

    def store_evidence(
        self,
        attestation_id: str,
        attestation_hash: str,
        mode: str,
        service_id: str,
        operation_type: str,
        timestamp: int,
        input_data: Any,
        output_data: Any,
        control_plane_results: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        sampling_level: str = "L0",
    ) -> None:
        """Append full evidence as a JSONL line."""
        cpr_serialized = None
        if control_plane_results:
            if hasattr(control_plane_results, "model_dump"):
                cpr_serialized = control_plane_results.model_dump()
            elif isinstance(control_plane_results, dict):
                cpr_serialized = control_plane_results

        data = {
            "attestation_id": attestation_id,
            "attestation_hash": attestation_hash,
            "mode": mode,
            "service_id": service_id,
            "operation_type": operation_type,
            "timestamp": str(timestamp),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": input_data,
            "output": output_data,
            "control_plane_results": cpr_serialized,
            "metadata": metadata,
            "sampling_level": sampling_level,
        }
        self._append_line(self._evidence_path, data)

    def get_evidence(self, attestation_id: str) -> Optional[dict[str, Any]]:
        """Retrieve full evidence by attestation ID (last occurrence wins)."""
        by_id = self._read_evidence_deduped()
        return by_id.get(attestation_id)

    def get_evidence_by_hash(self, attestation_hash: str) -> Optional[dict[str, Any]]:
        """Retrieve full evidence by attestation hash."""
        by_id = self._read_evidence_deduped()
        for data in by_id.values():
            if data.get("attestation_hash") == attestation_hash:
                return data
        return None

    def query_evidence(
        self,
        service_id: Optional[str] = None,
        mode: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query evidence with optional filters."""
        by_id = self._read_evidence_deduped()
        results: list[dict[str, Any]] = []

        for data in by_id.values():
            if service_id and data.get("service_id") != service_id:
                continue
            if mode and data.get("mode") != mode:
                continue
            ts = data.get("timestamp", "")
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            results.append(data)

        results.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return results[:limit]

    def count_evidence(
        self, service_id: Optional[str] = None, mode: Optional[str] = None
    ) -> int:
        """Count unique evidence records."""
        by_id = self._read_evidence_deduped()
        if service_id is None and mode is None:
            return len(by_id)
        count = 0
        for data in by_id.values():
            if service_id and data.get("service_id") != service_id:
                continue
            if mode and data.get("mode") != mode:
                continue
            count += 1
        return count

    def close(self) -> None:
        """No-op for JSONL backend (no persistent connections)."""
        pass

    def __enter__(self) -> "JsonStorageBackend":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.close()
