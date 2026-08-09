"""
Schema migrations either run or say they did not.

`_run_migrations()` used to wrap every step in `except sqlite3.OperationalError:
pass  # Column already exists` and then stamp version 5 unconditionally. Only
one of those errors means "already exists". A corrupt or crafted v4 database —
one whose `offline_receipts` table is missing, say — therefore came out the
other side labelled a successfully migrated v5 and was then read through a
schema it did not have.

These tests pin the two halves: a step that legitimately re-runs is tolerated
and reaches v5, and a step that genuinely fails raises by name and leaves the
recorded version alone.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from glacis.storage import (
    SCHEMA_VERSION,
    ReceiptStorage,
    StorageMigrationError,
    _is_duplicate_column,
)

V4_OFFLINE_RECEIPTS = """
CREATE TABLE offline_receipts (
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
    cpr_hash TEXT
);
"""

# The same table with the v5 column already added: what a v4 database looks
# like when the ALTER ran but the version stamp never did.
V4_WITH_V5_COLUMN = V4_OFFLINE_RECEIPTS.replace(
    "    cpr_hash TEXT\n);",
    "    cpr_hash TEXT,\n    control_plane_json TEXT\n);",
)

VERSION_TABLE = """
CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
"""


def _build(db_path: Path, script: str, version: int) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(script + VERSION_TABLE)
    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    conn.commit()
    conn.close()


def _recorded_version(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return int(conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0])
    finally:
        conn.close()


def _columns(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


class TestDuplicateColumnIsTheOnlyToleratedError:
    def test_recognises_sqlites_duplicate_column_message(self, tmp_path: Path):
        conn = sqlite3.connect(str(tmp_path / "x.db"))
        conn.execute("CREATE TABLE t (a TEXT)")
        with pytest.raises(sqlite3.OperationalError) as exc:
            conn.execute("ALTER TABLE t ADD COLUMN a TEXT")
        conn.close()
        assert _is_duplicate_column(exc.value) is True

    def test_does_not_recognise_a_missing_table(self, tmp_path: Path):
        conn = sqlite3.connect(str(tmp_path / "x.db"))
        with pytest.raises(sqlite3.OperationalError) as exc:
            conn.execute("ALTER TABLE nope ADD COLUMN a TEXT")
        conn.close()
        assert _is_duplicate_column(exc.value) is False


class TestHealthyV4Migrates:
    def test_v4_reaches_v5_and_gains_the_column(self, tmp_path: Path):
        db_path = tmp_path / "v4.db"
        _build(db_path, V4_OFFLINE_RECEIPTS, 4)

        with ReceiptStorage(db_path) as storage:
            storage._get_connection()

        assert "control_plane_json" in _columns(db_path, "offline_receipts")
        assert _recorded_version(db_path) == SCHEMA_VERSION

    def test_a_partly_applied_v4_migration_re_runs_cleanly(self, tmp_path: Path):
        """The column is already there — the one error a step may ignore."""
        db_path = tmp_path / "partial.db"
        _build(db_path, V4_WITH_V5_COLUMN, 4)
        assert "control_plane_json" in _columns(db_path, "offline_receipts")

        with ReceiptStorage(db_path) as storage:
            storage._get_connection()

        assert _recorded_version(db_path) == SCHEMA_VERSION

    def test_one_version_row_after_migrating(self, tmp_path: Path):
        db_path = tmp_path / "v4.db"
        _build(db_path, V4_OFFLINE_RECEIPTS, 4)

        with ReceiptStorage(db_path) as storage:
            storage._get_connection()

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT version FROM schema_version").fetchall()
        conn.close()
        assert rows == [(SCHEMA_VERSION,)]


class TestCorruptV4FailsLoudly:
    def test_missing_receipts_table_is_never_stamped_v5(self, tmp_path: Path):
        """A crafted v4: the version says 4, the table it names is not there."""
        db_path = tmp_path / "corrupt.db"
        _build(db_path, "CREATE TABLE unrelated (a TEXT);", 4)

        storage = ReceiptStorage(db_path)
        with pytest.raises(StorageMigrationError) as exc:
            storage._get_connection()

        assert "v4->v5" in str(exc.value)
        assert "offline_receipts" in str(exc.value)
        # The database still says what it actually is.
        assert _recorded_version(db_path) == 4

    def test_the_store_stays_shut_rather_than_serving_rows(self, tmp_path: Path):
        db_path = tmp_path / "corrupt.db"
        _build(db_path, "CREATE TABLE unrelated (a TEXT);", 4)

        storage = ReceiptStorage(db_path)
        with pytest.raises(StorageMigrationError):
            storage.get_receipt("oatt_anything")
        # And it does not quietly succeed on the second attempt either.
        with pytest.raises(StorageMigrationError):
            storage.count_receipts()
        assert _recorded_version(db_path) == 4

    def test_a_v3_database_without_a_hash_column_fails_by_name(self, tmp_path: Path):
        """Neither evidence_hash nor payload_hash: nothing to carry across."""
        db_path = tmp_path / "corrupt-v3.db"
        _build(
            db_path,
            """
            CREATE TABLE offline_receipts (
                attestation_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL
            );
            CREATE TABLE evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attestation_id TEXT NOT NULL
            );
            """,
            3,
        )

        storage = ReceiptStorage(db_path)
        with pytest.raises(StorageMigrationError) as exc:
            storage._get_connection()

        assert "v3->v4" in str(exc.value)
        assert _recorded_version(db_path) == 3

    def test_a_v2_database_without_the_evidence_table_fails_by_name(
        self, tmp_path: Path
    ):
        db_path = tmp_path / "corrupt-v2.db"
        _build(db_path, V4_OFFLINE_RECEIPTS, 2)

        storage = ReceiptStorage(db_path)
        with pytest.raises(StorageMigrationError) as exc:
            storage._get_connection()

        assert "v2->v3" in str(exc.value)
        assert _recorded_version(db_path) == 2

    def test_the_failure_says_nothing_was_stamped(self, tmp_path: Path):
        db_path = tmp_path / "corrupt.db"
        _build(db_path, "CREATE TABLE unrelated (a TEXT);", 4)

        storage = ReceiptStorage(db_path)
        with pytest.raises(StorageMigrationError) as exc:
            storage._get_connection()

        assert "nothing was stamped as migrated" in str(exc.value)


class TestFreshDatabaseIsUnaffected:
    def test_new_database_is_created_at_the_current_version(self, tmp_path: Path):
        db_path = tmp_path / "fresh.db"
        with ReceiptStorage(db_path) as storage:
            storage._get_connection()
        assert _recorded_version(db_path) == SCHEMA_VERSION
        assert "control_plane_json" in _columns(db_path, "offline_receipts")

    def test_reopening_a_current_database_does_not_migrate(self, tmp_path: Path):
        db_path = tmp_path / "fresh.db"
        with ReceiptStorage(db_path) as storage:
            storage._get_connection()
        with ReceiptStorage(db_path) as storage:
            storage._get_connection()
        assert _recorded_version(db_path) == SCHEMA_VERSION
