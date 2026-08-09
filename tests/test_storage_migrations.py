"""
Schema migrations either run or say they did not.

`_run_migrations()` used to wrap every step in `except sqlite3.OperationalError:
pass  # Column already exists` and then stamp version 5 unconditionally. Only
one of those errors means "already exists". A corrupt or crafted v4 database —
one whose `offline_receipts` table is missing, say — therefore came out the
other side labelled a successfully migrated v5 and was then read through a
schema it did not have.

Round 4 made every step fail loudly. That was still not enough, and pass 4 of
the Codex review showed why: a v4 database whose `offline_receipts` table held
**only** `control_plane_json` swallowed the duplicate-column error — because
that column really was already there — and finished stamped v5 with one column
to its name. A tolerated error proves one thing about one column. It is not
evidence that the schema is now what the version claims.

So the required shape is now stated per version in `REQUIRED_SCHEMA` and
validated against the live database after every step set, before anything is
stamped. These tests pin all three halves: a step that legitimately re-runs
reaches v5, a step that genuinely fails raises by name, and a shape that
survives the ALTERs but is not what the version means raises too.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from glacis.storage import (
    DECLARED_INDEXES,
    REQUIRED_SCHEMA,
    SCHEMA_VERSION,
    ReceiptStorage,
    StorageMigrationError,
    _is_duplicate_column,
)
from tests.conftest import V4_EVIDENCE_TABLE

# The receipt table on its own — a database that has it and nothing else. Not
# a v4: `evidence` arrived two migrations earlier.
RECEIPTS_TABLE_ONLY = """
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

# A database that really is v4: both tables, both shapes, the indexes the
# migration path creates.
V4_OFFLINE_RECEIPTS = RECEIPTS_TABLE_ONLY + V4_EVIDENCE_TABLE

# The same database with the v5 column already added: what a v4 looks like when
# the ALTER ran but the version stamp never did.
V4_WITH_V5_COLUMN = V4_OFFLINE_RECEIPTS.replace(
    "    cpr_hash TEXT\n);",
    "    cpr_hash TEXT,\n    control_plane_json TEXT\n);",
)

# Codex's pass-4 reproduction, exactly: a database declaring v4 whose
# `offline_receipts` has nothing but the column v4->v5 is there to add. The
# ALTER fails with "duplicate column name", which is the one error a step may
# ignore, so every step "succeeds" and nothing else about the table is true.
V4_ONLY_CONTROL_PLANE_JSON = """
CREATE TABLE offline_receipts (control_plane_json TEXT);
"""

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


def _indexes(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
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
        _build(db_path, RECEIPTS_TABLE_ONLY, 2)

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


class TestPostconditionsAreCheckedBeforeStamping:
    """A tolerated error is not proof the schema is what the version says.

    One test per version target, each built so that every migration step for
    that target *succeeds* and the result is still not the shape the version
    means. Without the postcondition these all end up stamped v5.
    """

    def test_codex_v4_with_only_control_plane_json_is_not_v5(self, tmp_path: Path):
        """The pass-4 reproduction, in full.

        `ALTER TABLE offline_receipts ADD COLUMN control_plane_json` raises
        "duplicate column name" — legitimately, the column is there — and the
        run used to finish with `version 5, columns ['control_plane_json']`.
        """
        db_path = tmp_path / "codex.db"
        _build(db_path, V4_ONLY_CONTROL_PLANE_JSON, 4)

        storage = ReceiptStorage(db_path)
        with pytest.raises(StorageMigrationError) as exc:
            storage._get_connection()

        message = str(exc.value)
        assert "schema postcondition for v5 failed" in message
        # It names what is missing rather than just refusing.
        assert "offline_receipts columns" in message
        assert "attestation_id" in message
        assert "signature" in message
        # And the database still says exactly what it is.
        assert _recorded_version(db_path) == 4
        assert _columns(db_path, "offline_receipts") == {"control_plane_json"}

    def test_v2_postcondition_catches_an_evidence_table_that_is_not_one(
        self, tmp_path: Path
    ):
        """`CREATE TABLE IF NOT EXISTS evidence` is a no-op on any table.

        A v1 database that already has something called `evidence` — with just
        enough columns for the four indexes to build — comes through the
        v1->v2 script without a single error, and is not v2.
        """
        db_path = tmp_path / "v1-fake-evidence.db"
        _build(
            db_path,
            V4_OFFLINE_RECEIPTS.split("CREATE TABLE evidence")[0]
            + """
            CREATE TABLE evidence (
                attestation_id TEXT NOT NULL,
                attestation_hash TEXT NOT NULL,
                service_id TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            """,
            1,
        )

        storage = ReceiptStorage(db_path)
        with pytest.raises(StorageMigrationError) as exc:
            storage._get_connection()

        message = str(exc.value)
        assert "schema postcondition for v2 failed" in message
        assert "evidence columns" in message
        assert "input_json" in message
        assert _recorded_version(db_path) == 1

    def test_v3_postcondition_catches_a_missing_receipts_table(self, tmp_path: Path):
        """The v2->v3 ALTER only touches `evidence`, so it cannot notice."""
        db_path = tmp_path / "v2-no-receipts.db"
        _build(
            db_path,
            V4_OFFLINE_RECEIPTS.replace("CREATE TABLE offline_receipts", "CREATE TABLE unused")
            .replace("ON offline_receipts", "ON unused")
            .replace("sampling_level TEXT NOT NULL DEFAULT 'L0',\n", ""),
            2,
        )

        storage = ReceiptStorage(db_path)
        with pytest.raises(StorageMigrationError) as exc:
            storage._get_connection()

        message = str(exc.value)
        assert "schema postcondition for v3 failed" in message
        assert "table offline_receipts" in message
        assert _recorded_version(db_path) == 2

    def test_v4_postcondition_catches_evidence_left_at_v2(self, tmp_path: Path):
        """v3->v4 only touches `offline_receipts`, and every ALTER succeeds.

        The database claims v3, so the step that adds `sampling_level` never
        runs. Nothing in the v4 path would ever look at `evidence`.
        """
        db_path = tmp_path / "v3-stale-evidence.db"
        _build(
            db_path,
            V4_OFFLINE_RECEIPTS.replace(
                "    sampling_level TEXT NOT NULL DEFAULT 'L0',\n", ""
            ),
            3,
        )

        storage = ReceiptStorage(db_path)
        with pytest.raises(StorageMigrationError) as exc:
            storage._get_connection()

        message = str(exc.value)
        assert "schema postcondition for v4 failed" in message
        assert "evidence columns sampling_level" in message
        assert _recorded_version(db_path) == 3

    def test_the_postcondition_failure_says_nothing_was_stamped(self, tmp_path: Path):
        db_path = tmp_path / "codex.db"
        _build(db_path, V4_ONLY_CONTROL_PLANE_JSON, 4)

        storage = ReceiptStorage(db_path)
        with pytest.raises(StorageMigrationError) as exc:
            storage._get_connection()

        assert "nothing was stamped as migrated" in str(exc.value)

    def test_the_store_stays_shut_after_a_postcondition_failure(self, tmp_path: Path):
        db_path = tmp_path / "codex.db"
        _build(db_path, V4_ONLY_CONTROL_PLANE_JSON, 4)

        storage = ReceiptStorage(db_path)
        with pytest.raises(StorageMigrationError):
            storage.get_receipt("oatt_anything")
        with pytest.raises(StorageMigrationError):
            storage.count_receipts()
        assert _recorded_version(db_path) == 4

    def test_a_migrated_database_ends_up_with_every_declared_index(
        self, tmp_path: Path
    ):
        """The three convenience indexes no version step creates, too."""
        db_path = tmp_path / "v4.db"
        _build(db_path, V4_OFFLINE_RECEIPTS, 4)

        with ReceiptStorage(db_path) as storage:
            storage._get_connection()

        assert DECLARED_INDEXES <= _indexes(db_path)

    def test_required_schema_covers_every_version_the_migrations_target(self):
        """No version target may reach the stamp without a stated shape."""
        assert set(REQUIRED_SCHEMA) == {2, 3, 4, SCHEMA_VERSION}
        for version, (tables, _indexes_required) in REQUIRED_SCHEMA.items():
            assert "schema_version" in tables, version
            assert "offline_receipts" in tables, version
            assert "evidence" in tables, version


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
