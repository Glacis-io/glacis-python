"""
SQLite storage for offline receipts.

Stores local attestation receipts in ~/.glacis/receipts.db for persistence
and later verification.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from glacis.models import OfflineAttestReceipt

DEFAULT_DB_PATH = Path.home() / ".glacis" / "receipts.db"

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS offline_receipts (
    attestation_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    service_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    signature TEXT NOT NULL,
    public_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    input_preview TEXT,
    output_preview TEXT,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_service_id ON offline_receipts(service_id);
CREATE INDEX IF NOT EXISTS idx_timestamp ON offline_receipts(timestamp);
CREATE INDEX IF NOT EXISTS idx_payload_hash ON offline_receipts(payload_hash);
CREATE INDEX IF NOT EXISTS idx_created_at ON offline_receipts(created_at);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""


class ReceiptStorage:
    """
    SQLite storage for offline attestation receipts.

    Default location: ~/.glacis/receipts.db
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Initialize the receipt storage.

        Args:
            db_path: Path to SQLite database file. Defaults to ~/.glacis/receipts.db
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        """Initialize database schema if needed."""
        conn = self._conn
        if conn is None:
            return

        cursor = conn.cursor()

        # Check if schema_version table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        if cursor.fetchone() is None:
            # Fresh database - create schema
            cursor.executescript(SCHEMA)
            cursor.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            conn.commit()
        else:
            # Check version for migrations
            cursor.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()
            if row is None or row[0] < SCHEMA_VERSION:
                # Run migrations if needed
                self._run_migrations(row[0] if row else 0)

    def _run_migrations(self, from_version: int) -> None:
        """Run schema migrations."""
        # No migrations needed yet since this is v1
        conn = self._conn
        if conn is None:
            return

        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
        conn.commit()

    def store_receipt(
        self,
        receipt: "OfflineAttestReceipt",
        input_preview: Optional[str] = None,
        output_preview: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Store an offline receipt.

        Args:
            receipt: The offline attestation receipt to store
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
             payload_hash, signature, public_key, created_at,
             input_preview, output_preview, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.attestation_id,
                receipt.timestamp,
                receipt.service_id,
                receipt.operation_type,
                receipt.payload_hash,
                receipt.signature,
                receipt.public_key,
                datetime.utcnow().isoformat() + "Z",
                input_preview[:100] if input_preview else None,
                output_preview[:100] if output_preview else None,
                json.dumps(metadata) if metadata else None,
            ),
        )
        conn.commit()

    def get_receipt(self, attestation_id: str) -> Optional["OfflineAttestReceipt"]:
        """
        Retrieve a receipt by ID.

        Args:
            attestation_id: The attestation ID (oatt_xxx)

        Returns:
            The receipt if found, None otherwise
        """
        from glacis.models import OfflineAttestReceipt

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM offline_receipts WHERE attestation_id = ?",
            (attestation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return OfflineAttestReceipt(
            attestation_id=row["attestation_id"],
            timestamp=row["timestamp"],
            service_id=row["service_id"],
            operation_type=row["operation_type"],
            payload_hash=row["payload_hash"],
            signature=row["signature"],
            public_key=row["public_key"],
        )

    def get_last_receipt(self) -> Optional["OfflineAttestReceipt"]:
        """
        Get the most recently created receipt.

        Returns:
            The most recent receipt, or None if no receipts exist
        """
        from glacis.models import OfflineAttestReceipt

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM offline_receipts ORDER BY created_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return OfflineAttestReceipt(
            attestation_id=row["attestation_id"],
            timestamp=row["timestamp"],
            service_id=row["service_id"],
            operation_type=row["operation_type"],
            payload_hash=row["payload_hash"],
            signature=row["signature"],
            public_key=row["public_key"],
        )

    def query_receipts(
        self,
        service_id: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 50,
    ) -> list["OfflineAttestReceipt"]:
        """
        Query receipts with optional filters.

        Args:
            service_id: Filter by service ID
            start: Filter by timestamp >= start (ISO 8601)
            end: Filter by timestamp <= end (ISO 8601)
            limit: Maximum number of results (default 50)

        Returns:
            List of matching receipts
        """
        from glacis.models import OfflineAttestReceipt

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

        return [
            OfflineAttestReceipt(
                attestation_id=row["attestation_id"],
                timestamp=row["timestamp"],
                service_id=row["service_id"],
                operation_type=row["operation_type"],
                payload_hash=row["payload_hash"],
                signature=row["signature"],
                public_key=row["public_key"],
            )
            for row in rows
        ]

    def count_receipts(self, service_id: Optional[str] = None) -> int:
        """
        Count receipts, optionally filtered by service ID.

        Args:
            service_id: Optional service ID filter

        Returns:
            Number of matching receipts
        """
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
        """
        Delete a receipt by ID.

        Args:
            attestation_id: The attestation ID to delete

        Returns:
            True if a receipt was deleted, False otherwise
        """
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
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Context manager exit."""
        self.close()
