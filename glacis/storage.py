"""
SQLite storage for attestation receipts and evidence.

Stores local attestation receipts and full evidence in ~/.glacis/glacis.db
for persistence, audit trails, and later verification.

Evidence (input, output, control_plane_results) is stored locally for zero-egress
compliance - only hashes are sent to the GLACIS server.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from glacis.models import ControlPlaneAttestation, OfflineAttestReceipt

DEFAULT_DB_PATH = Path.home() / ".glacis" / "glacis.db"

SCHEMA_VERSION = 2

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
            # check_same_thread=False allows the connection to be used across threads
            # This is safe because we're only doing simple CRUD operations
            # and SQLite handles locking internally
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
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
        conn = self._conn
        if conn is None:
            return

        cursor = conn.cursor()

        # Migration from v1 to v2: Add evidence table
        if from_version < 2:
            cursor.executescript(MIGRATION_V1_TO_V2)

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
        timestamp: str,
        input_data: Any,
        output_data: Any,
        control_plane_results: Optional["ControlPlaneAttestation"] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Store full evidence for an attestation.

        This stores the complete input, output, and control plane results locally
        for audit trails and dispute resolution. Only the hash was sent to GLACIS.

        Args:
            attestation_id: The attestation ID (att_xxx or oatt_xxx)
            attestation_hash: The hash that was attested (payload_hash)
            mode: 'online' or 'offline'
            service_id: Service identifier
            operation_type: Type of operation
            timestamp: ISO 8601 timestamp
            input_data: Full input data (will be JSON serialized)
            output_data: Full output data (will be JSON serialized)
            control_plane_results: Optional control plane attestation
            metadata: Optional metadata dict
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        control_plane_json = None
        if control_plane_results:
            control_plane_json = json.dumps(
                control_plane_results.model_dump(by_alias=True)
            )

        cursor.execute(
            """
            INSERT OR REPLACE INTO evidence
            (attestation_id, attestation_hash, mode, service_id, operation_type,
             timestamp, created_at, input_json, output_json, control_plane_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attestation_id,
                attestation_hash,
                mode,
                service_id,
                operation_type,
                timestamp,
                datetime.utcnow().isoformat() + "Z",
                json.dumps(input_data),
                json.dumps(output_data),
                control_plane_json,
                json.dumps(metadata) if metadata else None,
            ),
        )
        conn.commit()

    def get_evidence(self, attestation_id: str) -> Optional[dict[str, Any]]:
        """
        Retrieve full evidence by attestation ID.

        Args:
            attestation_id: The attestation ID (att_xxx or oatt_xxx)

        Returns:
            Dict with input, output, control_plane_results, and metadata,
            or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM evidence WHERE attestation_id = ?",
            (attestation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        result: dict[str, Any] = {
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
        }
        return result

    def get_evidence_by_hash(self, attestation_hash: str) -> Optional[dict[str, Any]]:
        """
        Retrieve full evidence by attestation hash.

        Useful for verifying that stored evidence matches the attested hash.

        Args:
            attestation_hash: The payload hash that was attested

        Returns:
            Dict with input, output, control_plane_results, and metadata,
            or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM evidence WHERE attestation_hash = ?",
            (attestation_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        result: dict[str, Any] = {
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
        }
        return result

    def query_evidence(
        self,
        service_id: Optional[str] = None,
        mode: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Query evidence with optional filters.

        Args:
            service_id: Filter by service ID
            mode: Filter by mode ('online' or 'offline')
            start: Filter by timestamp >= start (ISO 8601)
            end: Filter by timestamp <= end (ISO 8601)
            limit: Maximum number of results (default 50)

        Returns:
            List of evidence records
        """
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

        results = []
        for row in rows:
            results.append({
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
            })
        return results

    def count_evidence(
        self, service_id: Optional[str] = None, mode: Optional[str] = None
    ) -> int:
        """
        Count evidence records.

        Args:
            service_id: Optional service ID filter
            mode: Optional mode filter ('online' or 'offline')

        Returns:
            Number of matching evidence records
        """
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

    # NOTE: Evidence is intentionally append-only (no delete method).
    # For a compliant audit trail, evidence must be immutable.
    # Evidence deletion should only happen through:
    # 1. Data retention policies (automated, time-based)
    # 2. Explicit admin/compliance operations
    # 3. Legal requirements (GDPR right to be forgotten)
    # These operations should be logged and audited separately.
