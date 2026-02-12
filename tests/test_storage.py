"""
Tests for SQLite storage (glacis/storage.py).
"""

import time
from datetime import datetime
from pathlib import Path

from glacis.models import OfflineAttestReceipt
from glacis.storage import SCHEMA_VERSION, ReceiptStorage


class TestStorageInit:
    """Tests for storage initialization."""

    def test_storage_creates_database(self, temp_db_path: Path):
        """Storage auto-creates database file."""
        assert not temp_db_path.exists()

        with ReceiptStorage(temp_db_path) as storage:
            # Trigger connection
            storage._get_connection()

        assert temp_db_path.exists()

    def test_storage_creates_parent_dirs(self, tmp_path: Path):
        """Storage creates parent directories."""
        deep_path = tmp_path / "nested" / "dirs" / "test.db"
        assert not deep_path.parent.exists()

        with ReceiptStorage(deep_path) as storage:
            storage._get_connection()

        assert deep_path.parent.exists()

    def test_storage_schema_version(self, temp_db_path: Path):
        """Storage initializes with correct schema version."""
        with ReceiptStorage(temp_db_path) as storage:
            conn = storage._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT version FROM schema_version")
            row = cursor.fetchone()
            assert row[0] == SCHEMA_VERSION


class TestReceiptStorage:
    """Tests for receipt CRUD operations."""

    def _create_receipt(
        self,
        attestation_id: str = "oatt_test123",
        service_id: str = "test-service",
    ) -> OfflineAttestReceipt:
        """Helper to create a receipt."""
        return OfflineAttestReceipt(
            id=attestation_id,
            evidence_hash="a" * 64,
            timestamp=int(datetime.utcnow().timestamp() * 1000),  # Unix ms
            service_id=service_id,
            operation_type="inference",
            payload_hash="a" * 64,
            signature="b" * 128,
            public_key="c" * 64,
        )

    def test_store_and_retrieve_receipt(self, temp_db_path: Path):
        """Store and retrieve a receipt."""
        receipt = self._create_receipt()

        with ReceiptStorage(temp_db_path) as storage:
            storage.store_receipt(receipt, input_preview="Hello", output_preview="Hi")

            retrieved = storage.get_receipt(receipt.id)

        assert retrieved is not None
        assert retrieved.id == receipt.id
        assert retrieved.payload_hash == receipt.payload_hash
        assert retrieved.service_id == receipt.service_id

    def test_get_receipt_not_found(self, temp_db_path: Path):
        """Get nonexistent receipt returns None."""
        with ReceiptStorage(temp_db_path) as storage:
            result = storage.get_receipt("oatt_nonexistent")

        assert result is None

    def test_get_last_receipt(self, temp_db_path: Path):
        """get_last_receipt returns most recent."""
        with ReceiptStorage(temp_db_path) as storage:
            # Store receipts with slight delay
            receipt1 = self._create_receipt("oatt_first")
            storage.store_receipt(receipt1)
            time.sleep(0.01)  # Ensure different timestamps

            receipt2 = self._create_receipt("oatt_second")
            storage.store_receipt(receipt2)

            last = storage.get_last_receipt()

        assert last is not None
        assert last.id == "oatt_second"

    def test_get_last_receipt_empty(self, temp_db_path: Path):
        """get_last_receipt returns None when empty."""
        with ReceiptStorage(temp_db_path) as storage:
            result = storage.get_last_receipt()

        assert result is None

    def test_query_receipts_by_service(self, temp_db_path: Path):
        """Query receipts filtered by service_id."""
        with ReceiptStorage(temp_db_path) as storage:
            storage.store_receipt(self._create_receipt("oatt_1", service_id="svc-a"))
            storage.store_receipt(self._create_receipt("oatt_2", service_id="svc-b"))
            storage.store_receipt(self._create_receipt("oatt_3", service_id="svc-a"))

            results = storage.query_receipts(service_id="svc-a")

        assert len(results) == 2
        assert all(r.service_id == "svc-a" for r in results)

    def test_query_receipts_limit(self, temp_db_path: Path):
        """Query receipts respects limit."""
        with ReceiptStorage(temp_db_path) as storage:
            for i in range(10):
                storage.store_receipt(self._create_receipt(f"oatt_{i}"))

            results = storage.query_receipts(limit=3)

        assert len(results) == 3

    def test_count_receipts(self, temp_db_path: Path):
        """Count receipts with and without filter."""
        with ReceiptStorage(temp_db_path) as storage:
            storage.store_receipt(self._create_receipt("oatt_1", service_id="svc-a"))
            storage.store_receipt(self._create_receipt("oatt_2", service_id="svc-b"))
            storage.store_receipt(self._create_receipt("oatt_3", service_id="svc-a"))

            total = storage.count_receipts()
            svc_a_count = storage.count_receipts(service_id="svc-a")

        assert total == 3
        assert svc_a_count == 2

    def test_delete_receipt(self, temp_db_path: Path):
        """Delete a receipt."""
        receipt = self._create_receipt()

        with ReceiptStorage(temp_db_path) as storage:
            storage.store_receipt(receipt)
            assert storage.get_receipt(receipt.id) is not None

            deleted = storage.delete_receipt(receipt.id)
            assert deleted is True
            assert storage.get_receipt(receipt.id) is None

    def test_delete_receipt_not_found(self, temp_db_path: Path):
        """Delete nonexistent receipt returns False."""
        with ReceiptStorage(temp_db_path) as storage:
            deleted = storage.delete_receipt("oatt_nonexistent")

        assert deleted is False


class TestEvidenceStorage:
    """Tests for evidence storage."""

    def test_store_and_get_evidence(self, temp_db_path: Path):
        """Store and retrieve evidence."""
        with ReceiptStorage(temp_db_path) as storage:
            storage.store_evidence(
                attestation_id="att_test123",
                attestation_hash="a" * 64,
                mode="online",
                service_id="test-service",
                operation_type="inference",
                timestamp=datetime.utcnow().isoformat() + "Z",
                input_data={"prompt": "Hello, world!"},
                output_data={"response": "Hi there!"},
                metadata={"model": "gpt-4"},
            )

            evidence = storage.get_evidence("att_test123")

        assert evidence is not None
        assert evidence["attestation_id"] == "att_test123"
        assert evidence["mode"] == "online"
        assert evidence["input"] == {"prompt": "Hello, world!"}
        assert evidence["output"] == {"response": "Hi there!"}
        assert evidence["metadata"] == {"model": "gpt-4"}

    def test_get_evidence_by_hash(self, temp_db_path: Path):
        """Retrieve evidence by attestation hash."""
        hash_value = "b" * 64

        with ReceiptStorage(temp_db_path) as storage:
            storage.store_evidence(
                attestation_id="att_byhash",
                attestation_hash=hash_value,
                mode="online",
                service_id="test",
                operation_type="inference",
                timestamp=datetime.utcnow().isoformat() + "Z",
                input_data={"x": 1},
                output_data={"y": 2},
            )

            evidence = storage.get_evidence_by_hash(hash_value)

        assert evidence is not None
        assert evidence["attestation_hash"] == hash_value

    def test_evidence_not_found(self, temp_db_path: Path):
        """Get nonexistent evidence returns None."""
        with ReceiptStorage(temp_db_path) as storage:
            result = storage.get_evidence("att_nonexistent")

        assert result is None

    def test_query_evidence_by_mode(self, temp_db_path: Path):
        """Query evidence filtered by mode."""
        with ReceiptStorage(temp_db_path) as storage:
            storage.store_evidence(
                attestation_id="att_online",
                attestation_hash="a" * 64,
                mode="online",
                service_id="test",
                operation_type="inference",
                timestamp=datetime.utcnow().isoformat() + "Z",
                input_data={},
                output_data={},
            )
            storage.store_evidence(
                attestation_id="oatt_offline",
                attestation_hash="b" * 64,
                mode="offline",
                service_id="test",
                operation_type="inference",
                timestamp=datetime.utcnow().isoformat() + "Z",
                input_data={},
                output_data={},
            )

            online_results = storage.query_evidence(mode="online")
            offline_results = storage.query_evidence(mode="offline")

        assert len(online_results) == 1
        assert online_results[0]["mode"] == "online"
        assert len(offline_results) == 1
        assert offline_results[0]["mode"] == "offline"

    def test_count_evidence(self, temp_db_path: Path):
        """Count evidence with filters."""
        with ReceiptStorage(temp_db_path) as storage:
            for i in range(5):
                storage.store_evidence(
                    attestation_id=f"att_{i}",
                    attestation_hash=f"{i}" * 64,
                    mode="online" if i % 2 == 0 else "offline",
                    service_id="test",
                    operation_type="inference",
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    input_data={},
                    output_data={},
                )

            total = storage.count_evidence()
            online_count = storage.count_evidence(mode="online")

        assert total == 5
        assert online_count == 3  # 0, 2, 4

    def test_evidence_stores_full_payload(self, temp_db_path: Path):
        """Evidence preserves full input/output structure."""
        complex_input = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
            "model": "gpt-4",
            "temperature": 0.7,
        }
        complex_output = {
            "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with ReceiptStorage(temp_db_path) as storage:
            storage.store_evidence(
                attestation_id="att_complex",
                attestation_hash="c" * 64,
                mode="online",
                service_id="test",
                operation_type="inference",
                timestamp=datetime.utcnow().isoformat() + "Z",
                input_data=complex_input,
                output_data=complex_output,
            )

            evidence = storage.get_evidence("att_complex")

        assert evidence["input"] == complex_input
        assert evidence["output"] == complex_output

    def test_evidence_upsert(self, temp_db_path: Path):
        """Evidence can be updated (upsert behavior)."""
        with ReceiptStorage(temp_db_path) as storage:
            # Store initial
            storage.store_evidence(
                attestation_id="att_upsert",
                attestation_hash="a" * 64,
                mode="online",
                service_id="test",
                operation_type="inference",
                timestamp=datetime.utcnow().isoformat() + "Z",
                input_data={"version": 1},
                output_data={},
            )

            # Update
            storage.store_evidence(
                attestation_id="att_upsert",
                attestation_hash="a" * 64,
                mode="online",
                service_id="test",
                operation_type="inference",
                timestamp=datetime.utcnow().isoformat() + "Z",
                input_data={"version": 2},
                output_data={},
            )

            evidence = storage.get_evidence("att_upsert")

        assert evidence["input"]["version"] == 2


class TestContextManager:
    """Tests for context manager."""

    def test_context_manager_closes(self, temp_db_path: Path):
        """Context manager properly closes connection."""
        storage = ReceiptStorage(temp_db_path)
        storage._get_connection()
        assert storage._conn is not None

        storage.__exit__(None, None, None)
        assert storage._conn is None
