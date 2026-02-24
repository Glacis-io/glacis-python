"""
Tests for storage backends (glacis/storage.py).
"""

import time
from datetime import datetime, timezone
from pathlib import Path

from glacis.models import Attestation
from glacis.storage import (
    SCHEMA_VERSION,
    JsonStorageBackend,
    ReceiptStorage,
    StorageBackend,
    create_storage,
)


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

    def _create_attestation(
        self,
        attestation_id: str = "oatt_test123",
        service_id: str = "test-service",
    ) -> Attestation:
        """Helper to create an Attestation."""
        return Attestation(
            id=attestation_id,
            evidence_hash="a" * 64,
            timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
            service_id=service_id,
            operation_type="inference",
            signature="b" * 128,
            public_key="c" * 64,
            is_offline=True,
        )

    def test_store_and_retrieve_receipt(self, temp_db_path: Path):
        """Store and retrieve an attestation."""
        receipt = self._create_attestation()

        with ReceiptStorage(temp_db_path) as storage:
            storage.store_receipt(receipt, input_preview="Hello", output_preview="Hi")

            retrieved = storage.get_receipt(receipt.id)

        assert retrieved is not None
        assert retrieved.id == receipt.id
        assert retrieved.evidence_hash == receipt.evidence_hash
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
            receipt1 = self._create_attestation("oatt_first")
            storage.store_receipt(receipt1)
            time.sleep(0.01)  # Ensure different timestamps

            receipt2 = self._create_attestation("oatt_second")
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
            storage.store_receipt(self._create_attestation("oatt_1", service_id="svc-a"))
            storage.store_receipt(self._create_attestation("oatt_2", service_id="svc-b"))
            storage.store_receipt(self._create_attestation("oatt_3", service_id="svc-a"))

            results = storage.query_receipts(service_id="svc-a")

        assert len(results) == 2
        assert all(r.service_id == "svc-a" for r in results)

    def test_query_receipts_limit(self, temp_db_path: Path):
        """Query receipts respects limit."""
        with ReceiptStorage(temp_db_path) as storage:
            for i in range(10):
                storage.store_receipt(self._create_attestation(f"oatt_{i}"))

            results = storage.query_receipts(limit=3)

        assert len(results) == 3

    def test_count_receipts(self, temp_db_path: Path):
        """Count receipts with and without filter."""
        with ReceiptStorage(temp_db_path) as storage:
            storage.store_receipt(self._create_attestation("oatt_1", service_id="svc-a"))
            storage.store_receipt(self._create_attestation("oatt_2", service_id="svc-b"))
            storage.store_receipt(self._create_attestation("oatt_3", service_id="svc-a"))

            total = storage.count_receipts()
            svc_a_count = storage.count_receipts(service_id="svc-a")

        assert total == 3
        assert svc_a_count == 2

    def test_delete_receipt(self, temp_db_path: Path):
        """Delete a receipt."""
        receipt = self._create_attestation()

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

    def test_stored_receipt_is_offline(self, temp_db_path: Path):
        """Retrieved attestation has is_offline=True."""
        receipt = self._create_attestation()

        with ReceiptStorage(temp_db_path) as storage:
            storage.store_receipt(receipt)
            retrieved = storage.get_receipt(receipt.id)

        assert retrieved is not None
        assert retrieved.is_offline is True

    def test_operation_id_persisted(self, temp_db_path: Path):
        """operation_id is stored and retrieved."""
        att = Attestation(
            id="oatt_op_test",
            evidence_hash="a" * 64,
            timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
            service_id="test",
            operation_type="inference",
            signature="b" * 128,
            public_key="c" * 64,
            is_offline=True,
            operation_id="op_12345",
            operation_sequence=3,
        )

        with ReceiptStorage(temp_db_path) as storage:
            storage.store_receipt(att)
            retrieved = storage.get_receipt(att.id)

        assert retrieved is not None
        assert retrieved.operation_id == "op_12345"
        assert retrieved.operation_sequence == 3


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
                timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
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
                timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
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
                timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
                input_data={},
                output_data={},
            )
            storage.store_evidence(
                attestation_id="oatt_offline",
                attestation_hash="b" * 64,
                mode="offline",
                service_id="test",
                operation_type="inference",
                timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
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
                    timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
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
                timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
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
                timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
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
                timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
                input_data={"version": 2},
                output_data={},
            )

            evidence = storage.get_evidence("att_upsert")

        assert evidence["input"]["version"] == 2

    def test_evidence_stores_control_plane_results(self, temp_db_path: Path):
        """Evidence stores CPR as JSON."""
        cpr = {"schema_version": "1.0", "determination": {"action": "forwarded"}}

        with ReceiptStorage(temp_db_path) as storage:
            storage.store_evidence(
                attestation_id="att_cpr_test",
                attestation_hash="a" * 64,
                mode="online",
                service_id="test",
                operation_type="inference",
                timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
                input_data={},
                output_data={},
                control_plane_results=cpr,
            )

            evidence = storage.get_evidence("att_cpr_test")

        assert evidence["control_plane_results"] == cpr


class TestContextManager:
    """Tests for context manager."""

    def test_context_manager_closes(self, temp_db_path: Path):
        """Context manager properly closes connection."""
        storage = ReceiptStorage(temp_db_path)
        storage._get_connection()
        assert storage._conn is not None

        storage.__exit__(None, None, None)
        assert storage._conn is None


# ==============================================================================
# JSON Storage Backend Tests
# ==============================================================================


class TestJsonStorageInit:
    """Tests for JSON storage initialization."""

    def test_creates_base_directory(self, tmp_path: Path):
        """JSON backend creates the base directory."""
        base = tmp_path / "glacis_json"
        storage = JsonStorageBackend(base)

        assert base.is_dir()

    def test_context_manager(self, tmp_path: Path):
        """Context manager works (close is a no-op)."""
        with JsonStorageBackend(tmp_path / "json") as storage:
            assert storage is not None


class TestJsonReceiptStorage:
    """Tests for JSON receipt CRUD operations."""

    def _create_attestation(
        self,
        attestation_id: str = "oatt_test123",
        service_id: str = "test-service",
    ) -> Attestation:
        return Attestation(
            id=attestation_id,
            evidence_hash="a" * 64,
            timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
            service_id=service_id,
            operation_type="inference",
            signature="b" * 128,
            public_key="c" * 64,
            is_offline=True,
        )

    def test_store_and_retrieve_receipt(self, tmp_path: Path):
        """Store and retrieve an attestation."""
        storage = JsonStorageBackend(tmp_path)
        receipt = self._create_attestation()

        storage.store_receipt(receipt, input_preview="Hello", output_preview="Hi")
        retrieved = storage.get_receipt(receipt.id)

        assert retrieved is not None
        assert retrieved.id == receipt.id
        assert retrieved.evidence_hash == receipt.evidence_hash
        assert retrieved.service_id == receipt.service_id

    def test_receipt_written_to_jsonl_file(self, tmp_path: Path):
        """Receipt is appended to the JSONL file."""
        storage = JsonStorageBackend(tmp_path)
        receipt = self._create_attestation("oatt_filetest")

        storage.store_receipt(receipt)

        jsonl_path = tmp_path / "receipts.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        import json
        data = json.loads(lines[0])
        assert data["attestation_id"] == "oatt_filetest"

    def test_get_receipt_not_found(self, tmp_path: Path):
        """Get nonexistent receipt returns None."""
        storage = JsonStorageBackend(tmp_path)
        assert storage.get_receipt("oatt_nonexistent") is None

    def test_get_last_receipt(self, tmp_path: Path):
        """get_last_receipt returns most recent."""
        storage = JsonStorageBackend(tmp_path)

        r1 = self._create_attestation("oatt_first")
        storage.store_receipt(r1)
        time.sleep(0.01)

        r2 = self._create_attestation("oatt_second")
        storage.store_receipt(r2)

        last = storage.get_last_receipt()
        assert last is not None
        assert last.id == "oatt_second"

    def test_get_last_receipt_empty(self, tmp_path: Path):
        """get_last_receipt returns None when empty."""
        storage = JsonStorageBackend(tmp_path)
        assert storage.get_last_receipt() is None

    def test_query_receipts_by_service(self, tmp_path: Path):
        """Query receipts filtered by service_id."""
        storage = JsonStorageBackend(tmp_path)
        storage.store_receipt(self._create_attestation("oatt_1", service_id="svc-a"))
        storage.store_receipt(self._create_attestation("oatt_2", service_id="svc-b"))
        storage.store_receipt(self._create_attestation("oatt_3", service_id="svc-a"))

        results = storage.query_receipts(service_id="svc-a")
        assert len(results) == 2
        assert all(r.service_id == "svc-a" for r in results)

    def test_query_receipts_limit(self, tmp_path: Path):
        """Query receipts respects limit."""
        storage = JsonStorageBackend(tmp_path)
        for i in range(10):
            storage.store_receipt(self._create_attestation(f"oatt_{i}"))

        results = storage.query_receipts(limit=3)
        assert len(results) == 3

    def test_count_receipts(self, tmp_path: Path):
        """Count receipts with and without filter."""
        storage = JsonStorageBackend(tmp_path)
        storage.store_receipt(self._create_attestation("oatt_1", service_id="svc-a"))
        storage.store_receipt(self._create_attestation("oatt_2", service_id="svc-b"))
        storage.store_receipt(self._create_attestation("oatt_3", service_id="svc-a"))

        assert storage.count_receipts() == 3
        assert storage.count_receipts(service_id="svc-a") == 2

    def test_delete_receipt(self, tmp_path: Path):
        """Delete a receipt."""
        storage = JsonStorageBackend(tmp_path)
        receipt = self._create_attestation()

        storage.store_receipt(receipt)
        assert storage.get_receipt(receipt.id) is not None

        assert storage.delete_receipt(receipt.id) is True
        assert storage.get_receipt(receipt.id) is None

    def test_delete_receipt_not_found(self, tmp_path: Path):
        """Delete nonexistent receipt returns False."""
        storage = JsonStorageBackend(tmp_path)
        assert storage.delete_receipt("oatt_nonexistent") is False

    def test_stored_receipt_is_offline(self, tmp_path: Path):
        """Retrieved attestation has is_offline=True."""
        storage = JsonStorageBackend(tmp_path)
        receipt = self._create_attestation()

        storage.store_receipt(receipt)
        retrieved = storage.get_receipt(receipt.id)

        assert retrieved is not None
        assert retrieved.is_offline is True

    def test_operation_id_persisted(self, tmp_path: Path):
        """operation_id is stored and retrieved."""
        storage = JsonStorageBackend(tmp_path)
        att = Attestation(
            id="oatt_op_test",
            evidence_hash="a" * 64,
            timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
            service_id="test",
            operation_type="inference",
            signature="b" * 128,
            public_key="c" * 64,
            is_offline=True,
            operation_id="op_12345",
            operation_sequence=3,
        )

        storage.store_receipt(att)
        retrieved = storage.get_receipt(att.id)

        assert retrieved is not None
        assert retrieved.operation_id == "op_12345"
        assert retrieved.operation_sequence == 3


class TestJsonEvidenceStorage:
    """Tests for JSON evidence storage."""

    def test_store_and_get_evidence(self, tmp_path: Path):
        """Store and retrieve evidence."""
        storage = JsonStorageBackend(tmp_path)
        storage.store_evidence(
            attestation_id="att_test123",
            attestation_hash="a" * 64,
            mode="online",
            service_id="test-service",
            operation_type="inference",
            timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
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

    def test_evidence_written_to_jsonl_file(self, tmp_path: Path):
        """Evidence is appended to the JSONL file."""
        storage = JsonStorageBackend(tmp_path)
        storage.store_evidence(
            attestation_id="att_filetest",
            attestation_hash="a" * 64,
            mode="online",
            service_id="test",
            operation_type="inference",
            timestamp=1000,
            input_data={},
            output_data={},
        )

        jsonl_path = tmp_path / "evidence.jsonl"
        assert jsonl_path.exists()
        import json
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["attestation_id"] == "att_filetest"

    def test_get_evidence_by_hash(self, tmp_path: Path):
        """Retrieve evidence by attestation hash."""
        storage = JsonStorageBackend(tmp_path)
        hash_value = "b" * 64

        storage.store_evidence(
            attestation_id="att_byhash",
            attestation_hash=hash_value,
            mode="online",
            service_id="test",
            operation_type="inference",
            timestamp=1000,
            input_data={"x": 1},
            output_data={"y": 2},
        )

        evidence = storage.get_evidence_by_hash(hash_value)
        assert evidence is not None
        assert evidence["attestation_hash"] == hash_value

    def test_evidence_not_found(self, tmp_path: Path):
        """Get nonexistent evidence returns None."""
        storage = JsonStorageBackend(tmp_path)
        assert storage.get_evidence("att_nonexistent") is None

    def test_query_evidence_by_mode(self, tmp_path: Path):
        """Query evidence filtered by mode."""
        storage = JsonStorageBackend(tmp_path)
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)

        storage.store_evidence(
            attestation_id="att_online",
            attestation_hash="a" * 64,
            mode="online",
            service_id="test",
            operation_type="inference",
            timestamp=ts,
            input_data={},
            output_data={},
        )
        storage.store_evidence(
            attestation_id="oatt_offline",
            attestation_hash="b" * 64,
            mode="offline",
            service_id="test",
            operation_type="inference",
            timestamp=ts,
            input_data={},
            output_data={},
        )

        assert len(storage.query_evidence(mode="online")) == 1
        assert len(storage.query_evidence(mode="offline")) == 1

    def test_count_evidence(self, tmp_path: Path):
        """Count evidence with filters."""
        storage = JsonStorageBackend(tmp_path)
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)

        for i in range(5):
            storage.store_evidence(
                attestation_id=f"att_{i}",
                attestation_hash=f"{i}" * 64,
                mode="online" if i % 2 == 0 else "offline",
                service_id="test",
                operation_type="inference",
                timestamp=ts,
                input_data={},
                output_data={},
            )

        assert storage.count_evidence() == 5
        assert storage.count_evidence(mode="online") == 3

    def test_evidence_stores_control_plane_results(self, tmp_path: Path):
        """Evidence stores CPR."""
        storage = JsonStorageBackend(tmp_path)
        cpr = {"schema_version": "1.0", "determination": {"action": "forwarded"}}

        storage.store_evidence(
            attestation_id="att_cpr_test",
            attestation_hash="a" * 64,
            mode="online",
            service_id="test",
            operation_type="inference",
            timestamp=1000,
            input_data={},
            output_data={},
            control_plane_results=cpr,
        )

        evidence = storage.get_evidence("att_cpr_test")
        assert evidence["control_plane_results"] == cpr

    def test_evidence_upsert(self, tmp_path: Path):
        """Evidence can be overwritten (upsert behavior)."""
        storage = JsonStorageBackend(tmp_path)

        storage.store_evidence(
            attestation_id="att_upsert",
            attestation_hash="a" * 64,
            mode="online",
            service_id="test",
            operation_type="inference",
            timestamp=1000,
            input_data={"version": 1},
            output_data={},
        )
        storage.store_evidence(
            attestation_id="att_upsert",
            attestation_hash="a" * 64,
            mode="online",
            service_id="test",
            operation_type="inference",
            timestamp=1000,
            input_data={"version": 2},
            output_data={},
        )

        evidence = storage.get_evidence("att_upsert")
        assert evidence["input"]["version"] == 2


# ==============================================================================
# create_storage Factory Tests
# ==============================================================================


class TestCreateStorage:
    """Tests for the create_storage factory function."""

    def test_default_creates_sqlite(self):
        """Default backend is SQLite."""
        storage = create_storage()
        assert isinstance(storage, ReceiptStorage)

    def test_sqlite_backend(self, temp_db_path: Path):
        """Explicit 'sqlite' backend creates ReceiptStorage."""
        storage = create_storage(backend="sqlite", path=temp_db_path)
        assert isinstance(storage, ReceiptStorage)

    def test_json_backend(self, tmp_path: Path):
        """Explicit 'json' backend creates JsonStorageBackend."""
        storage = create_storage(backend="json", path=tmp_path / "json")
        assert isinstance(storage, JsonStorageBackend)

    def test_both_backends_implement_protocol(self, tmp_path: Path):
        """Both backends satisfy the StorageBackend protocol."""
        sqlite_storage = create_storage(
            backend="sqlite", path=tmp_path / "test.db"
        )
        json_storage = create_storage(
            backend="json", path=tmp_path / "json"
        )

        assert isinstance(sqlite_storage, StorageBackend)
        assert isinstance(json_storage, StorageBackend)


# ==============================================================================
# Config Integration Tests
# ==============================================================================


class TestStorageConfig:
    """Tests for evidence storage configuration."""

    def test_default_config_has_sqlite_storage(self):
        """Default config uses sqlite backend."""
        from glacis.config import GlacisConfig

        cfg = GlacisConfig()
        assert cfg.evidence_storage.backend == "sqlite"
        assert cfg.evidence_storage.path is None
        # Backward-compatible alias
        assert cfg.storage.backend == "sqlite"

    def test_config_json_backend(self):
        """Config can specify json backend."""
        from glacis.config import EvidenceStorageConfig

        sc = EvidenceStorageConfig(backend="json", path="/tmp/glacis")
        assert sc.backend == "json"
        assert sc.path == "/tmp/glacis"

    def test_config_from_yaml_evidence_storage_key(self):
        """Config loads from evidence_storage YAML key."""
        from glacis.config import GlacisConfig

        cfg = GlacisConfig(evidence_storage={"backend": "json", "path": "/tmp/test"})
        assert cfg.evidence_storage.backend == "json"
        assert cfg.evidence_storage.path == "/tmp/test"


# ==============================================================================
# Client storage_backend Parameter Tests
# ==============================================================================


class TestClientStorageBackend:
    """Tests for client storage_backend parameter."""

    def test_offline_client_default_sqlite(self, tmp_path: Path):
        """Offline client defaults to SQLite storage."""
        import os
        from glacis import Glacis

        seed = os.urandom(32)
        db_path = tmp_path / "test.db"
        glacis = Glacis(mode="offline", signing_seed=seed, db_path=db_path)

        assert isinstance(glacis._storage, ReceiptStorage)
        glacis.close()

    def test_offline_client_json_backend(self, tmp_path: Path):
        """Offline client with storage_backend='json' uses JSON."""
        import os
        from glacis import Glacis

        seed = os.urandom(32)
        json_dir = tmp_path / "json"
        glacis = Glacis(
            mode="offline",
            signing_seed=seed,
            storage_backend="json",
            storage_path=json_dir,
        )

        assert isinstance(glacis._storage, JsonStorageBackend)

        # Attest and verify it writes to JSONL file
        receipt = glacis.attest(
            service_id="test",
            operation_type="inference",
            input={"a": 1},
            output={"b": 2},
        )

        jsonl_file = json_dir / "receipts.jsonl"
        assert jsonl_file.exists()

        # Retrieve it back
        stored = glacis._storage.get_receipt(receipt.id)
        assert stored is not None
        assert stored.evidence_hash == receipt.evidence_hash

        glacis.close()

    def test_storage_path_overrides_db_path(self, tmp_path: Path):
        """storage_path takes precedence over db_path."""
        import os
        from glacis import Glacis

        seed = os.urandom(32)
        glacis = Glacis(
            mode="offline",
            signing_seed=seed,
            db_path=tmp_path / "ignored.db",
            storage_backend="json",
            storage_path=tmp_path / "json",
        )

        assert isinstance(glacis._storage, JsonStorageBackend)
        glacis.close()
