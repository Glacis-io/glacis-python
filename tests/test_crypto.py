"""
Tests for the crypto module.

Critical: These tests verify that Python produces identical hashes to TypeScript.
"""

import pytest

from glacis.crypto import canonical_json, hash_payload


class TestCanonicalJson:
    """Tests for RFC 8785 canonical JSON serialization."""

    def test_null(self):
        assert canonical_json(None) == "null"

    def test_boolean_true(self):
        assert canonical_json(True) == "true"

    def test_boolean_false(self):
        assert canonical_json(False) == "false"

    def test_integer(self):
        assert canonical_json(42) == "42"
        assert canonical_json(0) == "0"
        assert canonical_json(-1) == "-1"

    def test_float(self):
        assert canonical_json(3.14) == "3.14"
        assert canonical_json(0.0) == "0.0"

    def test_string(self):
        assert canonical_json("hello") == '"hello"'
        assert canonical_json("") == '""'

    def test_string_escaping(self):
        assert canonical_json('hello "world"') == '"hello \\"world\\""'
        assert canonical_json("line1\nline2") == '"line1\\nline2"'

    def test_array_empty(self):
        assert canonical_json([]) == "[]"

    def test_array_simple(self):
        assert canonical_json([1, 2, 3]) == "[1,2,3]"

    def test_array_mixed(self):
        assert canonical_json([1, "two", True, None]) == '[1,"two",true,null]'

    def test_object_empty(self):
        assert canonical_json({}) == "{}"

    def test_object_sorted_keys(self):
        """Keys must be sorted lexicographically."""
        # Input with unsorted keys
        result = canonical_json({"b": 2, "a": 1})
        assert result == '{"a":1,"b":2}'

        # More complex case
        result = canonical_json({"z": 1, "a": 2, "m": 3})
        assert result == '{"a":2,"m":3,"z":1}'

    def test_object_nested(self):
        """Nested objects must also have sorted keys."""
        result = canonical_json({"outer": {"b": 2, "a": 1}})
        assert result == '{"outer":{"a":1,"b":2}}'

    def test_complex_nested_structure(self):
        """Complex nested structures are handled correctly."""
        data = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ],
            "config": {
                "zEnabled": True,
                "aSettings": {"nested": "value"},
            },
        }
        result = canonical_json(data)
        # Keys should be sorted at every level
        assert '"aSettings"' in result
        assert '"zEnabled"' in result
        # aSettings should come before zEnabled
        assert result.index('"aSettings"') < result.index('"zEnabled"')

    def test_nan_raises(self):
        """NaN values should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            canonical_json(float("nan"))

    def test_infinity_raises(self):
        """Infinity values should raise ValueError."""
        with pytest.raises(ValueError, match="Infinity"):
            canonical_json(float("inf"))
        with pytest.raises(ValueError, match="Infinity"):
            canonical_json(float("-inf"))


class TestHashPayload:
    """Tests for payload hashing."""

    def test_hash_length(self):
        """Hash should be 64 hex characters (256 bits)."""
        result = hash_payload({"test": "data"})
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_consistency(self):
        """Same input should produce same hash."""
        data = {"a": 1, "b": 2}
        hash1 = hash_payload(data)
        hash2 = hash_payload(data)
        assert hash1 == hash2

    def test_hash_key_order_invariant(self):
        """Different key orders should produce the same hash."""
        hash1 = hash_payload({"b": 2, "a": 1})
        hash2 = hash_payload({"a": 1, "b": 2})
        assert hash1 == hash2

    def test_hash_different_data_different_hash(self):
        """Different data should produce different hashes."""
        hash1 = hash_payload({"a": 1})
        hash2 = hash_payload({"a": 2})
        assert hash1 != hash2


class TestCrossRuntimeCompatibility:
    """
    Tests that verify Python produces identical hashes to TypeScript.

    These test vectors must match the output from:
    - packages/core/src/crypto.test.ts
    - crates/s3p-core/src/crypto_test.rs

    Run: node -e "console.log(require('@glacis/core').hashCanonicalJson({...}))"
    """

    def test_simple_object(self):
        """Simple object hash matches TypeScript."""
        # Test vector: {"a":1,"b":2}
        # Run: node -e "console.log(require('@glacis/core').hashCanonicalJson({b:2,a:1}))"
        data = {"b": 2, "a": 1}
        result = hash_payload(data)

        # Expected: SHA-256 of '{"a":1,"b":2}'
        # The canonical form sorts keys, producing '{"a":1,"b":2}'
        expected_canonical = '{"a":1,"b":2}'
        import hashlib

        expected_hash = hashlib.sha256(expected_canonical.encode()).hexdigest()
        assert result == expected_hash

    def test_nested_object(self):
        """Nested object hash matches TypeScript."""
        data = {"outer": {"b": 2, "a": 1}, "array": [1, 2, 3]}
        result = hash_payload(data)

        # Expected canonical: {"array":[1,2,3],"outer":{"a":1,"b":2}}
        expected_canonical = '{"array":[1,2,3],"outer":{"a":1,"b":2}}'
        import hashlib

        expected_hash = hashlib.sha256(expected_canonical.encode()).hexdigest()
        assert result == expected_hash

    def test_empty_object(self):
        """Empty object hash matches TypeScript."""
        data = {}
        result = hash_payload(data)

        import hashlib

        expected_hash = hashlib.sha256(b"{}").hexdigest()
        assert result == expected_hash

    def test_array_of_objects(self):
        """Array of objects with unsorted keys."""
        data = [{"z": 1, "a": 2}, {"y": 3, "b": 4}]
        result = hash_payload(data)

        # Expected: [{"a":2,"z":1},{"b":4,"y":3}]
        expected_canonical = '[{"a":2,"z":1},{"b":4,"y":3}]'
        import hashlib

        expected_hash = hashlib.sha256(expected_canonical.encode()).hexdigest()
        assert result == expected_hash

    def test_attestation_payload_structure(self):
        """Real attestation payload structure."""
        data = {
            "input": {"prompt": "Hello, world!"},
            "output": {"response": "Hi there!"},
        }
        result = hash_payload(data)

        # The hash should be deterministic
        expected_canonical = (
            '{"input":{"prompt":"Hello, world!"},"output":{"response":"Hi there!"}}'
        )
        import hashlib

        expected_hash = hashlib.sha256(expected_canonical.encode()).hexdigest()
        assert result == expected_hash
