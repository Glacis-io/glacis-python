"""
Tests for the CLI verification module (glacis/verify.py).
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from glacis.verify import (
    DEFAULT_BASE_URL,
    verify_command,
    verify_offline,
    verify_online,
)


class TestVerifyOnline:
    """Tests for online verification via HTTP."""

    def test_verify_online_success(
        self, httpx_mock: HTTPXMock, sample_verify_response: dict[str, Any],
    ):
        """Successful online verification."""
        httpx_mock.add_response(
            method="GET",
            url=f"{DEFAULT_BASE_URL}/v1/verify/abc123",
            json=sample_verify_response,
        )

        result = verify_online("abc123", DEFAULT_BASE_URL)

        assert result.valid is True
        assert result.verification is not None
        assert result.verification.signature_valid is True
        assert result.verification.proof_valid is True

    def test_verify_online_invalid(self, httpx_mock: HTTPXMock):
        """Online verification with invalid attestation."""
        httpx_mock.add_response(
            method="GET",
            url=f"{DEFAULT_BASE_URL}/v1/verify/invalid123",
            json={"valid": False, "error": "Attestation not found"},
        )

        result = verify_online("invalid123", DEFAULT_BASE_URL)

        assert result.valid is False
        assert result.error == "Attestation not found"

    def test_verify_online_http_error_404(self, httpx_mock: HTTPXMock):
        """Online verification handles 404 error."""
        httpx_mock.add_response(
            method="GET",
            url=f"{DEFAULT_BASE_URL}/v1/verify/notfound",
            status_code=404,
            text="Not Found",
        )

        result = verify_online("notfound", DEFAULT_BASE_URL)

        assert result.valid is False
        assert "404" in result.error

    def test_verify_online_http_error_500(self, httpx_mock: HTTPXMock):
        """Online verification handles 500 error."""
        httpx_mock.add_response(
            method="GET",
            url=f"{DEFAULT_BASE_URL}/v1/verify/servererror",
            status_code=500,
            text="Internal Server Error",
        )

        result = verify_online("servererror", DEFAULT_BASE_URL)

        assert result.valid is False
        assert "500" in result.error

    def test_verify_online_network_error(self, httpx_mock: HTTPXMock):
        """Online verification handles network errors."""
        httpx_mock.add_exception(
            httpx.ConnectError("Connection refused"),
            url=f"{DEFAULT_BASE_URL}/v1/verify/network_fail",
        )

        result = verify_online("network_fail", DEFAULT_BASE_URL)

        assert result.valid is False
        assert "Connection refused" in result.error or "Request failed" in result.error

    def test_verify_online_custom_base_url(
        self, httpx_mock: HTTPXMock, sample_verify_response: dict[str, Any],
    ):
        """Online verification with custom base URL."""
        custom_url = "https://custom.glacis.io"
        httpx_mock.add_response(
            method="GET",
            url=f"{custom_url}/v1/verify/customtest",
            json=sample_verify_response,
        )

        result = verify_online("customtest", custom_url)

        assert result.valid is True


class TestVerifyOffline:
    """Tests for offline verification."""

    def test_verify_offline_success(self, signing_seed: bytes, temp_db_path: Path):
        """Successful offline verification."""
        from glacis import Glacis

        # Create an attestation
        glacis = Glacis(mode="offline", signing_seed=signing_seed, db_path=temp_db_path)
        receipt = glacis.attest(
            service_id="test",
            operation_type="inference",
            input={"prompt": "Hello"},
            output={"response": "Hi"},
        )
        glacis.close()

        # Verify it
        result = verify_offline(receipt)

        assert result.valid is True
        assert result.signature_valid is True
        assert result.witness_status == "UNVERIFIED"


class TestVerifyCommand:
    """Tests for the CLI verify_command function."""

    def test_verify_command_file_not_found(self, capsys):
        """verify_command handles missing file."""
        import argparse

        args = argparse.Namespace(receipt="/nonexistent/file.json", base_url=DEFAULT_BASE_URL)

        with pytest.raises(SystemExit) as exc:
            verify_command(args)

        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "File not found" in captured.err

    def test_verify_command_invalid_json(self, capsys):
        """verify_command handles invalid JSON."""
        import argparse

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            f.flush()
            temp_path = f.name

        try:
            args = argparse.Namespace(receipt=temp_path, base_url=DEFAULT_BASE_URL)

            with pytest.raises(SystemExit) as exc:
                verify_command(args)

            assert exc.value.code == 1
            captured = capsys.readouterr()
            assert "Invalid JSON" in captured.err
        finally:
            os.unlink(temp_path)

    def test_verify_command_detects_offline_receipt(
        self,
        httpx_mock: HTTPXMock,
        signing_seed: bytes,
        temp_db_path: Path,
        capsys,
    ):
        """verify_command detects offline receipt by oatt_ prefix."""
        from glacis import Glacis

        # Create offline receipt
        glacis = Glacis(mode="offline", signing_seed=signing_seed, db_path=temp_db_path)
        receipt = glacis.attest(
            service_id="test",
            operation_type="inference",
            input={"x": 1},
            output={"y": 2},
        )
        glacis.close()

        # Save to file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(receipt.model_dump(by_alias=True), f)
            temp_path = f.name

        try:
            import argparse

            args = argparse.Namespace(receipt=temp_path, base_url=DEFAULT_BASE_URL)
            verify_command(args)

            captured = capsys.readouterr()
            assert "Type: Offline" in captured.out
            assert "VALID" in captured.out
        finally:
            os.unlink(temp_path)

    def test_verify_command_online_receipt(
        self,
        httpx_mock: HTTPXMock,
        sample_online_receipt_data: dict[str, Any],
        sample_verify_response: dict[str, Any],
        capsys,
    ):
        """verify_command handles online receipt."""
        # Mock the verification endpoint
        httpx_mock.add_response(
            method="GET",
            url=f"{DEFAULT_BASE_URL}/v1/verify/{sample_online_receipt_data['evidenceHash']}",
            json=sample_verify_response,
        )

        # Save receipt to file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_online_receipt_data, f)
            temp_path = f.name

        try:
            import argparse

            args = argparse.Namespace(receipt=temp_path, base_url=DEFAULT_BASE_URL)
            verify_command(args)

            captured = capsys.readouterr()
            assert "Type: Online" in captured.out
            assert "VALID" in captured.out
        finally:
            os.unlink(temp_path)


class TestCLI:
    """Tests for CLI entry point."""

    def test_cli_help(self):
        """CLI --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "glacis", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "verify" in result.stdout
        assert "Glacis CLI" in result.stdout

    def test_cli_verify_help(self):
        """CLI verify --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "glacis", "verify", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "--base-url" in result.stdout
        assert "receipt" in result.stdout

    def test_cli_verify_missing_file(self):
        """CLI verify with missing file exits with error."""
        result = subprocess.run(
            [sys.executable, "-m", "glacis", "verify", "/nonexistent/file.json"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 1
        assert "File not found" in result.stderr
