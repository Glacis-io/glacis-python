"""
Tests for glacis/streaming.py - streaming session management.

All HTTP calls are mocked via httpx mocking.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from glacis.streaming import SessionContext, SessionReceipt, StreamingSession

DO_URL = "https://session-do.glacis.io"

# ─── Fixtures ────────────────────────────────────────────────────────────────


def _mock_glacis():
    """Create a mock Glacis client for streaming tests."""
    glacis = MagicMock()
    glacis.get_api_key.return_value = "glsk_live_test"
    glacis.attest = AsyncMock(return_value=MagicMock(id="att_test123"))
    return glacis


def _session_config():
    """Default session config."""
    return {
        "service_id": "voice-assistant",
        "operation_type": "completion",
        "session_do_url": DO_URL,
    }


def _attest_payload():
    """Mock attestation payload returned by /end."""
    return {
        "sessionId": "ses_test123",
        "sessionRoot": "abc123",
        "chunkCount": 2,
        "serviceId": "voice-assistant",
        "operationType": "completion",
        "startedAt": "2024-01-01T00:00:00Z",
        "endedAt": "2024-01-01T00:05:00Z",
        "metadata": {},
    }


def _make_session(glacis=None):
    """Create a StreamingSession directly (bypassing start())."""
    return StreamingSession(
        glacis=glacis or _mock_glacis(),
        session_id="ses_test123",
        session_do_url=DO_URL,
        service_id="voice-assistant",
        operation_type="completion",
        api_key="glsk_live_test",
        session_token="tok_test",
    )


def _mock_response(status_code=200, json_data=None):
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.json.return_value = json_data or {}
    return resp


# ─── StreamingSession Tests ──────────────────────────────────────────────────


class TestStreamingSession:
    """Tests for StreamingSession lifecycle."""

    async def test_session_start_creates_session(self):
        """start() returns a session with a ses_ prefixed ID."""
        glacis = _mock_glacis()
        start_resp = _mock_response(200, {"sessionToken": "tok_abc"})

        with patch("glacis.streaming.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = start_resp
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            session = await StreamingSession.start(glacis, _session_config())

        assert session.session_id.startswith("ses_")

    async def test_session_start_sends_correct_headers(self):
        """start() sends X-Glacis-Key header."""
        glacis = _mock_glacis()
        start_resp = _mock_response(200, {"sessionToken": "tok_abc"})

        with patch("glacis.streaming.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = start_resp
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            await StreamingSession.start(glacis, _session_config())

        call_kwargs = mock_instance.post.call_args
        assert call_kwargs[1]["headers"]["X-Glacis-Key"] == "glsk_live_test"

    async def test_session_start_sends_config(self):
        """start() sends serviceId and operationType in request body."""
        glacis = _mock_glacis()
        start_resp = _mock_response(200, {"sessionToken": "tok_abc"})

        with patch("glacis.streaming.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = start_resp
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            await StreamingSession.start(glacis, _session_config())

        call_kwargs = mock_instance.post.call_args
        body = call_kwargs[1]["json"]
        assert body["serviceId"] == "voice-assistant"
        assert body["operationType"] == "completion"

    async def test_session_start_failure_raises(self):
        """start() raises RuntimeError on HTTP failure."""
        glacis = _mock_glacis()
        fail_resp = _mock_response(500, {"error": "Internal Server Error"})

        with patch("glacis.streaming.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = fail_resp
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            with pytest.raises(RuntimeError, match="Failed to start session"):
                await StreamingSession.start(glacis, _session_config())

    async def test_attest_chunk_sends_hashes(self):
        """attest_chunk() sends hashes, not raw data."""
        session = _make_session()
        chunk_resp = _mock_response(200)
        session._client = AsyncMock()
        session._client.post.return_value = chunk_resp

        await session.attest_chunk(
            input={"prompt": "Hello"},
            output={"response": "Hi!"},
        )

        call_kwargs = session._client.post.call_args
        body = call_kwargs[1]["json"]
        assert "inputHash" in body
        assert "outputHash" in body
        assert len(body["inputHash"]) == 64  # SHA-256 hex
        assert len(body["outputHash"]) == 64
        # Raw data NOT in request
        assert "prompt" not in str(body)

    async def test_attest_chunk_increments_sequence(self):
        """Sequential chunks get incrementing sequence numbers."""
        session = _make_session()
        chunk_resp = _mock_response(200)
        session._client = AsyncMock()
        session._client.post.return_value = chunk_resp

        await session.attest_chunk(input={"a": 1}, output={"b": 1})
        first_seq = session._client.post.call_args[1]["json"]["sequence"]

        await session.attest_chunk(input={"a": 2}, output={"b": 2})
        second_seq = session._client.post.call_args[1]["json"]["sequence"]

        assert first_seq == 0
        assert second_seq == 1

    async def test_attest_chunk_sends_session_token(self):
        """attest_chunk() sends X-Glacis-Session-Token header."""
        session = _make_session()
        chunk_resp = _mock_response(200)
        session._client = AsyncMock()
        session._client.post.return_value = chunk_resp

        await session.attest_chunk(input={"a": 1}, output={"b": 1})

        call_kwargs = session._client.post.call_args
        assert call_kwargs[1]["headers"]["X-Glacis-Session-Token"] == "tok_test"

    async def test_attest_chunk_failure_raises(self):
        """attest_chunk() raises RuntimeError on HTTP failure."""
        session = _make_session()
        fail_resp = _mock_response(400, {"error": "Bad Request"})
        session._client = AsyncMock()
        session._client.post.return_value = fail_resp

        with pytest.raises(RuntimeError, match="Failed to attest chunk"):
            await session.attest_chunk(input={"a": 1}, output={"b": 1})

    async def test_attest_chunk_after_end_raises(self):
        """attest_chunk() raises RuntimeError after session ended."""
        session = _make_session()
        session._ended = True

        with pytest.raises(RuntimeError, match="has ended"):
            await session.attest_chunk(input={"a": 1}, output={"b": 1})

    async def test_attest_chunk_sync_after_end_no_error(self, capsys):
        """attest_chunk_sync() prints warning but doesn't raise after end."""
        session = _make_session()
        session._ended = True

        # Should not raise
        session.attest_chunk_sync(input={"a": 1}, output={"b": 1})

        captured = capsys.readouterr()
        assert "has ended" in captured.out

    async def test_end_calls_glacis_attest(self):
        """end() calls glacis.attest() with the attestation payload."""
        glacis = _mock_glacis()
        session = _make_session(glacis=glacis)
        end_resp = _mock_response(200, {"attestPayload": _attest_payload()})
        session._client = AsyncMock()
        session._client.post.return_value = end_resp

        await session.end()

        glacis.attest.assert_called_once()
        call_kwargs = glacis.attest.call_args[1]
        assert call_kwargs["service_id"] == "voice-assistant"
        assert call_kwargs["input"]["sessionId"] == "ses_test123"

    async def test_end_returns_session_receipt(self):
        """end() returns a SessionReceipt with correct fields."""
        session = _make_session()
        end_resp = _mock_response(200, {"attestPayload": _attest_payload()})
        session._client = AsyncMock()
        session._client.post.return_value = end_resp

        receipt = await session.end()

        assert isinstance(receipt, SessionReceipt)
        assert receipt.session_id == "ses_test123"
        assert receipt.session_root == "abc123"
        assert receipt.chunk_count == 2
        assert receipt.started_at == "2024-01-01T00:00:00Z"
        assert receipt.ended_at == "2024-01-01T00:05:00Z"

    async def test_end_failure_raises(self):
        """end() raises RuntimeError on HTTP failure."""
        session = _make_session()
        fail_resp = _mock_response(500, {"error": "Server Error"})
        session._client = AsyncMock()
        session._client.post.return_value = fail_resp

        with pytest.raises(RuntimeError, match="Failed to end session"):
            await session.end()

    async def test_end_already_ended_raises(self):
        """Calling end() twice raises RuntimeError."""
        session = _make_session()
        session._ended = True

        with pytest.raises(RuntimeError, match="already ended"):
            await session.end()

    async def test_abort_session(self):
        """abort() sends session_id and reason to /abandon."""
        session = _make_session()
        abort_resp = _mock_response(200)
        session._client = AsyncMock()
        session._client.post.return_value = abort_resp

        await session.abort(reason="User cancelled")

        call_args = session._client.post.call_args
        assert "/abandon" in call_args[0][0]
        body = call_args[1]["json"]
        assert body["sessionId"] == "ses_test123"
        assert body["reason"] == "User cancelled"
        assert session._ended is True

    async def test_abort_already_ended_noop(self):
        """abort() on an already-ended session is a no-op."""
        session = _make_session()
        session._ended = True
        session._client = AsyncMock()

        await session.abort()

        session._client.post.assert_not_called()

    async def test_get_status(self):
        """get_status() returns parsed JSON from /status."""
        session = _make_session()
        status_data = {"sessionId": "ses_test123", "status": "active", "chunkCount": 5}
        status_resp = _mock_response(200, status_data)
        session._client = AsyncMock()
        session._client.get.return_value = status_resp

        result = await session.get_status()

        assert result["status"] == "active"
        assert result["chunkCount"] == 5

    async def test_context_manager_auto_ends(self):
        """async with session auto-calls end() on clean exit."""
        glacis = _mock_glacis()
        session = _make_session(glacis=glacis)

        end_resp = _mock_response(200, {"attestPayload": _attest_payload()})
        session._client = AsyncMock()
        session._client.post.return_value = end_resp
        session._client.aclose = AsyncMock()

        async with session:
            pass  # Normal exit

        assert session._ended is True
        # aclose should be called for cleanup
        session._client.aclose.assert_called_once()

    async def test_context_manager_aborts_on_exception(self):
        """async with session calls abort() when exception occurs."""
        session = _make_session()
        abort_resp = _mock_response(200)
        session._client = AsyncMock()
        session._client.post.return_value = abort_resp
        session._client.aclose = AsyncMock()

        with pytest.raises(ValueError):
            async with session:
                raise ValueError("test error")

        assert session._ended is True
        # Verify abort was called (not end)
        call_args = session._client.post.call_args
        assert "/abandon" in call_args[0][0]


# ─── SessionContext Tests ────────────────────────────────────────────────────


class TestSessionContext:
    """Tests for SessionContext higher-level context manager."""

    async def test_session_context_starts_and_ends(self):
        """SessionContext creates and ends a session."""
        glacis = _mock_glacis()
        config = _session_config()

        start_resp = _mock_response(200, {"sessionToken": "tok_abc"})
        end_resp = _mock_response(200, {"attestPayload": _attest_payload()})

        with patch("glacis.streaming.httpx.AsyncClient") as MockClient:
            # Mock for start() context manager
            start_client = AsyncMock()
            start_client.post.return_value = start_resp
            start_client.__aenter__ = AsyncMock(return_value=start_client)
            start_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = start_client

            ctx = SessionContext(glacis, config)
            session = await ctx.__aenter__()

            assert session.session_id.startswith("ses_")

            # Swap client for end() calls
            session._client = AsyncMock()
            session._client.post.return_value = end_resp

            await ctx.__aexit__(None, None, None)

        assert session._ended is True

    async def test_session_context_aborts_on_exception(self):
        """SessionContext aborts session when exception occurs."""
        glacis = _mock_glacis()
        config = _session_config()

        start_resp = _mock_response(200, {"sessionToken": "tok_abc"})
        abort_resp = _mock_response(200)

        with patch("glacis.streaming.httpx.AsyncClient") as MockClient:
            start_client = AsyncMock()
            start_client.post.return_value = start_resp
            start_client.__aenter__ = AsyncMock(return_value=start_client)
            start_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = start_client

            ctx = SessionContext(glacis, config)
            session = await ctx.__aenter__()

            # Swap client for abort() calls
            session._client = AsyncMock()
            session._client.post.return_value = abort_resp

            await ctx.__aexit__(ValueError, ValueError("test"), None)

        assert session._ended is True
