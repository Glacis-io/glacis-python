"""
Streaming SDK Extensions for Python

Provides streaming session support for real-time AI interactions
(voice, healthcare, etc.) where chunks are attested individually.

Example:
    >>> from glacis import Glacis
    >>> from glacis.streaming import StreamingSession
    >>>
    >>> glacis = Glacis(api_key="glsk_live_xxx")
    >>> session = await StreamingSession.start(glacis, {
    ...     "service_id": "voice-assistant",
    ...     "operation_type": "completion",
    ...     "session_do_url": "https://session-do.glacis.io",
    ... })
    >>>
    >>> await session.attest_chunk({"input": audio_chunk, "output": transcript})
    >>> receipt = await session.end(metadata={"duration": "00:05:23"})

Context Manager:
    >>> async with glacis.session(config) as session:
    ...     await session.attest_chunk(input=audio, output=transcript)
    ... # Auto-ends on exit
"""

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

import httpx

from glacis.crypto import hash_payload


class StreamingSessionConfig(TypedDict, total=False):
    """Configuration for starting a streaming session."""

    service_id: str
    operation_type: str
    session_do_url: str
    auto_end_timeout_ms: Optional[int]
    chunk_batch_size: Optional[int]


@dataclass
class SessionReceipt:
    """Receipt from ending a session."""

    session_id: str
    session_root: str
    chunk_count: int
    started_at: str
    ended_at: str
    attest_receipt: Optional[Any] = None


class StreamingSession:
    """
    Streaming session for chunk-by-chunk attestation.

    Use StreamingSession.start() to create a new session.
    """

    def __init__(
        self,
        glacis: Any,
        session_id: str,
        session_do_url: str,
        service_id: str,
        operation_type: str,
        api_key: str,
        session_token: str,
    ):
        self._glacis = glacis
        self._session_id = session_id
        self._session_do_url = session_do_url.rstrip("/")
        self._service_id = service_id
        self._operation_type = operation_type
        self._api_key = api_key
        self._session_token = session_token
        self._sequence = 0
        self._ended = False
        self._pending_tasks: list[asyncio.Task[None]] = []
        self._client = httpx.AsyncClient()

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id

    @classmethod
    async def start(
        cls,
        glacis: Any,
        config: StreamingSessionConfig,
    ) -> "StreamingSession":
        """
        Start a new streaming session.

        Args:
            glacis: AsyncGlacis or Glacis client
            config: Session configuration

        Returns:
            StreamingSession instance
        """
        session_id = f"ses_{uuid.uuid4()}"
        api_key = glacis.get_api_key()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config['session_do_url']}/start",
                headers={
                    "X-Glacis-Key": api_key,
                },
                json={
                    "sessionId": session_id,
                    "serviceId": config["service_id"],
                    "operationType": config["operation_type"],
                    "config": {
                        "autoEndTimeoutMs": config.get("auto_end_timeout_ms"),
                        "chunkBatchSize": config.get("chunk_batch_size"),
                    },
                },
            )

            if not response.is_success:
                try:
                    error = response.json()
                except Exception:
                    error = {}
                raise RuntimeError(
                    f"Failed to start session: {error.get('error', response.status_code)}"
                )

            result = response.json()
            session_token = result.get("sessionToken", "")

        return cls(
            glacis=glacis,
            session_id=session_id,
            session_do_url=config["session_do_url"],
            service_id=config["service_id"],
            operation_type=config["operation_type"],
            api_key=api_key,
            session_token=session_token,
        )

    def attest_chunk_sync(self, input: Any, output: Any) -> None:
        """
        Attest a chunk synchronously (fire-and-forget).

        This queues the chunk for attestation and returns immediately.
        Errors are logged but do not raise.
        """
        if self._ended:
            print(f"[glacis] Cannot attest chunk: session {self._session_id} has ended")
            return

        task = asyncio.create_task(self._attest_chunk_internal(input, output))
        self._pending_tasks.append(task)

    async def attest_chunk(self, input: Any, output: Any) -> None:
        """
        Attest a chunk asynchronously.

        Args:
            input: Input data (hashed locally)
            output: Output data (hashed locally)
        """
        if self._ended:
            raise RuntimeError(f"Session {self._session_id} has ended")

        await self._attest_chunk_internal(input, output)

    async def _attest_chunk_internal(self, input: Any, output: Any) -> None:
        """Internal chunk attestation."""
        input_hash = hash_payload(input)
        output_hash = hash_payload(output)
        sequence = self._sequence
        self._sequence += 1

        response = await self._client.post(
            f"{self._session_do_url}/chunk",
            headers={
                "X-Glacis-Session-Token": self._session_token,
            },
            json={
                "sessionId": self._session_id,
                "sequence": sequence,
                "inputHash": input_hash,
                "outputHash": output_hash,
            },
        )

        if not response.is_success:
            try:
                error = response.json()
            except Exception:
                error = {}
            raise RuntimeError(
                f"Failed to attest chunk: {error.get('error', response.status_code)}"
            )

    async def end(self, metadata: Optional[dict[str, str]] = None) -> SessionReceipt:
        """
        End the session and submit to main log.

        Args:
            metadata: Optional metadata to include

        Returns:
            SessionReceipt with attestation info
        """
        if self._ended:
            raise RuntimeError(f"Session {self._session_id} already ended")

        # Wait for all pending chunks
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)

        self._ended = True

        # End session in DO
        response = await self._client.post(
            f"{self._session_do_url}/end",
            headers={
                "X-Glacis-Session-Token": self._session_token,
            },
            json={"sessionId": self._session_id, "metadata": metadata},
        )

        if not response.is_success:
            try:
                error = response.json()
            except Exception:
                error = {}
            raise RuntimeError(
                f"Failed to end session: {error.get('error', response.status_code)}"
            )

        result = response.json()
        attest_payload = result["attestPayload"]

        # Submit to main transparency log
        attest_receipt = await self._glacis.attest(
            service_id=attest_payload["serviceId"],
            operation_type=attest_payload["operationType"],
            input={
                "sessionId": attest_payload["sessionId"],
                "sessionRoot": attest_payload["sessionRoot"],
            },
            output={"chunkCount": attest_payload["chunkCount"]},
            metadata={
                **(attest_payload.get("metadata") or {}),
                "sessionId": attest_payload["sessionId"],
            },
        )

        return SessionReceipt(
            session_id=attest_payload["sessionId"],
            session_root=attest_payload["sessionRoot"],
            chunk_count=attest_payload["chunkCount"],
            started_at=attest_payload["startedAt"],
            ended_at=attest_payload["endedAt"],
            attest_receipt=attest_receipt,
        )

    async def abort(self, reason: Optional[str] = None) -> None:
        """
        Abort the session without submitting to main log.
        """
        if self._ended:
            return

        self._ended = True

        response = await self._client.post(
            f"{self._session_do_url}/abandon",
            headers={
                "X-Glacis-Session-Token": self._session_token,
            },
            json={"sessionId": self._session_id, "reason": reason},
        )

        if not response.is_success:
            try:
                error = response.json()
            except Exception:
                error = {}
            raise RuntimeError(
                f"Failed to abort session: {error.get('error', response.status_code)}"
            )

    async def get_status(self) -> dict[str, Any]:
        """Get current session status."""
        response = await self._client.get(
            f"{self._session_do_url}/status",
            params={"sessionId": self._session_id},
            headers={
                "X-Glacis-Session-Token": self._session_token,
            },
        )

        if not response.is_success:
            try:
                error = response.json()
            except Exception:
                error = {}
            raise RuntimeError(
                f"Failed to get status: {error.get('error', response.status_code)}"
            )

        result: dict[str, Any] = response.json()
        return result

    async def __aenter__(self) -> "StreamingSession":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            await self.abort(str(exc_val) if exc_val else "Exception occurred")
        elif not self._ended:
            await self.end()
        await self._client.aclose()


class SessionContext:
    """
    Context manager for streaming sessions.

    Example:
        >>> async with SessionContext(glacis, config) as session:
        ...     await session.attest_chunk(input=data, output=result)
        ... # Auto-ends and submits on exit
    """

    def __init__(
        self,
        glacis: Any,
        config: StreamingSessionConfig,
        metadata: Optional[dict[str, str]] = None,
    ):
        self._glacis = glacis
        self._config = config
        self._metadata = metadata
        self._session: Optional[StreamingSession] = None

    async def __aenter__(self) -> StreamingSession:
        self._session = await StreamingSession.start(self._glacis, self._config)
        return self._session

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._session is None:
            return

        if exc_type is not None:
            await self._session.abort(str(exc_val) if exc_val else "Exception occurred")
        elif not self._session._ended:
            await self._session.end(metadata=self._metadata)
