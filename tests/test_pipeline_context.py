"""
Tests for pipeline context managers: GlacisTagScope and GlacisOperationContext.

Validates per-call metadata, operation linking, supersedes handling,
thread isolation, async isolation, and integration with
attest_and_store / handle_blocked_request.
"""

import asyncio
import threading
from typing import Optional
from unittest.mock import MagicMock

import pytest

from glacis.integrations.base import (
    ControlResultsAccumulator,
    GlacisTagScope,
    GlacisOperationContext,
    IntegrationContext,
    _call_metadata_var,
    _last_receipt_var,
    _operation_context_var,
    _supersedes_var,
    attest_and_store,
    build_metadata,
    create_control_plane_results,
    get_active_operation,
    get_call_metadata,
    get_last_receipt,
    get_pending_supersedes,
    handle_blocked_request,
    set_active_operation,
    set_call_metadata,
    set_pending_supersedes,
    setup_integration,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _temp_home(tmp_path, monkeypatch):
    """Redirect evidence storage to temp directory."""
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture(autouse=True)
def _reset_pipeline_state():
    """Reset context-var pipeline state between tests."""
    _vars = (_last_receipt_var, _call_metadata_var, _operation_context_var, _supersedes_var)
    tokens = [v.set(None) for v in _vars]
    yield
    for v, tok in zip(_vars, tokens):
        v.reset(tok)


def _default_config():
    from glacis.config import GlacisConfig

    return GlacisConfig()


def _make_ctx(signing_seed):
    return setup_integration(
        config=None,
        offline=True,
        glacis_api_key=None,
        glacis_base_url="https://api.glacis.io",
        default_service_id="openai",
        service_id="openai",
        debug=False,
        signing_seed=signing_seed,
        policy_key=None,
        input_controls=None,
        output_controls=None,
        metadata=None,
    )


# ─── Per-call Metadata (GlacisTagScope) ────────────────────────────────────


class TestGlacisTagScope:
    """Tests for per-call metadata context manager."""

    def test_sets_context_var(self):
        assert get_call_metadata() is None
        with GlacisTagScope({"step": "generate"}):
            assert get_call_metadata() == {"step": "generate"}

    def test_restores_on_exit(self):
        with GlacisTagScope({"step": "generate"}):
            pass
        assert get_call_metadata() is None

    def test_nesting_restores_previous(self):
        with GlacisTagScope({"step": "outer"}):
            assert get_call_metadata() == {"step": "outer"}
            with GlacisTagScope({"step": "inner"}):
                assert get_call_metadata() == {"step": "inner"}
            assert get_call_metadata() == {"step": "outer"}
        assert get_call_metadata() is None

    def test_restores_on_exception(self):
        try:
            with GlacisTagScope({"step": "fail"}):
                raise ValueError("boom")
        except ValueError:
            pass
        assert get_call_metadata() is None


class TestBuildMetadataWithCallContext:
    """Tests for build_metadata merging with per-call metadata."""

    def test_merges_per_call_metadata(self):
        with GlacisTagScope({"chunk_id": "c012"}):
            md = build_metadata("openai", "gpt-4")
        assert md["chunk_id"] == "c012"
        assert md["provider"] == "openai"
        assert md["model"] == "gpt-4"

    def test_per_call_overrides_client_level(self):
        with GlacisTagScope({"step": "validate"}):
            md = build_metadata("openai", "gpt-4", custom_metadata={"step": "generate"})
        assert md["step"] == "validate"

    def test_per_call_rejects_reserved_keys(self):
        with GlacisTagScope({"provider": "evil"}):
            with pytest.raises(ValueError, match="reserved metadata key"):
                build_metadata("openai", "gpt-4")

    def test_no_call_metadata_unchanged(self):
        md = build_metadata("openai", "gpt-4", custom_metadata={"env": "prod"})
        assert md == {"provider": "openai", "model": "gpt-4", "env": "prod"}

    def test_thread_isolation(self):
        results = {}
        barrier = threading.Barrier(2)

        def thread_fn(name, metadata):
            with GlacisTagScope(metadata):
                barrier.wait(timeout=5)
                results[name] = get_call_metadata()

        t1 = threading.Thread(target=thread_fn, args=("t1", {"step": "gen"}))
        t2 = threading.Thread(target=thread_fn, args=("t2", {"step": "val"}))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert results["t1"] == {"step": "gen"}
        assert results["t2"] == {"step": "val"}


# ─── Operation Context (GlacisOperationContext) ──────────────────────────────


class TestGlacisOperationContext:
    """Tests for operation linking context manager."""

    def test_sets_context_var(self):
        assert get_active_operation() is None
        with GlacisOperationContext() as op:
            assert get_active_operation() is not None
            assert op.operation_id  # non-empty UUID string

    def test_restores_on_exit(self):
        with GlacisOperationContext():
            pass
        assert get_active_operation() is None

    def test_generates_uuid(self):
        import uuid

        with GlacisOperationContext() as op:
            # Should be a valid UUID
            uuid.UUID(op.operation_id)

    def test_auto_increments_sequence(self):
        with GlacisOperationContext() as op:
            active = get_active_operation()
            assert active.next_sequence() == 0
            assert active.next_sequence() == 1
            assert active.next_sequence() == 2

    def test_supersedes_sets_pending(self):
        with GlacisOperationContext() as op:
            assert get_pending_supersedes() is None
            op.supersedes("att_abc123")
            assert get_pending_supersedes() == "att_abc123"

    def test_supersedes_cleared_on_exit(self):
        with GlacisOperationContext() as op:
            op.supersedes("att_abc123")
        assert get_pending_supersedes() is None

    def test_restores_on_exception(self):
        try:
            with GlacisOperationContext() as op:
                op.supersedes("att_xxx")
                raise ValueError("boom")
        except ValueError:
            pass
        assert get_active_operation() is None
        assert get_pending_supersedes() is None

    def test_thread_isolation(self):
        results = {}
        barrier = threading.Barrier(2)

        def thread_fn(name):
            with GlacisOperationContext() as op:
                barrier.wait(timeout=5)
                results[name] = op.operation_id

        t1 = threading.Thread(target=thread_fn, args=("t1",))
        t2 = threading.Thread(target=thread_fn, args=("t2",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert results["t1"] != results["t2"]


# ─── attest_and_store with Operation Context ──────────────────────────────────


class TestAttestAndStoreWithOperation:
    """Tests that attest_and_store reads context-var operation state."""

    def test_passes_operation_fields(self, signing_seed):
        ctx = _make_ctx(signing_seed)
        acc = ControlResultsAccumulator()
        cpr = create_control_plane_results(acc, ctx.cfg, "gpt-4", "openai")

        with GlacisOperationContext() as op:
            attest_and_store(
                ctx,
                input_data={"model": "gpt-4", "messages": []},
                output_data={"model": "gpt-4", "choices": []},
                metadata={"provider": "openai", "model": "gpt-4"},
                control_plane_results=cpr,
            )
            receipt = get_last_receipt()

        assert receipt is not None
        assert receipt.operation_id == op.operation_id
        assert receipt.operation_sequence == 0

    def test_auto_increments_across_calls(self, signing_seed):
        ctx = _make_ctx(signing_seed)
        acc = ControlResultsAccumulator()
        cpr = create_control_plane_results(acc, ctx.cfg, "gpt-4", "openai")

        with GlacisOperationContext() as op:
            attest_and_store(
                ctx,
                input_data={"messages": [{"role": "user", "content": "call 1"}]},
                output_data={"choices": []},
                metadata={"provider": "openai", "model": "gpt-4"},
                control_plane_results=cpr,
            )
            r1 = get_last_receipt()

            attest_and_store(
                ctx,
                input_data={"messages": [{"role": "user", "content": "call 2"}]},
                output_data={"choices": []},
                metadata={"provider": "openai", "model": "gpt-4"},
                control_plane_results=cpr,
            )
            r2 = get_last_receipt()

        assert r1.operation_id == r2.operation_id == op.operation_id
        assert r1.operation_sequence == 0
        assert r2.operation_sequence == 1

    def test_passes_supersedes(self, signing_seed):
        ctx = _make_ctx(signing_seed)
        acc = ControlResultsAccumulator()
        cpr = create_control_plane_results(acc, ctx.cfg, "gpt-4", "openai")

        with GlacisOperationContext() as op:
            attest_and_store(
                ctx,
                input_data={"messages": [{"role": "user", "content": "gen"}]},
                output_data={"choices": []},
                metadata={"provider": "openai", "model": "gpt-4"},
                control_plane_results=cpr,
            )
            gen_receipt = get_last_receipt()

            op.supersedes(gen_receipt.id)

            attest_and_store(
                ctx,
                input_data={"messages": [{"role": "user", "content": "regen"}]},
                output_data={"choices": []},
                metadata={"provider": "openai", "model": "gpt-4"},
                control_plane_results=cpr,
            )
            regen_receipt = get_last_receipt()

        assert regen_receipt.supersedes == gen_receipt.id

    def test_supersedes_consumed_after_use(self, signing_seed):
        ctx = _make_ctx(signing_seed)
        acc = ControlResultsAccumulator()
        cpr = create_control_plane_results(acc, ctx.cfg, "gpt-4", "openai")

        with GlacisOperationContext() as op:
            op.supersedes("att_xxx")

            attest_and_store(
                ctx,
                input_data={"messages": []},
                output_data={"choices": []},
                metadata={"provider": "openai", "model": "gpt-4"},
                control_plane_results=cpr,
            )
            r1 = get_last_receipt()

            # Second call should NOT have supersedes
            attest_and_store(
                ctx,
                input_data={"messages": []},
                output_data={"choices": []},
                metadata={"provider": "openai", "model": "gpt-4"},
                control_plane_results=cpr,
            )
            r2 = get_last_receipt()

        assert r1.supersedes == "att_xxx"
        assert r2.supersedes is None

    def test_without_operation_context_unchanged(self, signing_seed):
        """Existing behavior: no operation fields when no context active."""
        ctx = _make_ctx(signing_seed)
        acc = ControlResultsAccumulator()
        cpr = create_control_plane_results(acc, ctx.cfg, "gpt-4", "openai")

        attest_and_store(
            ctx,
            input_data={"messages": []},
            output_data={"choices": []},
            metadata={"provider": "openai", "model": "gpt-4"},
            control_plane_results=cpr,
        )
        receipt = get_last_receipt()

        assert receipt is not None
        # operation_id should be auto-generated by attest() but not from our context
        assert receipt.operation_sequence == 0
        assert receipt.supersedes is None


# ─── handle_blocked_request with Operation Context ───────────────────────────


class TestHandleBlockedWithOperation:
    """Tests that handle_blocked_request forwards operation context."""

    def test_blocked_request_gets_operation_fields(self, signing_seed):
        ctx = _make_ctx(signing_seed)
        acc = ControlResultsAccumulator()
        cpr = create_control_plane_results(acc, ctx.cfg, "gpt-4", "openai")

        with GlacisOperationContext() as op:
            with pytest.raises(Exception):
                handle_blocked_request(
                    glacis_client=ctx.glacis,
                    service_id="openai",
                    input_data={"messages": []},
                    control_plane_results=cpr,
                    provider="openai",
                    model="gpt-4",
                    blocking_control_type="jailbreak",
                    blocking_score=0.95,
                    debug=False,
                    storage_backend=ctx.storage_backend,
                    storage_path=ctx.storage_path,
                )

            receipt = get_last_receipt()

        assert receipt is not None
        assert receipt.operation_id == op.operation_id


# ─── Combined Context Managers ────────────────────────────────────────────────


class TestCombinedContexts:
    """Tests for using both context managers together."""

    def test_metadata_and_operation_together(self, signing_seed):
        ctx = _make_ctx(signing_seed)
        acc = ControlResultsAccumulator()
        cpr = create_control_plane_results(acc, ctx.cfg, "gpt-4", "openai")

        with GlacisOperationContext() as op:
            with GlacisTagScope({"step": "generate"}):
                md = build_metadata("openai", "gpt-4", ctx.custom_metadata)
                attest_and_store(ctx, {"messages": []}, {"choices": []}, md, cpr)
                r1 = get_last_receipt()

            with GlacisTagScope({"step": "validate"}):
                md = build_metadata("openai", "gpt-4", ctx.custom_metadata)
                attest_and_store(ctx, {"messages": []}, {"choices": []}, md, cpr)
                r2 = get_last_receipt()

        assert r1.operation_id == r2.operation_id == op.operation_id
        assert r1.operation_sequence == 0
        assert r2.operation_sequence == 1


# ─── Provider Wrapper Availability ───────────────────────────────────────────


class TestProviderContextManagersExist:
    """Tests that provider wrappers expose glacis_tags and glacis_operation."""

    def test_litellm_has_context_managers(self):
        """AttestedLiteLLM class has both methods."""
        from glacis.integrations.litellm import AttestedLiteLLM

        ctx_mock = MagicMock()
        client = AttestedLiteLLM(ctx_mock)
        assert hasattr(client, "glacis_tags")
        assert hasattr(client, "glacis_operation")

        # Verify they return the right types
        assert isinstance(client.glacis_tags({"k": "v"}), GlacisTagScope)
        assert isinstance(client.glacis_operation(), GlacisOperationContext)


# ─── Async Isolation ─────────────────────────────────────────────────────────


class TestAsyncIsolation:
    """Verify that ContextVar storage isolates concurrent coroutines.

    These tests would FAIL with the old threading.local() approach because
    all coroutines on the same event loop thread shared the same storage.
    With ContextVar, each task created via asyncio.gather() gets its own
    copy of the context.
    """

    def test_call_metadata_isolated_across_coroutines(self):
        """Concurrent coroutines with GlacisTagScope must not bleed metadata."""
        results: dict[str, Optional[dict[str, str]]] = {}

        async def coro(name: str, metadata: dict[str, str]):
            with GlacisTagScope(metadata):
                await asyncio.sleep(0)  # force a task switch
                results[name] = get_call_metadata()

        async def run():
            await asyncio.gather(
                coro("a", {"doc": "Eliquis.pdf"}),
                coro("b", {"doc": "Xarelto.pdf"}),
            )

        asyncio.run(run())

        assert results["a"] == {"doc": "Eliquis.pdf"}
        assert results["b"] == {"doc": "Xarelto.pdf"}

    def test_last_receipt_isolated_across_coroutines(self):
        """Each coroutine's set_last_receipt must not overwrite the other's."""
        from glacis.integrations.base import set_last_receipt
        from glacis.models import Attestation

        results: dict[str, Optional[str]] = {}

        def _make_attestation(name: str) -> Attestation:
            return Attestation(
                id=f"oatt_{name}",
                evidence_hash="a" * 64,
                public_key="b" * 64,
                signature="c" * 10,
            )

        async def coro(name: str):
            att = _make_attestation(name)
            set_last_receipt(att)
            await asyncio.sleep(0)  # force task switch
            receipt = get_last_receipt()
            results[name] = receipt.id if receipt else None

        async def run():
            await asyncio.gather(coro("first"), coro("second"))

        asyncio.run(run())

        assert results["first"] == "oatt_first"
        assert results["second"] == "oatt_second"

    def test_operation_context_isolated_across_coroutines(self):
        """Each coroutine's GlacisOperationContext must be independent."""
        results: dict[str, str] = {}

        async def coro(name: str):
            with GlacisOperationContext() as op:
                await asyncio.sleep(0)  # force task switch
                active = get_active_operation()
                results[name] = active.operation_id

        async def run():
            await asyncio.gather(coro("x"), coro("y"))

        asyncio.run(run())

        assert results["x"] != results["y"]

    def test_supersedes_isolated_across_coroutines(self):
        """Supersedes set in one coroutine must not leak to another."""
        results: dict[str, Optional[str]] = {}

        async def run():
            wrote_event = asyncio.Event()

            async def coro_setter():
                set_pending_supersedes("att_replaced")
                wrote_event.set()
                await asyncio.sleep(0)
                results["setter"] = get_pending_supersedes()

            async def coro_observer():
                await wrote_event.wait()  # ensure setter wrote before we read
                results["observer"] = get_pending_supersedes()

            await asyncio.gather(coro_setter(), coro_observer())

        asyncio.run(run())

        assert results["setter"] == "att_replaced"
        assert results["observer"] is None
