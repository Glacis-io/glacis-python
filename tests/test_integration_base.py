"""
Unit tests for glacis/integrations/base.py shared utilities.

Tests ControlResultsAccumulator, run_input_controls, run_output_controls,
create_control_plane_results, handle_blocked_request, and initialize_config.
"""

from unittest.mock import MagicMock, patch

import pytest

from glacis.integrations.base import (
    ControlResultsAccumulator,
    GlacisBlockedError,
    IntegrationContext,
    _map_control_type,
    attest_and_store,
    build_metadata,
    check_input_block,
    check_output_block,
    create_control_plane_results,
    handle_blocked_request,
    initialize_config,
    run_input_controls,
    run_output_controls,
    setup_integration,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _temp_home(tmp_path, monkeypatch):
    """Redirect evidence storage to temp directory."""
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture(autouse=True)
def _reset_receipt_state():
    """Reset thread-local receipt state between tests."""
    from glacis.integrations.base import _thread_local

    if hasattr(_thread_local, "last_receipt"):
        del _thread_local.last_receipt
    yield
    if hasattr(_thread_local, "last_receipt"):
        del _thread_local.last_receipt


def _default_config():
    """Load a default GlacisConfig."""
    from glacis.config import GlacisConfig

    return GlacisConfig()


def _make_stage_result(results=None, effective_text="test", should_block=False):
    """Create a StageResult for testing."""
    from glacis.controls import StageResult

    return StageResult(
        results=results or [],
        effective_text=effective_text,
        should_block=should_block,
    )


def _make_control_result(
    control_type="pii", detected=True, action="flag",
    score=None, categories=None, latency_ms=5, modified_text=None,
):
    """Create a ControlResult for testing."""
    from glacis.controls.base import ControlResult

    return ControlResult(
        control_type=control_type,
        detected=detected,
        action=action,
        score=score,
        categories=categories or [],
        latency_ms=latency_ms,
        modified_text=modified_text,
    )


# ─── ControlResultsAccumulator Tests ─────────────────────────────────────────


class TestControlResultsAccumulator:
    """Tests for the generic ControlResultsAccumulator."""

    def test_accumulator_initial_state(self):
        """Fresh accumulator has clean state."""
        acc = ControlResultsAccumulator()
        assert acc.control_executions == []
        assert acc.should_block is False
        assert acc.effective_input_text is None
        assert acc.effective_output_text is None

    def test_accumulator_update_from_input_stage(self):
        """Input stage results populate control_executions."""
        acc = ControlResultsAccumulator()
        pii_result = _make_control_result(
            control_type="pii", detected=True, action="flag",
            categories=["US_SSN"], latency_ms=5,
        )
        stage = _make_stage_result(
            results=[pii_result], effective_text="redacted text",
        )

        acc.update_from_stage(stage, "input")

        assert len(acc.control_executions) == 1
        assert acc.control_executions[0].type == "pii"
        assert acc.control_executions[0].status == "flag"
        assert acc.control_executions[0].stage == "input"
        assert acc.effective_input_text == "redacted text"

    def test_accumulator_update_from_output_stage(self):
        """Output stage results tracked separately."""
        acc = ControlResultsAccumulator()
        wf_result = _make_control_result(
            control_type="word_filter", detected=True, action="flag",
        )
        stage = _make_stage_result(
            results=[wf_result], effective_text="filtered output",
        )

        acc.update_from_stage(stage, "output")

        assert len(acc.control_executions) == 1
        assert acc.control_executions[0].stage == "output"
        assert acc.effective_output_text == "filtered output"

    def test_accumulator_should_block_from_stage(self):
        """should_block propagates from stage result."""
        acc = ControlResultsAccumulator()
        stage = _make_stage_result(should_block=True)

        acc.update_from_stage(stage, "input")
        assert acc.should_block is True

    def test_accumulator_multiple_stages(self):
        """Both input and output stage results accumulate."""
        acc = ControlResultsAccumulator()

        input_result = _make_control_result(control_type="pii")
        input_stage = _make_stage_result(
            results=[input_result], effective_text="input text",
        )
        acc.update_from_stage(input_stage, "input")

        output_result = _make_control_result(control_type="word_filter")
        output_stage = _make_stage_result(
            results=[output_result], effective_text="output text",
        )
        acc.update_from_stage(output_stage, "output")

        assert len(acc.control_executions) == 2
        assert acc.effective_input_text == "input text"
        assert acc.effective_output_text == "output text"

    def test_accumulator_generic_control_types(self):
        """Any control type works (not just pii/jailbreak)."""
        acc = ControlResultsAccumulator()
        results = [
            _make_control_result(control_type="pii", action="flag"),
            _make_control_result(control_type="jailbreak", action="forward", score=0.1),
            _make_control_result(control_type="word_filter", action="flag"),
            _make_control_result(control_type="content_safety", action="block", score=0.9),
        ]
        stage = _make_stage_result(results=results, should_block=True)
        acc.update_from_stage(stage, "input")

        types = [ce.type for ce in acc.control_executions]
        assert "pii" in types
        assert "jailbreak" in types
        assert "word_filter" in types
        assert "content_safety" in types

    def test_accumulator_get_blocking_control(self):
        """get_blocking_control returns the first blocking control execution."""
        acc = ControlResultsAccumulator()
        results = [
            _make_control_result(control_type="pii", action="flag"),
            _make_control_result(control_type="jailbreak", action="block", score=0.95),
        ]
        stage = _make_stage_result(results=results, should_block=True)
        acc.update_from_stage(stage, "input")

        blocking = acc.get_blocking_control()
        assert blocking is not None
        assert blocking.status == "block"
        assert blocking.type == "jailbreak"


# ─── Control Type Mapping Tests ──────────────────────────────────────────────


class TestControlTypeMapping:
    """Tests for _map_control_type."""

    def test_known_types_pass_through(self):
        assert _map_control_type("pii") == "pii"
        assert _map_control_type("jailbreak") == "jailbreak"
        assert _map_control_type("word_filter") == "word_filter"
        assert _map_control_type("content_safety") == "content_safety"
        assert _map_control_type("custom") == "custom"

    def test_unknown_types_map_to_custom(self):
        assert _map_control_type("toxicity") == "custom"
        assert _map_control_type("my_guard") == "custom"


# ─── run_input_controls / run_output_controls Tests ──────────────────────────


class TestRunControls:
    """Tests for run_input_controls and run_output_controls."""

    def test_run_input_controls(self):
        """run_input_controls calls runner.run_input and updates accumulator."""
        from glacis.controls.base import ControlResult

        pii = ControlResult(
            control_type="pii", detected=True, action="flag",
            categories=["US_SSN"], latency_ms=5,
        )
        stage = _make_stage_result(results=[pii], effective_text="redacted")

        runner = MagicMock()
        runner.run_input.return_value = stage

        acc = ControlResultsAccumulator()
        result = run_input_controls(runner, "SSN: 123-45-6789", acc)

        assert result == "redacted"
        assert acc.effective_input_text == "redacted"
        runner.run_input.assert_called_once_with("SSN: 123-45-6789")

    def test_run_output_controls(self):
        """run_output_controls calls runner.run_output and updates accumulator."""
        from glacis.controls.base import ControlResult

        wf = ControlResult(
            control_type="word_filter", detected=True, action="flag",
            modified_text="filtered",
        )
        stage = _make_stage_result(results=[wf], effective_text="filtered")

        runner = MagicMock()
        runner.run_output.return_value = stage

        acc = ControlResultsAccumulator()
        result = run_output_controls(runner, "secret data", acc)

        assert result == "filtered"
        assert acc.effective_output_text == "filtered"
        runner.run_output.assert_called_once_with("secret data")


# ─── create_control_plane_results Tests ──────────────────────────────────────


class TestCreateControlPlaneResults:
    """Tests for building ControlPlaneResults from accumulator state."""

    def test_forwarded_when_no_detections(self):
        """No detections -> action='forwarded'."""
        acc = ControlResultsAccumulator()
        cfg = _default_config()

        result = create_control_plane_results(acc, cfg, "gpt-4", "openai")

        assert result.determination.action == "forwarded"

    def test_blocked_when_should_block(self):
        """should_block=True -> action='blocked'."""
        acc = ControlResultsAccumulator()
        stage = _make_stage_result(should_block=True)
        acc.update_from_stage(stage, "input")
        cfg = _default_config()

        result = create_control_plane_results(acc, cfg, "gpt-4", "openai")

        assert result.determination.action == "blocked"

    def test_forwarded_when_flag_only(self):
        """Flag-only detections -> action='forwarded'."""
        acc = ControlResultsAccumulator()
        results = [_make_control_result(action="flag")]
        stage = _make_stage_result(results=results, should_block=False)
        acc.update_from_stage(stage, "input")
        cfg = _default_config()

        result = create_control_plane_results(acc, cfg, "gpt-4", "openai")

        assert result.determination.action == "forwarded"

    def test_system_prompt_hash_and_temperature(self):
        """system_prompt_hash and temperature passed through to ModelInfo."""
        acc = ControlResultsAccumulator()
        cfg = _default_config()

        result = create_control_plane_results(
            acc, cfg, "gpt-4", "openai",
            system_prompt_hash="abc123",
            temperature=0.7,
        )

        assert result.policy.model.system_prompt_hash == "abc123"
        assert result.policy.model.temperature == 0.7

    def test_policy_context_from_config(self):
        """Policy context uses values from config."""
        acc = ControlResultsAccumulator()
        cfg = _default_config()

        result = create_control_plane_results(acc, cfg, "gpt-4", "openai")

        assert result.policy.id == "default"
        assert result.policy.version == "1.0"
        assert result.policy.model.model_id == "gpt-4"
        assert result.policy.model.provider == "openai"
        assert result.policy.environment == "development"

    def test_control_executions_included(self):
        """Control executions from accumulator are included."""
        acc = ControlResultsAccumulator()
        results = [
            _make_control_result(control_type="pii", action="flag", latency_ms=5),
            _make_control_result(control_type="jailbreak", action="forward", score=0.1),
        ]
        stage = _make_stage_result(results=results)
        acc.update_from_stage(stage, "input")
        cfg = _default_config()

        result = create_control_plane_results(acc, cfg, "gpt-4", "openai")

        assert len(result.controls) == 2
        types = {c.type for c in result.controls}
        assert "pii" in types
        assert "jailbreak" in types


# ─── handle_blocked_request Tests ────────────────────────────────────────────


class TestHandleBlockedRequest:
    """Tests for handle_blocked_request utility."""

    def test_raises_glacis_blocked_error(self, signing_seed):
        """Raises GlacisBlockedError with correct type and score."""
        from glacis import Glacis

        glacis = Glacis(mode="offline", signing_seed=signing_seed)

        with pytest.raises(GlacisBlockedError) as exc_info:
            handle_blocked_request(
                glacis_client=glacis,
                service_id="test",
                input_data={"model": "gpt-4", "messages": []},
                control_plane_results=None,
                provider="openai",
                model="gpt-4",
                blocking_control_type="jailbreak",
                blocking_score=0.95,
                debug=False,
            )

        assert exc_info.value.control_type == "jailbreak"
        assert exc_info.value.score == 0.95
        assert "jailbreak" in str(exc_info.value)

    def test_generic_blocking_control_type(self, signing_seed):
        """Works with any control type, not just jailbreak."""
        from glacis import Glacis

        glacis = Glacis(mode="offline", signing_seed=signing_seed)

        with pytest.raises(GlacisBlockedError) as exc_info:
            handle_blocked_request(
                glacis_client=glacis,
                service_id="test",
                input_data={"model": "gpt-4", "messages": []},
                control_plane_results=None,
                provider="openai",
                model="gpt-4",
                blocking_control_type="content_safety",
                blocking_score=0.99,
                debug=False,
            )

        assert exc_info.value.control_type == "content_safety"
        assert exc_info.value.score == 0.99

    def test_attests_before_raising(self, signing_seed):
        """glacis.attest() is called before the error is raised."""
        from glacis import Glacis

        glacis = Glacis(mode="offline", signing_seed=signing_seed)

        with patch.object(glacis, "attest", wraps=glacis.attest) as mock_attest:
            with pytest.raises(GlacisBlockedError):
                handle_blocked_request(
                    glacis_client=glacis,
                    service_id="test",
                    input_data={"model": "gpt-4", "messages": []},
                    control_plane_results=None,
                    provider="openai",
                    model="gpt-4",
                    blocking_control_type="jailbreak",
                    blocking_score=0.95,
                    debug=False,
                )

        mock_attest.assert_called_once()

    def test_attestation_failure_still_raises(self):
        """Even if attest() throws, GlacisBlockedError is still raised."""
        glacis_mock = MagicMock()
        glacis_mock.attest.side_effect = Exception("Network error")

        with pytest.raises(GlacisBlockedError) as exc_info:
            handle_blocked_request(
                glacis_client=glacis_mock,
                service_id="test",
                input_data={"model": "gpt-4", "messages": []},
                control_plane_results=None,
                provider="openai",
                model="gpt-4",
                blocking_control_type="jailbreak",
                blocking_score=0.80,
                debug=False,
            )

        assert exc_info.value.score == 0.80


# ─── initialize_config Tests ─────────────────────────────────────────────────


class TestInitializeConfig:
    """Tests for initialize_config parameter precedence."""

    def test_offline_explicit_overrides_config(self):
        """Explicit offline=True overrides config default."""
        _, effective_offline, _ = initialize_config(
            config_path=None,
            offline=True,
            glacis_api_key=None,
            default_service_id="openai",
            service_id="openai",
        )

        assert effective_offline is True

    def test_glacis_api_key_implies_online(self):
        """Providing glacis_api_key with no explicit offline implies online mode."""
        _, effective_offline, _ = initialize_config(
            config_path=None,
            offline=None,
            glacis_api_key="glsk_live_test",
            default_service_id="openai",
            service_id="openai",
        )

        assert effective_offline is False

    def test_service_id_from_config_when_default(self):
        """When service_id matches default, uses config's attestation.service_id."""
        _, _, effective_service_id = initialize_config(
            config_path=None,
            offline=True,
            glacis_api_key=None,
            default_service_id="openai",
            service_id="openai",
        )

        assert effective_service_id == "openai"


# ─── build_metadata Tests ──────────────────────────────────────────────────


class TestBuildMetadata:
    """Tests for build_metadata helper."""

    def test_defaults_only(self):
        """Returns provider and model when no custom metadata."""
        result = build_metadata("openai", "gpt-4o")
        assert result == {"provider": "openai", "model": "gpt-4o"}

    def test_merges_custom_metadata(self):
        """Custom metadata is merged with defaults."""
        result = build_metadata(
            "openai", "gpt-4o",
            custom_metadata={"department": "legal", "use_case": "review"},
        )
        assert result == {
            "provider": "openai",
            "model": "gpt-4o",
            "department": "legal",
            "use_case": "review",
        }

    def test_extra_kwargs(self):
        """Extra kwargs are included in metadata."""
        result = build_metadata("openai", "gpt-4o", blocked="True")
        assert result == {
            "provider": "openai",
            "model": "gpt-4o",
            "blocked": "True",
        }

    def test_custom_metadata_with_extra_kwargs(self):
        """Custom metadata and extra kwargs combine."""
        result = build_metadata(
            "openai", "gpt-4o",
            custom_metadata={"dept": "eng"},
            blocked="True",
        )
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4o"
        assert result["dept"] == "eng"
        assert result["blocked"] == "True"

    def test_rejects_provider_override(self):
        """Cannot override 'provider' via custom metadata."""
        with pytest.raises(ValueError, match="provider"):
            build_metadata("openai", "gpt-4o", custom_metadata={"provider": "fake"})

    def test_rejects_model_override(self):
        """Cannot override 'model' via custom metadata."""
        with pytest.raises(ValueError, match="model"):
            build_metadata("openai", "gpt-4o", custom_metadata={"model": "fake"})

    def test_none_custom_metadata(self):
        """None custom_metadata is handled gracefully."""
        result = build_metadata("anthropic", "claude-3", custom_metadata=None)
        assert result == {"provider": "anthropic", "model": "claude-3"}

    def test_empty_custom_metadata(self):
        """Empty dict custom_metadata is handled gracefully."""
        result = build_metadata("gemini", "gemini-2.5-flash", custom_metadata={})
        assert result == {"provider": "gemini", "model": "gemini-2.5-flash"}


# ─── handle_blocked_request with custom_metadata Tests ─────────────────────


class TestHandleBlockedRequestCustomMetadata:
    """Tests for handle_blocked_request with custom_metadata parameter."""

    def test_custom_metadata_in_blocked_attestation(self, signing_seed):
        """Custom metadata is included in blocked request attestation."""
        from glacis import Glacis

        glacis = Glacis(mode="offline", signing_seed=signing_seed)

        with patch.object(glacis, "attest", wraps=glacis.attest) as mock_attest:
            with pytest.raises(GlacisBlockedError):
                handle_blocked_request(
                    glacis_client=glacis,
                    service_id="test",
                    input_data={"model": "gpt-4", "messages": []},
                    control_plane_results=None,
                    provider="openai",
                    model="gpt-4",
                    blocking_control_type="jailbreak",
                    blocking_score=0.95,
                    debug=False,
                    custom_metadata={"department": "legal"},
                )

        call_kwargs = mock_attest.call_args[1]
        metadata = call_kwargs["metadata"]
        assert metadata["provider"] == "openai"
        assert metadata["model"] == "gpt-4"
        assert metadata["blocked"] == "True"
        assert metadata["department"] == "legal"


# ─── IntegrationContext Tests ───────────────────────────────────────────────


class TestIntegrationContext:
    """Tests for IntegrationContext dataclass."""

    def test_construction(self):
        """IntegrationContext can be constructed with all fields."""
        ctx = IntegrationContext(
            glacis=MagicMock(),
            cfg=_default_config(),
            controls_runner=None,
            effective_service_id="openai",
            storage_backend="sqlite",
            storage_path=None,
            output_block_action="block",
            custom_metadata={"dept": "eng"},
            debug=False,
        )
        assert ctx.effective_service_id == "openai"
        assert ctx.custom_metadata == {"dept": "eng"}
        assert ctx.output_block_action == "block"


# ─── setup_integration Tests ───────────────────────────────────────────────


class TestSetupIntegration:
    """Tests for setup_integration helper."""

    def test_returns_integration_context(self, signing_seed):
        """setup_integration returns a fully populated IntegrationContext."""
        ctx = setup_integration(
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
            metadata={"dept": "eng"},
        )
        assert isinstance(ctx, IntegrationContext)
        assert ctx.effective_service_id == "openai"
        assert ctx.custom_metadata == {"dept": "eng"}
        assert ctx.debug is False

    def test_requires_signing_seed_offline(self):
        """Offline mode without signing_seed raises ValueError."""
        with pytest.raises(ValueError, match="signing_seed"):
            setup_integration(
                config=None,
                offline=True,
                glacis_api_key=None,
                glacis_base_url="https://api.glacis.io",
                default_service_id="openai",
                service_id="openai",
                debug=False,
                signing_seed=None,
                policy_key=None,
                input_controls=None,
                output_controls=None,
                metadata=None,
            )

    def test_requires_api_key_online(self):
        """Online mode without glacis_api_key raises ValueError."""
        with pytest.raises(ValueError, match="api_key|glacis_api_key"):
            setup_integration(
                config=None,
                offline=False,
                glacis_api_key=None,
                glacis_base_url="https://api.glacis.io",
                default_service_id="openai",
                service_id="openai",
                debug=False,
                signing_seed=None,
                policy_key=None,
                input_controls=None,
                output_controls=None,
                metadata=None,
            )


# ─── check_input_block Tests ───────────────────────────────────────────────


class TestCheckInputBlock:
    """Tests for check_input_block helper."""

    def _make_ctx(self, signing_seed):
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

    def test_noop_when_not_blocking(self, signing_seed):
        """Does nothing when accumulator has no block."""
        ctx = self._make_ctx(signing_seed)
        acc = ControlResultsAccumulator()

        # Should not raise
        check_input_block(ctx, acc, "gpt-4", "openai", {"model": "gpt-4"})

    def test_raises_when_blocking(self, signing_seed):
        """Raises GlacisBlockedError when accumulator has a block."""
        ctx = self._make_ctx(signing_seed)
        acc = ControlResultsAccumulator()
        results = [_make_control_result(control_type="jailbreak", action="block", score=0.95)]
        stage = _make_stage_result(results=results, should_block=True)
        acc.update_from_stage(stage, "input")

        with pytest.raises(GlacisBlockedError) as exc_info:
            check_input_block(
                ctx, acc, "gpt-4", "openai",
                {"model": "gpt-4", "messages": []},
            )

        assert exc_info.value.control_type == "jailbreak"
        assert exc_info.value.score == 0.95


# ─── check_output_block Tests ──────────────────────────────────────────────


class TestCheckOutputBlock:
    """Tests for check_output_block helper."""

    def _make_ctx(self, signing_seed, output_block_action="block"):
        ctx = setup_integration(
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
        # Override block action for testing
        ctx.output_block_action = output_block_action
        return ctx

    def test_noop_when_not_blocking(self, signing_seed):
        """Does nothing when accumulator has no block."""
        ctx = self._make_ctx(signing_seed)
        acc = ControlResultsAccumulator()
        cpr = create_control_plane_results(acc, ctx.cfg, "gpt-4", "openai")

        # Should not raise
        check_output_block(ctx, acc, "gpt-4", "openai", {"model": "gpt-4"}, cpr)

    def test_noop_when_forward_action(self, signing_seed):
        """Does nothing when output_block_action is 'forward'."""
        ctx = self._make_ctx(signing_seed, output_block_action="forward")
        # Need a controls_runner to pass the guard condition
        ctx.controls_runner = MagicMock()
        ctx.controls_runner.has_output_controls = True

        acc = ControlResultsAccumulator()
        results = [_make_control_result(control_type="pii", action="block", score=0.9)]
        stage = _make_stage_result(results=results, should_block=True)
        acc.update_from_stage(stage, "output")
        cpr = create_control_plane_results(acc, ctx.cfg, "gpt-4", "openai")

        # Should not raise because action is "forward"
        check_output_block(ctx, acc, "gpt-4", "openai", {"model": "gpt-4"}, cpr)

    def test_raises_when_block_action(self, signing_seed):
        """Raises GlacisBlockedError when output_block_action is 'block'."""
        ctx = self._make_ctx(signing_seed, output_block_action="block")
        ctx.controls_runner = MagicMock()
        ctx.controls_runner.has_output_controls = True

        acc = ControlResultsAccumulator()
        results = [_make_control_result(control_type="pii", action="block", score=0.9)]
        stage = _make_stage_result(results=results, should_block=True)
        acc.update_from_stage(stage, "output")
        cpr = create_control_plane_results(acc, ctx.cfg, "gpt-4", "openai")

        with pytest.raises(GlacisBlockedError) as exc_info:
            check_output_block(
                ctx, acc, "gpt-4", "openai",
                {"model": "gpt-4", "messages": []}, cpr,
            )

        assert exc_info.value.control_type == "pii"

    def test_noop_when_no_controls_runner(self, signing_seed):
        """Does nothing when controls_runner is None."""
        ctx = self._make_ctx(signing_seed)
        # controls_runner is None by default (no controls enabled)
        acc = ControlResultsAccumulator()
        acc.should_block = True
        cpr = create_control_plane_results(acc, ctx.cfg, "gpt-4", "openai")

        # Should not raise
        check_output_block(ctx, acc, "gpt-4", "openai", {"model": "gpt-4"}, cpr)


# ─── attest_and_store Tests ────────────────────────────────────────────────


class TestAttestAndStore:
    """Tests for attest_and_store helper."""

    def _make_ctx(self, signing_seed):
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

    def test_happy_path(self, signing_seed):
        """Attest and store succeeds and sets last receipt."""
        from glacis.integrations.base import get_last_receipt

        ctx = self._make_ctx(signing_seed)
        acc = ControlResultsAccumulator()
        cpr = create_control_plane_results(acc, ctx.cfg, "gpt-4", "openai")

        attest_and_store(
            ctx,
            input_data={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            output_data={"model": "gpt-4", "choices": []},
            metadata={"provider": "openai", "model": "gpt-4"},
            control_plane_results=cpr,
        )

        receipt = get_last_receipt()
        assert receipt is not None
        assert receipt.id is not None

    def test_exception_swallowed(self):
        """Attestation errors are swallowed silently."""
        ctx = IntegrationContext(
            glacis=MagicMock(),
            cfg=_default_config(),
            controls_runner=None,
            effective_service_id="openai",
            storage_backend="sqlite",
            storage_path=None,
            output_block_action="block",
            custom_metadata=None,
            debug=False,
        )
        ctx.glacis.attest.side_effect = Exception("Network error")

        # Should not raise
        attest_and_store(
            ctx,
            input_data={"model": "gpt-4"},
            output_data={},
            metadata={"provider": "openai", "model": "gpt-4"},
            control_plane_results=None,
        )

    def test_debug_prints_on_failure(self, capsys):
        """Debug mode prints attestation failure."""
        ctx = IntegrationContext(
            glacis=MagicMock(),
            cfg=_default_config(),
            controls_runner=None,
            effective_service_id="openai",
            storage_backend="sqlite",
            storage_path=None,
            output_block_action="block",
            custom_metadata=None,
            debug=True,
        )
        ctx.glacis.attest.side_effect = Exception("Network error")

        attest_and_store(
            ctx,
            input_data={"model": "gpt-4"},
            output_data={},
            metadata={"provider": "openai", "model": "gpt-4"},
            control_plane_results=None,
        )

        captured = capsys.readouterr()
        assert "[glacis] Attestation failed:" in captured.out
