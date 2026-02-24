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
    _map_control_type,
    create_control_plane_results,
    handle_blocked_request,
    initialize_config,
    run_input_controls,
    run_output_controls,
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
