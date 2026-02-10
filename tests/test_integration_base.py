"""
Unit tests for glacis/integrations/base.py shared utilities.

Tests ControlResultsAccumulator, process_text_for_controls,
create_control_plane_results_from_accumulator, handle_blocked_request,
and initialize_config.
"""

from unittest.mock import MagicMock, patch

import pytest

from glacis.integrations.base import (
    ControlResultsAccumulator,
    GlacisBlockedError,
    create_control_plane_results_from_accumulator,
    handle_blocked_request,
    initialize_config,
    process_text_for_controls,
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


def _pii_result(categories=None, latency_ms=5):
    """Create a mock PII detection result."""
    r = MagicMock()
    r.control_type = "pii"
    r.detected = True
    r.categories = categories or ["US_SSN"]
    r.latency_ms = latency_ms
    r.action = "flag"
    r.score = None
    return r


def _jailbreak_result(detected=True, score=0.95, action="block", latency_ms=10):
    """Create a mock jailbreak detection result."""
    r = MagicMock()
    r.control_type = "jailbreak"
    r.detected = detected
    r.score = score
    r.action = action
    r.latency_ms = latency_ms
    r.categories = []
    return r


def _default_config():
    """Load a default GlacisConfig."""
    from glacis.config import GlacisConfig

    return GlacisConfig()


# ─── ControlResultsAccumulator Tests ─────────────────────────────────────────


class TestControlResultsAccumulator:
    """Tests for ControlResultsAccumulator state management."""

    def test_accumulator_initial_state(self):
        """Fresh accumulator has no detections and should not block."""
        acc = ControlResultsAccumulator()
        assert acc._pii_detected is False
        assert acc._pii_categories == []
        assert acc._jailbreak_detected is False
        assert acc._jailbreak_score == 0.0
        assert acc._jailbreak_action == "pass"
        assert acc.control_executions == []
        assert acc.should_block is False

    def test_accumulator_pii_detected(self):
        """PII result sets detected flag and populates categories."""
        acc = ControlResultsAccumulator()
        acc.update([_pii_result(categories=["US_SSN", "EMAIL_ADDRESS"])])

        assert acc._pii_detected is True
        assert "US_SSN" in acc._pii_categories
        assert "EMAIL_ADDRESS" in acc._pii_categories
        assert len(acc.control_executions) == 1
        assert acc.control_executions[0].type == "pii"
        assert acc.control_executions[0].status == "flag"

    def test_accumulator_pii_categories_merge(self):
        """Multiple PII results merge categories into sorted union."""
        acc = ControlResultsAccumulator()
        acc.update([_pii_result(categories=["US_SSN"])])
        acc.update([_pii_result(categories=["EMAIL_ADDRESS", "US_SSN"])])

        assert acc._pii_categories == ["EMAIL_ADDRESS", "US_SSN"]
        assert len(acc.control_executions) == 2

    def test_accumulator_jailbreak_detected(self):
        """Jailbreak result sets detected, score, and action."""
        acc = ControlResultsAccumulator()
        acc.update([_jailbreak_result(detected=True, score=0.85, action="flag")])

        assert acc._jailbreak_detected is True
        assert acc._jailbreak_score == 0.85
        assert acc._jailbreak_action == "flag"
        assert len(acc.control_executions) == 1
        assert acc.control_executions[0].type == "jailbreak"

    def test_accumulator_jailbreak_block_sets_should_block(self):
        """Jailbreak with action='block' sets should_block=True."""
        acc = ControlResultsAccumulator()
        acc.update([_jailbreak_result(action="block")])

        assert acc.should_block is True

    def test_accumulator_jailbreak_flag_does_not_block(self):
        """Jailbreak with action='flag' does not block."""
        acc = ControlResultsAccumulator()
        acc.update([_jailbreak_result(action="flag")])

        assert acc.should_block is False

    def test_accumulator_jailbreak_higher_score_wins(self):
        """Higher jailbreak score replaces lower score."""
        acc = ControlResultsAccumulator()
        acc.update([_jailbreak_result(score=0.3, action="flag")])
        acc.update([_jailbreak_result(score=0.9, action="block")])

        assert acc._jailbreak_score == 0.9
        assert acc._jailbreak_action == "block"

    def test_accumulator_control_executions_populated(self):
        """Control executions list grows with each update."""
        acc = ControlResultsAccumulator()
        acc.update([_pii_result()])
        assert len(acc.control_executions) == 1

        acc.update([_jailbreak_result()])
        assert len(acc.control_executions) == 2

        acc.update([_pii_result(categories=["PHONE_NUMBER"])])
        assert len(acc.control_executions) == 3


# ─── process_text_for_controls Tests ─────────────────────────────────────────


class TestProcessTextForControls:
    """Tests for process_text_for_controls utility."""

    def test_process_text_runs_and_updates_accumulator(self):
        """Runs controls, updates accumulator, returns modified text."""
        pii = _pii_result()
        runner = MagicMock()
        runner.run.return_value = [pii]
        runner.get_final_text.return_value = "My SSN is [US_SSN]"

        acc = ControlResultsAccumulator()
        result = process_text_for_controls(runner, "My SSN is 123-45-6789", acc)

        assert result == "My SSN is [US_SSN]"
        assert acc._pii_detected is True
        runner.run.assert_called_once_with("My SSN is 123-45-6789")

    def test_process_text_returns_original_when_no_modification(self):
        """Returns original text when get_final_text returns None."""
        runner = MagicMock()
        runner.run.return_value = []
        runner.get_final_text.return_value = None

        acc = ControlResultsAccumulator()
        result = process_text_for_controls(runner, "Hello world", acc)

        assert result == "Hello world"


# ─── create_control_plane_results_from_accumulator Tests ─────────────────────


class TestCreateControlPlaneResults:
    """Tests for building ControlPlaneResults from accumulator state."""

    def test_forwarded_when_no_detections(self):
        """No detections -> action='forwarded', trigger=None."""
        acc = ControlResultsAccumulator()
        cfg = _default_config()

        result = create_control_plane_results_from_accumulator(
            acc, cfg, "gpt-4", "openai"
        )

        assert result.determination.action == "forwarded"
        assert result.determination.trigger is None
        assert result.determination.confidence == 1.0

    def test_redacted_when_pii_detected(self):
        """PII detected -> action='redacted', trigger='pii'."""
        acc = ControlResultsAccumulator()
        acc.update([_pii_result()])
        cfg = _default_config()

        result = create_control_plane_results_from_accumulator(
            acc, cfg, "gpt-4", "openai"
        )

        assert result.determination.action == "redacted"
        assert result.determination.trigger == "pii"

    def test_blocked_when_jailbreak_blocks(self):
        """Jailbreak with block action -> action='blocked', trigger='jailbreak'."""
        acc = ControlResultsAccumulator()
        acc.update([_jailbreak_result(action="block")])
        cfg = _default_config()

        result = create_control_plane_results_from_accumulator(
            acc, cfg, "gpt-4", "openai"
        )

        assert result.determination.action == "blocked"
        assert result.determination.trigger == "jailbreak"

    def test_forwarded_when_jailbreak_flags(self):
        """Jailbreak with flag action -> action='forwarded', trigger='jailbreak'."""
        acc = ControlResultsAccumulator()
        acc.update([_jailbreak_result(action="flag")])
        cfg = _default_config()

        result = create_control_plane_results_from_accumulator(
            acc, cfg, "gpt-4", "openai"
        )

        assert result.determination.action == "forwarded"
        assert result.determination.trigger == "jailbreak"

    def test_policy_context_from_config(self):
        """Policy context uses values from config."""
        acc = ControlResultsAccumulator()
        cfg = _default_config()

        result = create_control_plane_results_from_accumulator(
            acc, cfg, "gpt-4", "openai"
        )

        assert result.policy.id == "default"
        assert result.policy.version == "1.0"
        assert result.policy.model.model_id == "gpt-4"
        assert result.policy.model.provider == "openai"
        assert result.policy.scope.environment == "development"


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
                jailbreak_score=0.95,
                debug=False,
            )

        assert exc_info.value.control_type == "jailbreak"
        assert exc_info.value.score == 0.95
        assert "0.95" in str(exc_info.value)

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
                    jailbreak_score=0.95,
                    debug=False,
                )

        mock_attest.assert_called_once()

    def test_stores_evidence(self, signing_seed):
        """store_evidence is called for blocked requests."""
        from glacis import Glacis

        glacis = Glacis(mode="offline", signing_seed=signing_seed)

        with patch("glacis.integrations.base.store_evidence") as mock_store:
            with pytest.raises(GlacisBlockedError):
                handle_blocked_request(
                    glacis_client=glacis,
                    service_id="test",
                    input_data={"model": "gpt-4", "messages": []},
                    control_plane_results=None,
                    provider="openai",
                    model="gpt-4",
                    jailbreak_score=0.95,
                    debug=False,
                )

        mock_store.assert_called_once()

    def test_attestation_failure_still_raises(self, signing_seed):
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
                jailbreak_score=0.80,
                debug=False,
            )

        assert exc_info.value.score == 0.80


# ─── initialize_config Tests ─────────────────────────────────────────────────


class TestInitializeConfig:
    """Tests for initialize_config parameter precedence."""

    def test_redaction_true_enables_fast_mode(self):
        """redaction=True enables PII with fast mode."""
        cfg, _, _ = initialize_config(
            config_path=None,
            redaction=True,
            offline=True,
            glacis_api_key=None,
            default_service_id="openai",
            service_id="openai",
        )

        assert cfg.controls.pii_phi.enabled is True
        assert cfg.controls.pii_phi.mode == "fast"

    def test_redaction_false_disables(self):
        """redaction=False disables PII."""
        cfg, _, _ = initialize_config(
            config_path=None,
            redaction=False,
            offline=True,
            glacis_api_key=None,
            default_service_id="openai",
            service_id="openai",
        )

        assert cfg.controls.pii_phi.enabled is False

    def test_redaction_string_sets_mode(self):
        """redaction='full' enables PII with full mode."""
        cfg, _, _ = initialize_config(
            config_path=None,
            redaction="full",
            offline=True,
            glacis_api_key=None,
            default_service_id="openai",
            service_id="openai",
        )

        assert cfg.controls.pii_phi.enabled is True
        assert cfg.controls.pii_phi.mode == "full"

    def test_offline_explicit_overrides_config(self):
        """Explicit offline=True overrides config default."""
        _, effective_offline, _ = initialize_config(
            config_path=None,
            redaction=None,
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
            redaction=None,
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
            redaction=None,
            offline=True,
            glacis_api_key=None,
            default_service_id="openai",
            service_id="openai",
        )

        # Default config has attestation.service_id = "openai"
        assert effective_service_id == "openai"
