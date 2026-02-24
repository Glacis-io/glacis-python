"""
Comprehensive tests for the GLACIS LLM Judge Pipeline.

Covers:
- JudgeVerdict / ReviewResult model validation
- JudgeRunner aggregation, consensus, and error handling
- OpenAIJudge / AnthropicJudge (mocked API calls)
- should_review() L1 sampling gate
- Full pipeline (mocked LLM + real Glacis offline attestation)
- Human review with supersedes
- Edge cases
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from glacis import Glacis
from glacis.judges import BaseJudge, JudgeRunner, JudgeVerdict, JudgesConfig, ReviewResult
from glacis.models import Attestation, Review, SamplingDecision


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_qa_pair() -> dict[str, Any]:
    """A typical QA pair for judge evaluation."""
    return {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
    }


@pytest.fixture
def sample_reference() -> str:
    """Reference document for fact-checking."""
    return "France is a country in Western Europe. Its capital city is Paris."


class StubJudge(BaseJudge):
    """Deterministic judge for testing JudgeRunner logic."""

    def __init__(self, judge_id: str, score: float, rationale: str = "stub"):
        self.judge_id = judge_id
        self._score = score
        self._rationale = rationale

    def evaluate(self, item, reference=None, rubric=None) -> JudgeVerdict:
        return JudgeVerdict(
            judge_id=self.judge_id,
            score=self._score,
            rationale=self._rationale,
            latency_ms=10,
        )


class FailingJudge(BaseJudge):
    """Judge that always raises an exception."""

    judge_id = "failing-judge"

    def evaluate(self, item, reference=None, rubric=None) -> JudgeVerdict:
        raise RuntimeError("judge exploded")


# =============================================================================
# JudgeVerdict model tests
# =============================================================================


class TestJudgeVerdict:
    """Test JudgeVerdict Pydantic model validation."""

    def test_valid_verdict(self):
        v = JudgeVerdict(judge_id="test", score=2.5, rationale="Good")
        assert v.judge_id == "test"
        assert v.score == 2.5
        assert v.rationale == "Good"
        assert v.latency_ms == 0
        assert v.metadata == {}

    def test_score_bounds_valid(self):
        """Scores 0.0, 1.5, and 3.0 should all be valid."""
        for s in [0.0, 1.0, 1.5, 2.0, 3.0]:
            v = JudgeVerdict(judge_id="t", score=s, rationale="ok")
            assert v.score == s

    def test_score_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            JudgeVerdict(judge_id="t", score=-0.1, rationale="bad")

    def test_score_above_three_allowed(self):
        """Scores above 3.0 are valid — max is enforced by config, not the model."""
        v = JudgeVerdict(judge_id="t", score=5.0, rationale="custom scale")
        assert v.score == 5.0

    def test_metadata_preserved(self):
        v = JudgeVerdict(
            judge_id="t", score=1.0, rationale="ok",
            metadata={"model": "gpt-4o-mini", "tokens": 42},
        )
        assert v.metadata["model"] == "gpt-4o-mini"
        assert v.metadata["tokens"] == 42


# =============================================================================
# ReviewResult model tests
# =============================================================================


class TestReviewResult:
    """Test ReviewResult model and recommendation logic."""

    def _make_result(self, scores: list[float], threshold: float = 1.0) -> ReviewResult:
        """Helper to build ReviewResult from raw scores."""
        verdicts = [
            JudgeVerdict(judge_id=f"j{i}", score=s, rationale="test")
            for i, s in enumerate(scores)
        ]
        final = sum(scores) / len(scores) if scores else 0.0
        consensus = (
            (max(scores) - min(scores)) <= threshold
            if len(scores) >= 2
            else True
        )
        if final >= 2.0:
            rec = "uphold"
        elif final >= 1.0:
            rec = "borderline"
        else:
            rec = "escalate"

        return ReviewResult(
            verdicts=verdicts,
            final_score=round(final, 4),
            consensus=consensus,
            recommendation=rec,
        )

    def test_consensus_true(self):
        result = self._make_result([2.5, 3.0])
        assert result.consensus is True

    def test_consensus_false(self):
        result = self._make_result([0.0, 3.0])
        assert result.consensus is False

    def test_recommendation_uphold(self):
        result = self._make_result([2.0, 3.0])
        assert result.recommendation == "uphold"

    def test_recommendation_borderline(self):
        result = self._make_result([1.0, 1.5])
        assert result.recommendation == "borderline"

    def test_recommendation_escalate(self):
        result = self._make_result([0.0, 0.5])
        assert result.recommendation == "escalate"

    def test_recommendation_exact_boundary_uphold(self):
        """Score == 2.0 should be uphold."""
        result = self._make_result([2.0, 2.0])
        assert result.recommendation == "uphold"

    def test_recommendation_exact_boundary_borderline(self):
        """Score == 1.0 should be borderline."""
        result = self._make_result([1.0, 1.0])
        assert result.recommendation == "borderline"


# =============================================================================
# JudgeRunner tests
# =============================================================================


class TestJudgeRunner:
    """Test JudgeRunner aggregation and error handling."""

    def test_averages_two_judges(self, sample_qa_pair):
        runner = JudgeRunner(judges=[
            StubJudge("a", 2.0),
            StubJudge("b", 3.0),
        ])
        result = runner.run(sample_qa_pair)
        assert result.final_score == 2.5
        assert result.recommendation == "uphold"
        assert len(result.verdicts) == 2

    def test_averages_single_judge(self, sample_qa_pair):
        runner = JudgeRunner(judges=[StubJudge("a", 1.5)])
        result = runner.run(sample_qa_pair)
        assert result.final_score == 1.5
        assert result.recommendation == "borderline"
        assert result.consensus is True

    def test_consensus_threshold_custom(self, sample_qa_pair):
        """Custom threshold=0.5 flags disagreement of 1.0."""
        runner = JudgeRunner(
            judges=[StubJudge("a", 1.0), StubJudge("b", 2.0)],
            consensus_threshold=0.5,
        )
        result = runner.run(sample_qa_pair)
        assert result.consensus is False

    def test_all_max_scores(self, sample_qa_pair):
        runner = JudgeRunner(judges=[StubJudge("a", 3.0), StubJudge("b", 3.0)])
        result = runner.run(sample_qa_pair)
        assert result.final_score == 3.0
        assert result.consensus is True
        assert result.recommendation == "uphold"

    def test_all_zero_scores(self, sample_qa_pair):
        runner = JudgeRunner(judges=[StubJudge("a", 0.0), StubJudge("b", 0.0)])
        result = runner.run(sample_qa_pair)
        assert result.final_score == 0.0
        assert result.recommendation == "escalate"

    def test_max_disagreement(self, sample_qa_pair):
        runner = JudgeRunner(judges=[StubJudge("a", 0.0), StubJudge("b", 3.0)])
        result = runner.run(sample_qa_pair)
        assert result.final_score == 1.5
        assert result.consensus is False

    def test_mixed_scores(self, sample_qa_pair):
        runner = JudgeRunner(judges=[StubJudge("a", 1.0), StubJudge("b", 2.0)])
        result = runner.run(sample_qa_pair)
        assert result.final_score == 1.5
        assert result.recommendation == "borderline"

    def test_one_judge_fails(self, sample_qa_pair):
        """One failing judge should not crash the runner."""
        runner = JudgeRunner(judges=[StubJudge("a", 3.0), FailingJudge()])
        result = runner.run(sample_qa_pair)
        assert len(result.verdicts) == 2
        assert result.verdicts[0].score == 3.0
        assert result.verdicts[1].score == 0.0
        assert "Judge error" in result.verdicts[1].rationale

    def test_both_judges_fail(self, sample_qa_pair):
        runner = JudgeRunner(judges=[FailingJudge(), FailingJudge()])
        result = runner.run(sample_qa_pair)
        assert result.final_score == 0.0
        assert result.recommendation == "escalate"

    def test_reference_passed_through(self, sample_qa_pair, sample_reference):
        """Verify reference is forwarded to judges."""

        class CapturingJudge(BaseJudge):
            judge_id = "cap"
            captured_ref = None

            def evaluate(self, item, reference=None, rubric=None):
                CapturingJudge.captured_ref = reference
                return JudgeVerdict(judge_id=self.judge_id, score=2.0, rationale="ok")

        runner = JudgeRunner(judges=[CapturingJudge()])
        runner.run(sample_qa_pair, reference=sample_reference)
        assert CapturingJudge.captured_ref == sample_reference

    def test_rubric_passed_through(self, sample_qa_pair):
        """Verify custom rubric is forwarded to judges."""

        class CapturingJudge(BaseJudge):
            judge_id = "cap"
            captured_rubric = None

            def evaluate(self, item, reference=None, rubric=None):
                CapturingJudge.captured_rubric = rubric
                return JudgeVerdict(judge_id=self.judge_id, score=2.0, rationale="ok")

        custom_rubric = "Rate 0-3 on style only."
        runner = JudgeRunner(judges=[CapturingJudge()])
        runner.run(sample_qa_pair, rubric=custom_rubric)
        assert CapturingJudge.captured_rubric == custom_rubric

    def test_close_calls_all_judges(self):
        """Runner.close() should call close() on each judge."""

        class TrackingJudge(BaseJudge):
            judge_id = "track"
            closed = False

            def evaluate(self, item, reference=None, rubric=None):
                return JudgeVerdict(judge_id=self.judge_id, score=1.0, rationale="ok")

            def close(self):
                TrackingJudge.closed = True

        j = TrackingJudge()
        runner = JudgeRunner(judges=[j])
        runner.close()
        assert TrackingJudge.closed is True


# =============================================================================
# Prompt building + verdict parsing (from judges/qa_explain_judge.py)
# =============================================================================


class TestPromptBuilding:
    """Test _build_prompt helper."""

    def test_includes_reference(self):
        from judges.qa_explain_judge import _build_prompt

        prompt = _build_prompt(
            {"question": "Q?", "answer": "A."},
            reference="Ref doc here.",
            rubric=None,
        )
        assert "Ref doc here." in prompt
        assert "Question: Q?" in prompt
        assert "Answer: A." in prompt

    def test_no_reference(self):
        from judges.qa_explain_judge import _build_prompt

        prompt = _build_prompt(
            {"question": "Q?", "answer": "A."},
            reference=None,
            rubric=None,
        )
        assert "Reference document" not in prompt
        assert "Question: Q?" in prompt

    def test_custom_rubric_overrides_default(self):
        from judges.qa_explain_judge import SCORING_RUBRIC, _build_prompt

        custom = "Rate 0-3 on vibes."
        prompt = _build_prompt(
            {"question": "Q?", "answer": "A."},
            reference=None,
            rubric=custom,
        )
        assert custom in prompt
        assert SCORING_RUBRIC not in prompt

    def test_default_rubric_used_when_none(self):
        from judges.qa_explain_judge import SCORING_RUBRIC, _build_prompt

        prompt = _build_prompt(
            {"question": "Q?", "answer": "A."},
            reference=None,
            rubric=None,
        )
        assert "INCORRECT" in prompt  # Part of the default rubric


class TestVerdictParsing:
    """Test _parse_verdict helper."""

    def test_valid_json(self):
        from judges.qa_explain_judge import _parse_verdict

        v = _parse_verdict("test", '{"score": 3, "rationale": "correct"}', 100)
        assert v.score == 3.0
        assert v.rationale == "correct"
        assert v.latency_ms == 100

    def test_json_with_markdown_fences(self):
        from judges.qa_explain_judge import _parse_verdict

        raw = '```json\n{"score": 2, "rationale": "mostly good"}\n```'
        v = _parse_verdict("test", raw, 50)
        assert v.score == 2.0
        assert v.rationale == "mostly good"

    def test_json_with_plain_fences(self):
        from judges.qa_explain_judge import _parse_verdict

        raw = '```\n{"score": 1, "rationale": "partial"}\n```'
        v = _parse_verdict("test", raw, 50)
        assert v.score == 1.0

    def test_malformed_json_returns_zero(self):
        from judges.qa_explain_judge import _parse_verdict

        v = _parse_verdict("test", "This is not JSON at all.", 50)
        assert v.score == 0.0
        assert "Parse error" in v.rationale

    def test_missing_score_key_returns_zero(self):
        from judges.qa_explain_judge import _parse_verdict

        v = _parse_verdict("test", '{"rationale": "no score"}', 50)
        assert v.score == 0.0
        assert "parse_error" in v.metadata

    def test_score_clamped_above_three(self):
        from judges.qa_explain_judge import _parse_verdict

        v = _parse_verdict("test", '{"score": 5, "rationale": "too high"}', 50)
        assert v.score == 3.0

    def test_score_clamped_below_zero(self):
        from judges.qa_explain_judge import _parse_verdict

        v = _parse_verdict("test", '{"score": -1, "rationale": "too low"}', 50)
        assert v.score == 0.0

    def test_empty_response(self):
        from judges.qa_explain_judge import _parse_verdict

        v = _parse_verdict("test", "", 50)
        assert v.score == 0.0
        assert "Parse error" in v.rationale


# =============================================================================
# Mocked OpenAIJudge tests
# =============================================================================


class TestOpenAIJudge:
    """Test OpenAIJudge with mocked OpenAI client."""

    def _make_mock_response(self, content: str) -> MagicMock:
        """Build a mock OpenAI ChatCompletion response."""
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        return response

    def test_parses_valid_json(self, sample_qa_pair, sample_reference):
        from judges.qa_explain_judge import OpenAIJudge

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = self._make_mock_response(
                '{"score": 3, "rationale": "fully correct"}'
            )

            judge = OpenAIJudge(api_key="test-key")
            verdict = judge.evaluate(sample_qa_pair, reference=sample_reference)

            assert verdict.score == 3.0
            assert verdict.rationale == "fully correct"
            assert verdict.judge_id == "gpt-4o-mini"

    def test_handles_malformed_json(self, sample_qa_pair):
        from judges.qa_explain_judge import OpenAIJudge

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = self._make_mock_response(
                "I think this answer is correct because..."
            )

            judge = OpenAIJudge(api_key="test-key")
            verdict = judge.evaluate(sample_qa_pair)

            assert verdict.score == 0.0
            assert "Parse error" in verdict.rationale

    def test_handles_api_error(self, sample_qa_pair):
        from judges.qa_explain_judge import OpenAIJudge

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("Connection refused")

            judge = OpenAIJudge(api_key="test-key")
            verdict = judge.evaluate(sample_qa_pair)

            assert verdict.score == 0.0
            assert "API error" in verdict.rationale
            assert verdict.metadata.get("error") == "Connection refused"

    def test_tracks_latency(self, sample_qa_pair):
        from judges.qa_explain_judge import OpenAIJudge

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = self._make_mock_response(
                '{"score": 2, "rationale": "ok"}'
            )

            judge = OpenAIJudge(api_key="test-key")
            verdict = judge.evaluate(sample_qa_pair)

            assert verdict.latency_ms >= 0

    def test_custom_model_id(self, sample_qa_pair):
        from judges.qa_explain_judge import OpenAIJudge

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = self._make_mock_response(
                '{"score": 2, "rationale": "ok"}'
            )

            judge = OpenAIJudge(api_key="test-key", model="gpt-4o")
            assert judge.judge_id == "gpt-4o"

            verdict = judge.evaluate(sample_qa_pair)
            assert verdict.judge_id == "gpt-4o"

    def test_custom_rubric_in_prompt(self, sample_qa_pair):
        from judges.qa_explain_judge import OpenAIJudge

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = self._make_mock_response(
                '{"score": 1, "rationale": "custom"}'
            )

            judge = OpenAIJudge(api_key="test-key")
            judge.evaluate(sample_qa_pair, rubric="Rate on vibes only.")

            # Verify the rubric made it into the prompt
            call_args = mock_client.chat.completions.create.call_args
            prompt_content = call_args.kwargs["messages"][0]["content"]
            assert "Rate on vibes only." in prompt_content


# =============================================================================
# Mocked AnthropicJudge tests
# =============================================================================


class TestAnthropicJudge:
    """Test AnthropicJudge with mocked Anthropic client."""

    def _make_mock_response(self, text: str) -> MagicMock:
        """Build a mock Anthropic messages.create response."""
        content_block = MagicMock()
        content_block.text = text
        response = MagicMock()
        response.content = [content_block]
        return response

    def test_parses_valid_json(self, sample_qa_pair, sample_reference):
        from judges.qa_explain_judge import AnthropicJudge

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client
            mock_client.messages.create.return_value = self._make_mock_response(
                '{"score": 3, "rationale": "fully correct"}'
            )

            judge = AnthropicJudge(api_key="test-key")
            verdict = judge.evaluate(sample_qa_pair, reference=sample_reference)

            assert verdict.score == 3.0
            assert verdict.rationale == "fully correct"
            assert verdict.judge_id == "claude-haiku-4-5-20251001"

    def test_handles_malformed_json(self, sample_qa_pair):
        from judges.qa_explain_judge import AnthropicJudge

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client
            mock_client.messages.create.return_value = self._make_mock_response(
                "The answer looks good to me."
            )

            judge = AnthropicJudge(api_key="test-key")
            verdict = judge.evaluate(sample_qa_pair)

            assert verdict.score == 0.0
            assert "Parse error" in verdict.rationale

    def test_handles_api_error(self, sample_qa_pair):
        from judges.qa_explain_judge import AnthropicJudge

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("Rate limited")

            judge = AnthropicJudge(api_key="test-key")
            verdict = judge.evaluate(sample_qa_pair)

            assert verdict.score == 0.0
            assert "API error" in verdict.rationale
            assert verdict.metadata.get("error") == "Rate limited"

    def test_tracks_latency(self, sample_qa_pair):
        from judges.qa_explain_judge import AnthropicJudge

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client
            mock_client.messages.create.return_value = self._make_mock_response(
                '{"score": 2, "rationale": "ok"}'
            )

            judge = AnthropicJudge(api_key="test-key")
            verdict = judge.evaluate(sample_qa_pair)

            assert verdict.latency_ms >= 0

    def test_custom_model_id(self, sample_qa_pair):
        from judges.qa_explain_judge import AnthropicJudge

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client
            mock_client.messages.create.return_value = self._make_mock_response(
                '{"score": 2, "rationale": "ok"}'
            )

            judge = AnthropicJudge(api_key="test-key", model="claude-sonnet-4-6")
            assert judge.judge_id == "claude-sonnet-4-6"

            verdict = judge.evaluate(sample_qa_pair)
            assert verdict.judge_id == "claude-sonnet-4-6"

    def test_empty_content_handled(self, sample_qa_pair):
        """Empty response.content should not crash."""
        from judges.qa_explain_judge import AnthropicJudge

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client
            response = MagicMock()
            response.content = []
            mock_client.messages.create.return_value = response

            judge = AnthropicJudge(api_key="test-key")
            verdict = judge.evaluate(sample_qa_pair)

            assert verdict.score == 0.0


# =============================================================================
# should_review() sampling gate tests
# =============================================================================


class TestShouldReview:
    """Test the deterministic L1 sampling gate on Glacis client."""

    def _make_glacis(self, seed: bytes | None = None) -> Glacis:
        """Create an offline Glacis client."""
        return Glacis(mode="offline", signing_seed=seed or bytes(32))

    def _make_attestation(self, evidence_hash: str = "a" * 64) -> Attestation:
        return Attestation(
            id="att_test",
            evidence_hash=evidence_hash,
        )

    def test_sampling_rate_100_always_l1(self):
        g = self._make_glacis()
        att = self._make_attestation()
        sd = g.should_review(att, sampling_rate=1.0)
        assert sd.level == "L1"

    def test_sampling_rate_0_always_l0(self):
        g = self._make_glacis()
        att = self._make_attestation()
        sd = g.should_review(att, sampling_rate=0.0)
        assert sd.level == "L0"

    def test_deterministic(self):
        """Same seed + attestation + rate → same result."""
        seed = b"\x42" * 32
        g = self._make_glacis(seed)
        att = self._make_attestation()

        sd1 = g.should_review(att, sampling_rate=0.5)
        sd2 = g.should_review(att, sampling_rate=0.5)
        assert sd1.level == sd2.level
        assert sd1.sample_value == sd2.sample_value
        assert sd1.prf_tag == sd2.prf_tag

    def test_different_hashes_may_vary(self):
        """Different evidence hashes should produce different sample_values."""
        g = self._make_glacis(b"\x01" * 32)
        att_a = self._make_attestation("a" * 64)
        att_b = self._make_attestation("b" * 64)

        sd_a = g.should_review(att_a, sampling_rate=0.5)
        sd_b = g.should_review(att_b, sampling_rate=0.5)

        # Different evidence_hash → different HMAC → different sample_value
        assert sd_a.sample_value != sd_b.sample_value

    def test_returns_sampling_decision(self):
        g = self._make_glacis()
        att = self._make_attestation()
        sd = g.should_review(att, sampling_rate=1.0)

        assert isinstance(sd, SamplingDecision)
        assert sd.level in ("L0", "L1")
        assert isinstance(sd.sample_value, int)
        assert isinstance(sd.prf_tag, list)

    def test_prf_tag_is_32_bytes(self):
        """HMAC-SHA256 produces 32-byte tag."""
        g = self._make_glacis()
        att = self._make_attestation()
        sd = g.should_review(att)
        assert len(sd.prf_tag) == 32

    def test_different_seeds_different_decisions(self):
        """Different signing seeds should produce different PRF tags."""
        att = self._make_attestation()
        g1 = self._make_glacis(b"\x01" * 32)
        g2 = self._make_glacis(b"\x02" * 32)

        sd1 = g1.should_review(att, sampling_rate=0.5)
        sd2 = g2.should_review(att, sampling_rate=0.5)

        assert sd1.prf_tag != sd2.prf_tag

    def test_rate_100_many_hashes_all_l1(self):
        """At rate=1.0, every attestation should be L1."""
        g = self._make_glacis()
        for i in range(20):
            att = self._make_attestation(f"{i:064x}")
            sd = g.should_review(att, sampling_rate=1.0)
            assert sd.level == "L1"

    def test_rate_0_many_hashes_all_l0(self):
        """At rate=0.0, every attestation should be L0."""
        g = self._make_glacis()
        for i in range(20):
            att = self._make_attestation(f"{i:064x}")
            sd = g.should_review(att, sampling_rate=0.0)
            assert sd.level == "L0"

    def test_requires_signing_seed(self):
        """should_review should fail without signing_seed (online mode)."""
        g = Glacis(api_key="glsk_test_dummy")
        att = self._make_attestation()
        with pytest.raises(ValueError, match="signing_seed"):
            g.should_review(att)

    def test_config_l1_rate_used_as_default(self):
        """should_review uses l1_rate from sampling config when no explicit rate."""
        from glacis.config import SamplingConfig

        config = SamplingConfig(l1_rate=0.0)
        g = Glacis(mode="offline", signing_seed=bytes(32), sampling_config=config)
        att = self._make_attestation()
        sd = g.should_review(att)
        assert sd.level == "L0"

    def test_explicit_rate_overrides_config(self):
        """Explicit sampling_rate parameter overrides config l1_rate."""
        from glacis.config import SamplingConfig

        config = SamplingConfig(l1_rate=0.0)
        g = Glacis(mode="offline", signing_seed=bytes(32), sampling_config=config)
        att = self._make_attestation()
        sd = g.should_review(att, sampling_rate=1.0)
        assert sd.level == "L1"

    def test_l2_rate_produces_l2(self):
        """l2_rate=1.0 should always produce L2."""
        from glacis.config import SamplingConfig

        config = SamplingConfig(l1_rate=1.0, l2_rate=1.0)
        g = Glacis(mode="offline", signing_seed=bytes(32), sampling_config=config)
        att = self._make_attestation()
        sd = g.should_review(att)
        assert sd.level == "L2"

    def test_l2_zero_never_l2(self):
        """l2_rate=0.0 should never produce L2."""
        from glacis.config import SamplingConfig

        config = SamplingConfig(l1_rate=1.0, l2_rate=0.0)
        g = Glacis(mode="offline", signing_seed=bytes(32), sampling_config=config)
        for i in range(20):
            att = self._make_attestation(f"{i:064x}")
            sd = g.should_review(att)
            assert sd.level in ("L0", "L1")

    def test_l2_nested_implies_l1(self):
        """L2 sampling is nested — L2 ⊂ L1."""
        from glacis.config import SamplingConfig

        # With both rates at 1.0, everything is L2
        config = SamplingConfig(l1_rate=1.0, l2_rate=1.0)
        g = Glacis(mode="offline", signing_seed=bytes(32), sampling_config=config)
        att = self._make_attestation()
        sd = g.should_review(att)
        assert sd.level == "L2"


# =============================================================================
# Full pipeline tests (mocked LLM + real Glacis offline attestation)
# =============================================================================


class TestJudgePipeline:
    """Test the complete judge pipeline flow using offline Glacis."""

    def _make_glacis(self, seed: bytes = None, db_path: Path = None) -> Glacis:
        return Glacis(
            mode="offline",
            signing_seed=seed or os.urandom(32),
            db_path=db_path,
        )

    def test_full_flow(self, sample_qa_pair, sample_reference, tmp_path):
        """Generate → sample → judge → attest review attestation."""
        db_path = tmp_path / "test.db"
        seed = os.urandom(32)
        g = self._make_glacis(seed=seed, db_path=db_path)
        op = g.operation()

        # 1. Attest the QA pair
        pair_att = g.attest(
            service_id="test-service",
            operation_type="qa_generation",
            input={"chunk": sample_reference},
            output=sample_qa_pair,
            operation_id=op.operation_id,
            operation_sequence=op.next_sequence(),
        )
        assert pair_att.id.startswith("oatt_")

        # 2. Check sampling — at rate=1.0, should be L1
        sd = g.should_review(pair_att, sampling_rate=1.0)
        assert sd.level == "L1"

        # 3. Run judges (using stubs)
        runner = JudgeRunner(judges=[
            StubJudge("gpt-4o-mini", 3.0, "correct"),
            StubJudge("claude-haiku", 2.5, "mostly correct"),
        ])
        review = runner.run(sample_qa_pair, reference=sample_reference)
        assert review.final_score == 2.75
        assert review.recommendation == "uphold"

        # 4. Attest the review
        review_att = g.attest(
            service_id="test-service",
            operation_type="review",
            input={"pair_attestation_id": pair_att.id},
            output=review.model_dump(),
            operation_id=op.operation_id,
            operation_sequence=op.next_sequence(),
        )
        assert review_att.id.startswith("oatt_")
        assert review_att.operation_id == pair_att.operation_id

        g.close()

    def test_operation_id_continuity(self, tmp_path):
        """All attestations in a pipeline share the same operation_id."""
        db_path = tmp_path / "test.db"
        g = self._make_glacis(db_path=db_path)
        op = g.operation()

        att1 = g.attest(
            service_id="svc", operation_type="qa_generation",
            input={"q": 1}, output={"a": 1},
            operation_id=op.operation_id,
            operation_sequence=op.next_sequence(),
        )
        att2 = g.attest(
            service_id="svc", operation_type="review",
            input={"pair": att1.id}, output={"score": 2.5},
            operation_id=op.operation_id,
            operation_sequence=op.next_sequence(),
        )

        assert att1.operation_id == att2.operation_id
        assert att1.operation_sequence == 0
        assert att2.operation_sequence == 1

        g.close()

    def test_review_evidence_has_scores(self, sample_qa_pair, sample_reference, tmp_path):
        """Review attestation evidence should contain verdict details."""
        db_path = tmp_path / "test.db"
        g = self._make_glacis(db_path=db_path)

        runner = JudgeRunner(judges=[
            StubJudge("j1", 2.5, "good"),
            StubJudge("j2", 3.0, "great"),
        ])
        review = runner.run(sample_qa_pair, reference=sample_reference)
        output = review.model_dump()

        assert output["final_score"] == 2.75
        assert len(output["verdicts"]) == 2
        assert output["recommendation"] == "uphold"
        assert output["consensus"] is True

        g.close()

    def test_skipped_when_l0(self, sample_qa_pair, tmp_path):
        """At sampling_rate=0.0, no review should be triggered."""
        db_path = tmp_path / "test.db"
        g = self._make_glacis(db_path=db_path)

        att = g.attest(
            service_id="svc", operation_type="qa_generation",
            input={"q": 1}, output=sample_qa_pair,
        )

        sd = g.should_review(att, sampling_rate=0.0)
        assert sd.level == "L0"
        # In real code, the caller skips judge pipeline when L0

        g.close()

    def test_only_sampled_pairs_judged(self, tmp_path):
        """At a fractional rate, some pairs are L0 and some L1."""
        db_path = tmp_path / "test.db"
        seed = b"\x99" * 32
        g = self._make_glacis(seed=seed, db_path=db_path)

        l0_count = 0
        l1_count = 0
        for i in range(100):
            att = g.attest(
                service_id="svc", operation_type="qa_generation",
                input={"q": i}, output={"a": i},
            )
            sd = g.should_review(att, sampling_rate=0.5)
            if sd.level == "L0":
                l0_count += 1
            else:
                l1_count += 1

        # With 100 samples at 50% rate, both counts should be > 0
        # (extremely unlikely all land on one side)
        assert l0_count > 0, "Expected some L0 decisions at 50% rate"
        assert l1_count > 0, "Expected some L1 decisions at 50% rate"

        g.close()


# =============================================================================
# Human review (supersedes) tests
# =============================================================================


class TestHumanReview:
    """Test the supersedes flow for human-edited QA pairs."""

    def test_supersedes_creates_new_attestation(self, tmp_path):
        db_path = tmp_path / "test.db"
        g = Glacis(mode="offline", signing_seed=os.urandom(32), db_path=db_path)
        op = g.operation()

        # Original pair
        original = g.attest(
            service_id="svc", operation_type="qa_generation",
            input={"chunk": "ref"}, output={"q": "Q?", "a": "A."},
            operation_id=op.operation_id,
            operation_sequence=op.next_sequence(),
        )

        # Human edits → new attestation with supersedes
        edited = g.attest(
            service_id="svc", operation_type="qa_generation",
            input={"chunk": "ref"}, output={"q": "Q?", "a": "Better A."},
            operation_id=op.operation_id,
            operation_sequence=op.next_sequence(),
            supersedes=original.id,
        )

        assert edited.id != original.id
        assert edited.supersedes == original.id
        assert edited.operation_id == original.operation_id

        g.close()

    def test_rejudge_after_edit(self, tmp_path):
        """Edited pair should get its own review attestation."""
        db_path = tmp_path / "test.db"
        g = Glacis(mode="offline", signing_seed=os.urandom(32), db_path=db_path)
        op = g.operation()

        # Original pair + review
        original = g.attest(
            service_id="svc", operation_type="qa_generation",
            input={"chunk": "ref"}, output={"q": "Q?", "a": "Wrong A."},
            operation_id=op.operation_id,
            operation_sequence=op.next_sequence(),
        )
        review_1 = g.attest(
            service_id="svc", operation_type="review",
            input={"pair_id": original.id},
            output={"final_score": 0.5, "recommendation": "escalate"},
            operation_id=op.operation_id,
            operation_sequence=op.next_sequence(),
        )

        # Human edits pair → new attestation + new review
        edited = g.attest(
            service_id="svc", operation_type="qa_generation",
            input={"chunk": "ref"}, output={"q": "Q?", "a": "Correct A."},
            operation_id=op.operation_id,
            operation_sequence=op.next_sequence(),
            supersedes=original.id,
        )
        review_2 = g.attest(
            service_id="svc", operation_type="review",
            input={"pair_id": edited.id},
            output={"final_score": 3.0, "recommendation": "uphold"},
            operation_id=op.operation_id,
            operation_sequence=op.next_sequence(),
        )

        # All share operation_id
        ids = {original.operation_id, review_1.operation_id,
               edited.operation_id, review_2.operation_id}
        assert len(ids) == 1

        # Sequence is monotonic
        assert original.operation_sequence == 0
        assert review_1.operation_sequence == 1
        assert edited.operation_sequence == 2
        assert review_2.operation_sequence == 3

        g.close()


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_qa_pair(self):
        """Empty question/answer should not crash judges."""
        runner = JudgeRunner(judges=[StubJudge("a", 1.0)])
        result = runner.run({"question": "", "answer": ""})
        assert result.final_score == 1.0

    def test_long_reference_document(self, sample_qa_pair):
        """A very long reference document should not break prompt building."""
        from judges.qa_explain_judge import _build_prompt

        long_ref = "A" * 10_000
        prompt = _build_prompt(sample_qa_pair, reference=long_ref, rubric=None)
        assert len(prompt) > 10_000
        assert "A" * 100 in prompt

    def test_special_characters_in_qa(self):
        """QA pairs with special chars should not break JSON parsing."""
        runner = JudgeRunner(judges=[StubJudge("a", 2.0)])
        result = runner.run({
            "question": 'What does "hello" mean in 日本語?',
            "answer": "It means こんにちは (konnichiwa).",
        })
        assert result.final_score == 2.0

    def test_missing_keys_in_item(self):
        """Item without question/answer keys should still work."""
        from judges.qa_explain_judge import _build_prompt

        prompt = _build_prompt({}, reference=None, rubric=None)
        assert "Question: " in prompt
        assert "Answer: " in prompt

    def test_judge_verdict_serializes_to_dict(self):
        """JudgeVerdict.model_dump() should produce a clean dict."""
        v = JudgeVerdict(judge_id="test", score=2.5, rationale="Good", latency_ms=42)
        d = v.model_dump()
        assert d["judge_id"] == "test"
        assert d["score"] == 2.5
        assert d["latency_ms"] == 42

    def test_review_result_serializes_to_dict(self):
        """ReviewResult.model_dump() should produce a clean dict for attestation."""
        result = ReviewResult(
            verdicts=[
                JudgeVerdict(judge_id="j1", score=2.5, rationale="good"),
                JudgeVerdict(judge_id="j2", score=3.0, rationale="great"),
            ],
            final_score=2.75,
            consensus=True,
            recommendation="uphold",
        )
        d = result.model_dump()
        assert d["final_score"] == 2.75
        assert len(d["verdicts"]) == 2
        assert d["recommendation"] == "uphold"

    def test_context_manager_on_base_judge(self):
        """BaseJudge context manager should work."""
        judge = StubJudge("test", 2.0)
        with judge as j:
            v = j.evaluate({"question": "Q", "answer": "A"})
            assert v.score == 2.0


# =============================================================================
# JudgesConfig tests
# =============================================================================


class TestJudgesConfig:
    """Test JudgesConfig model validation and defaults."""

    def test_defaults_match_current_behavior(self):
        """Default config should match the previously hardcoded values."""
        cfg = JudgesConfig()
        assert cfg.max_score == 3.0
        assert cfg.consensus_threshold == 1.0
        assert cfg.uphold_threshold == 2.0
        assert cfg.borderline_threshold == 1.0
        assert cfg.score_precision == 4

    def test_custom_values(self):
        cfg = JudgesConfig(
            max_score=1.0,
            uphold_threshold=0.7,
            borderline_threshold=0.4,
            consensus_threshold=0.2,
            score_precision=2,
        )
        assert cfg.max_score == 1.0
        assert cfg.uphold_threshold == 0.7
        assert cfg.borderline_threshold == 0.4
        assert cfg.consensus_threshold == 0.2
        assert cfg.score_precision == 2

    def test_max_score_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            JudgesConfig(max_score=-1.0)

    def test_thresholds_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            JudgesConfig(uphold_threshold=-0.1)
        with pytest.raises(ValidationError):
            JudgesConfig(borderline_threshold=-0.1)
        with pytest.raises(ValidationError):
            JudgesConfig(consensus_threshold=-0.1)

    def test_score_precision_bounds(self):
        JudgesConfig(score_precision=0)  # min valid
        JudgesConfig(score_precision=10)  # max valid
        with pytest.raises(ValidationError):
            JudgesConfig(score_precision=-1)
        with pytest.raises(ValidationError):
            JudgesConfig(score_precision=11)


# =============================================================================
# Config-driven JudgeRunner tests
# =============================================================================


class TestJudgeRunnerWithConfig:
    """Test JudgeRunner behavior with explicit JudgesConfig."""

    def test_config_overrides_uphold_threshold(self, sample_qa_pair):
        """Custom uphold_threshold shifts where 'uphold' kicks in."""
        cfg = JudgesConfig(uphold_threshold=2.8)
        runner = JudgeRunner(
            judges=[StubJudge("a", 2.5), StubJudge("b", 3.0)],
            config=cfg,
        )
        result = runner.run(sample_qa_pair)
        # avg = 2.75, below uphold_threshold=2.8
        assert result.final_score == 2.75
        assert result.recommendation == "borderline"

    def test_config_overrides_borderline_threshold(self, sample_qa_pair):
        """Custom borderline_threshold shifts where 'escalate' kicks in."""
        cfg = JudgesConfig(borderline_threshold=2.0)
        runner = JudgeRunner(
            judges=[StubJudge("a", 1.5), StubJudge("b", 1.5)],
            config=cfg,
        )
        result = runner.run(sample_qa_pair)
        # avg = 1.5, below borderline_threshold=2.0
        assert result.recommendation == "escalate"

    def test_config_consensus_threshold(self, sample_qa_pair):
        """Config consensus_threshold overrides the init param."""
        cfg = JudgesConfig(consensus_threshold=0.3)
        runner = JudgeRunner(
            judges=[StubJudge("a", 2.0), StubJudge("b", 2.5)],
            config=cfg,
        )
        result = runner.run(sample_qa_pair)
        # spread = 0.5 > 0.3
        assert result.consensus is False

    def test_config_score_precision(self, sample_qa_pair):
        """Config score_precision controls rounding."""
        cfg = JudgesConfig(score_precision=2)
        runner = JudgeRunner(
            judges=[StubJudge("a", 1.0), StubJudge("b", 2.0), StubJudge("c", 2.0)],
            config=cfg,
        )
        result = runner.run(sample_qa_pair)
        # avg = 5/3 = 1.6666...
        assert result.final_score == 1.67

    def test_config_max_score_in_result(self, sample_qa_pair):
        """ReviewResult.max_score should come from config."""
        cfg = JudgesConfig(max_score=1.0)
        runner = JudgeRunner(judges=[StubJudge("a", 0.8)], config=cfg)
        result = runner.run(sample_qa_pair)
        assert result.max_score == 1.0

    def test_l2_conformity_preset(self, sample_qa_pair):
        """L2 conformity review config: 0-1 scale, thresholds adjusted."""
        cfg = JudgesConfig(
            max_score=1.0,
            uphold_threshold=0.7,
            borderline_threshold=0.4,
            consensus_threshold=0.2,
        )

        # High conformity → uphold
        runner = JudgeRunner(judges=[StubJudge("a", 0.9)], config=cfg)
        result = runner.run(sample_qa_pair)
        assert result.recommendation == "uphold"

        # Medium conformity → borderline
        runner = JudgeRunner(judges=[StubJudge("a", 0.5)], config=cfg)
        result = runner.run(sample_qa_pair)
        assert result.recommendation == "borderline"

        # Low conformity → escalate
        runner = JudgeRunner(judges=[StubJudge("a", 0.2)], config=cfg)
        result = runner.run(sample_qa_pair)
        assert result.recommendation == "escalate"

    def test_backward_compat_no_config(self, sample_qa_pair):
        """JudgeRunner without config produces identical results to before."""
        runner = JudgeRunner(judges=[StubJudge("a", 2.5), StubJudge("b", 3.0)])
        result = runner.run(sample_qa_pair)
        assert result.final_score == 2.75
        assert result.recommendation == "uphold"
        assert result.consensus is True
        assert result.max_score == 3.0

    def test_backward_compat_consensus_threshold_param(self, sample_qa_pair):
        """Legacy consensus_threshold param still works without config."""
        runner = JudgeRunner(
            judges=[StubJudge("a", 1.0), StubJudge("b", 2.5)],
            consensus_threshold=1.0,
        )
        result = runner.run(sample_qa_pair)
        # spread = 1.5 > 1.0
        assert result.consensus is False

    def test_config_takes_precedence_over_param(self, sample_qa_pair):
        """When both config and consensus_threshold param given, config wins."""
        cfg = JudgesConfig(consensus_threshold=2.0)
        runner = JudgeRunner(
            judges=[StubJudge("a", 1.0), StubJudge("b", 2.5)],
            consensus_threshold=0.5,  # would flag as False
            config=cfg,  # but config says 2.0 — spread 1.5 < 2.0 → True
        )
        result = runner.run(sample_qa_pair)
        assert result.consensus is True

    def test_config_property_exposed(self):
        """JudgeRunner.config property should return the config."""
        cfg = JudgesConfig(uphold_threshold=2.5)
        runner = JudgeRunner(judges=[], config=cfg)
        assert runner.config.uphold_threshold == 2.5


# =============================================================================
# Config YAML round-trip tests
# =============================================================================


class TestConfigYamlIntegration:
    """Test that JudgesConfig integrates with GlacisConfig/glacis.yaml."""

    def test_glacis_config_has_judges_field(self):
        from glacis.config import GlacisConfig

        cfg = GlacisConfig()
        assert hasattr(cfg, "judges")
        assert isinstance(cfg.judges, JudgesConfig)
        assert cfg.judges.max_score == 3.0

    def test_glacis_config_from_dict(self):
        from glacis.config import GlacisConfig

        cfg = GlacisConfig(
            judges={
                "max_score": 1.0,
                "uphold_threshold": 0.7,
                "borderline_threshold": 0.4,
            }
        )
        assert cfg.judges.max_score == 1.0
        assert cfg.judges.uphold_threshold == 0.7


# =============================================================================
# Conformity score rename test
# =============================================================================


class TestConformityScoreRename:
    """Test that Review model uses conformity_score (not nonconformity_score)."""

    def test_review_has_conformity_score(self):
        review = Review(
            sample_probability=0.1,
            judge_ids=["j1"],
            conformity_score=0.8,
            recommendation="uphold",
            rationale="Controls performed well",
        )
        assert review.conformity_score == 0.8

    def test_review_no_nonconformity_score(self):
        """The old nonconformity_score field should not exist."""
        assert not hasattr(Review, "nonconformity_score")
