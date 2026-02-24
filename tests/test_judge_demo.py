"""
Live API integration test for the LLM Judge Pipeline.

Requires OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables.
Skipped automatically when keys are not set.

Run with:
    OPENAI_API_KEY=sk-xxx ANTHROPIC_API_KEY=sk-ant-xxx \
        python -m pytest tests/test_judge_demo.py -v -s
"""

import os
import tempfile
from pathlib import Path

import pytest

from glacis import Glacis
from glacis.judges import JudgeRunner

# Skip the entire module if API keys are not available
pytestmark = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")),
    reason="OPENAI_API_KEY and ANTHROPIC_API_KEY required for live judge tests",
)


SAMPLE_REFERENCE = (
    "France is a country in Western Europe. Its capital city is Paris, "
    "which is also the largest city in France. Paris is known for the "
    "Eiffel Tower, the Louvre Museum, and the Seine River. France has "
    "a population of approximately 67 million people."
)

SAMPLE_QA_PAIRS = [
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
        "expected_min_score": 2.0,  # Should score high — factually correct
    },
    {
        "question": "What is France known for?",
        "answer": "France is known for the Eiffel Tower, the Louvre Museum, and the Seine River.",
        "expected_min_score": 2.0,  # Should score high — directly from source
    },
    {
        "question": "What is the population of France?",
        "answer": "France has a population of 100 million people.",
        "expected_min_score": 0.0,  # Incorrect — should score low
        "expected_max_score": 2.0,
    },
]


class TestJudgePipelineLive:
    """End-to-end judge pipeline test with real LLM API calls."""

    def test_judge_pipeline_live(self):
        """Full flow: attest QA → sample → judge → attest reviews."""
        from judges.qa_explain_judge import AnthropicJudge, OpenAIJudge

        # 1. Setup Glacis offline + judges
        seed = os.urandom(32)
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            glacis = Glacis(mode="offline", signing_seed=seed, db_path=db_path)
            op = glacis.operation()

            runner = JudgeRunner(judges=[
                OpenAIJudge(api_key=os.environ["OPENAI_API_KEY"]),
                AnthropicJudge(api_key=os.environ["ANTHROPIC_API_KEY"]),
            ])

            reviews = []

            for qa in SAMPLE_QA_PAIRS:
                pair = {"question": qa["question"], "answer": qa["answer"]}

                # 2. Attest the QA pair
                pair_att = glacis.attest(
                    service_id="judge-test",
                    operation_type="qa_generation",
                    input={"chunk": SAMPLE_REFERENCE},
                    output=pair,
                    operation_id=op.operation_id,
                    operation_sequence=op.next_sequence(),
                )

                # 3. L1 sampling gate (100% → always review)
                sd = glacis.should_review(pair_att, sampling_rate=1.0)
                assert sd.level == "L1"

                # 4. Run judges
                review = runner.run(pair, reference=SAMPLE_REFERENCE)
                reviews.append((qa, review))

                print(f"\nQ: {qa['question']}")
                print(f"A: {qa['answer']}")
                print(f"Score: {review.final_score:.2f} | "
                      f"Consensus: {review.consensus} | "
                      f"Rec: {review.recommendation}")
                for v in review.verdicts:
                    print(f"  {v.judge_id}: {v.score} ({v.latency_ms}ms) — {v.rationale[:80]}")

                # 5. Attest the review
                review_att = glacis.attest(
                    service_id="judge-test",
                    operation_type="review",
                    input={"pair_attestation_id": pair_att.id},
                    output=review.model_dump(),
                    operation_id=op.operation_id,
                    operation_sequence=op.next_sequence(),
                )

                # 6. Verify operation_id continuity
                assert review_att.operation_id == pair_att.operation_id

            # 7. Verify score expectations
            for qa, review in reviews:
                min_score = qa.get("expected_min_score", 0.0)
                max_score = qa.get("expected_max_score", 3.0)
                assert review.final_score >= min_score, (
                    f"Expected >= {min_score} for: {qa['question']}, "
                    f"got {review.final_score}"
                )
                assert review.final_score <= max_score, (
                    f"Expected <= {max_score} for: {qa['question']}, "
                    f"got {review.final_score}"
                )

            runner.close()
            glacis.close()

    def test_single_judge_openai(self):
        """Verify OpenAI judge works in isolation."""
        from judges.qa_explain_judge import OpenAIJudge

        judge = OpenAIJudge(api_key=os.environ["OPENAI_API_KEY"])
        verdict = judge.evaluate(
            {"question": "What is 2+2?", "answer": "4"},
            reference="Basic arithmetic: 2+2 equals 4.",
        )

        assert verdict.judge_id == "gpt-4o-mini"
        assert 0.0 <= verdict.score <= 3.0
        assert len(verdict.rationale) > 0
        assert verdict.latency_ms > 0

        print(f"\nOpenAI: score={verdict.score}, latency={verdict.latency_ms}ms")
        print(f"  Rationale: {verdict.rationale}")

    def test_single_judge_anthropic(self):
        """Verify Anthropic judge works in isolation."""
        from judges.qa_explain_judge import AnthropicJudge

        judge = AnthropicJudge(api_key=os.environ["ANTHROPIC_API_KEY"])
        verdict = judge.evaluate(
            {"question": "What is 2+2?", "answer": "4"},
            reference="Basic arithmetic: 2+2 equals 4.",
        )

        assert verdict.judge_id == "claude-haiku-4-5-20251001"
        assert 0.0 <= verdict.score <= 3.0
        assert len(verdict.rationale) > 0
        assert verdict.latency_ms > 0

        print(f"\nAnthropic: score={verdict.score}, latency={verdict.latency_ms}ms")
        print(f"  Rationale: {verdict.rationale}")
