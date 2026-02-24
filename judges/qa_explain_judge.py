"""
LLM Judges for QA pair evaluation.

Two judges score QA pairs against a reference document on a 0-3 scale:
- OpenAIJudge: GPT-4o-mini
- AnthropicJudge: Claude Haiku 4.5

Usage:
    from judges.qa_explain_judge import OpenAIJudge, AnthropicJudge
    from glacis.judges import JudgeRunner

    runner = JudgeRunner(judges=[
        OpenAIJudge(api_key=os.environ["OPENAI_API_KEY"]),
        AnthropicJudge(api_key=os.environ["ANTHROPIC_API_KEY"]),
    ])
    result = runner.run(qa_pair, reference=source_document)
    print(result.final_score, result.recommendation)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from glacis.judges.base import BaseJudge, JudgeVerdict

logger = logging.getLogger(__name__)


SCORING_RUBRIC = """Score this QA pair from 0-3 based ONLY on the reference document.

0 - INCORRECT: Answer contradicts or is unsupported by the source
1 - PARTIALLY CORRECT: Some factual basis but missing key details or has errors
2 - MOSTLY CORRECT: Factually grounded but minor issues (wording, completeness)
3 - FULLY CORRECT: Accurate, complete, directly supported by the source

Respond with JSON only: {"score": <0-3>, "rationale": "<brief explanation>"}"""


def _build_prompt(
    item: dict[str, Any],
    reference: Optional[str],
    rubric: Optional[str],
) -> str:
    """Build the judge prompt from QA pair + reference + rubric."""
    parts = []
    if reference:
        parts.append(f"Reference document:\n{reference}")
    parts.append(f"Question: {item.get('question', '')}")
    parts.append(f"Answer: {item.get('answer', '')}")
    parts.append(rubric or SCORING_RUBRIC)
    return "\n\n".join(parts)


def _parse_verdict(
    judge_id: str,
    raw_text: str,
    latency_ms: int,
) -> JudgeVerdict:
    """Parse JSON verdict from LLM response.

    Expected format: {"score": 0-3, "rationale": "..."}
    On parse failure, returns score=0 with error rationale.
    """
    try:
        # Strip markdown code fences if present
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()

        data = json.loads(text)
        score = float(data["score"])
        score = max(0.0, min(3.0, score))  # Clamp to valid range
        rationale = str(data.get("rationale", ""))

        return JudgeVerdict(
            judge_id=judge_id,
            score=score,
            rationale=rationale,
            latency_ms=latency_ms,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning("Failed to parse judge %s response: %s", judge_id, e)
        return JudgeVerdict(
            judge_id=judge_id,
            score=0.0,
            rationale=f"Parse error: {e}. Raw response: {raw_text[:200]}",
            latency_ms=latency_ms,
            metadata={"parse_error": str(e)},
        )


class OpenAIJudge(BaseJudge):
    """Fact-checking judge using OpenAI models.

    Args:
        api_key: OpenAI API key.
        model: Model ID. Default "gpt-4o-mini".
    """

    judge_id = "gpt-4o-mini"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.judge_id = model

    def evaluate(
        self,
        item: dict[str, Any],
        reference: Optional[str] = None,
        rubric: Optional[str] = None,
    ) -> JudgeVerdict:
        """Evaluate a QA pair using OpenAI."""
        prompt = _build_prompt(item, reference, rubric)
        start = time.perf_counter()

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            raw_text = response.choices[0].message.content or ""
            return _parse_verdict(self.judge_id, raw_text, latency_ms)
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("OpenAI judge API error: %s", e)
            return JudgeVerdict(
                judge_id=self.judge_id,
                score=0.0,
                rationale=f"API error: {e}",
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )


class AnthropicJudge(BaseJudge):
    """Fact-checking judge using Anthropic models.

    Args:
        api_key: Anthropic API key.
        model: Model ID. Default "claude-haiku-4-5-20251001".
    """

    judge_id = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self.judge_id = model

    def evaluate(
        self,
        item: dict[str, Any],
        reference: Optional[str] = None,
        rubric: Optional[str] = None,
    ) -> JudgeVerdict:
        """Evaluate a QA pair using Anthropic."""
        prompt = _build_prompt(item, reference, rubric)
        start = time.perf_counter()

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            raw_text = response.content[0].text if response.content else ""
            return _parse_verdict(self.judge_id, raw_text, latency_ms)
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("Anthropic judge API error: %s", e)
            return JudgeVerdict(
                judge_id=self.judge_id,
                score=0.0,
                rationale=f"API error: {e}",
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )
