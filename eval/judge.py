"""judge — pinned Anthropic LLM-as-judge for the audit protocol.

AUDIT-PROTOCOL.md §3 / DISPUTE-FORENSICS Fork C: LoCoMo's headline "J" is
LLM-as-judge accuracy, so the JUDGE MODEL IS PART OF THE MEASUREMENT. The famous
spread (58–75 vs 92.5) is largely a judge/answer-model-generation effect, not a
pure architecture win. The audit therefore PINS ONE judge model for every cell
(default `claude-sonnet-4-5`) and reports a second judge for robustness.

This module implements the binary-correctness judge in the LoCoMo / Memora style:
given (question, predicted, gold), decide whether the prediction is CORRECT —
tolerant of paraphrase, formatting, extra words, and equivalent dates/units, but
strict about the actual fact. It returns {correct: bool, reason: str} plus the
REAL `cost_usd` of the judge call.

Offline-safe import (same contract as answer_gen): no SDK/key touched at import;
`judge_answer` builds the client lazily and raises `SystemUnavailable` if absent.
Deterministic: temperature=0.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Reuse the answer_gen client builder + pricing so there is ONE Anthropic client
# and ONE price sheet across answer generation and judging.
from eval.answer_gen import (
    PRICE_IN_PER_MTOK,
    PRICE_OUT_PER_MTOK,
    _get_client,
)

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 200  # verdict JSON + a short reason


@dataclass
class JudgeResult:
    """Output of a pinned LLM-judge call."""

    correct: bool
    reason: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    extra: dict[str, Any] = field(default_factory=dict)


_JUDGE_INSTRUCTIONS = (
    "You are a strict but fair grader for a question-answering benchmark (LoCoMo "
    "style). You are given a QUESTION, the GOLD (reference) answer, and a model's "
    "PREDICTED answer. Decide whether the PREDICTED answer is CORRECT.\n\n"
    "Rules:\n"
    "- CORRECT if the prediction conveys the same factual answer as the gold, even "
    "if it is paraphrased, reordered, more verbose, or differently formatted.\n"
    "- Equivalent dates, numbers, and units count as correct (e.g. 'Sept 2023' == "
    "'2023-09'; 'two' == '2').\n"
    "- A prediction that contains the correct answer plus extra correct context is "
    "still correct.\n"
    "- INCORRECT if it states a different fact, is missing the asked-for fact, "
    "contradicts the gold, or is an 'I don't know' / refusal.\n\n"
    "Respond with ONLY a JSON object on a single line: "
    '{"correct": true|false, "reason": "<one short sentence>"}'
)


def _build_prompt(question: str, predicted: str, gold: str) -> str:
    return (
        f"{_JUDGE_INSTRUCTIONS}\n\n"
        f"QUESTION: {question}\n"
        f"GOLD ANSWER: {gold}\n"
        f"PREDICTED ANSWER: {predicted}\n\n"
        "JSON verdict:"
    )


def _parse_verdict(text: str) -> tuple[bool, str]:
    """Extract {correct, reason} from the judge's reply, robust to stray prose."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            correct = bool(obj.get("correct"))
            reason = str(obj.get("reason", "")).strip()
            return correct, reason or "(no reason given)"
        except (ValueError, TypeError):
            pass
    # Fallback: sniff a yes/no if the model didn't emit clean JSON.
    low = text.strip().lower()
    if low.startswith("true") or low.startswith("yes") or '"correct": true' in low:
        return True, "parsed from non-JSON reply"
    return False, f"unparseable judge reply: {text[:120]!r}"


def judge_answer(
    question: str,
    predicted: str,
    gold: str,
    model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: str | None = None,
    client: Any | None = None,
    price_in_per_mtok: float = PRICE_IN_PER_MTOK,
    price_out_per_mtok: float = PRICE_OUT_PER_MTOK,
) -> JudgeResult:
    """Judge predicted-vs-gold as correct/incorrect via the pinned Claude judge.

    Real Anthropic usage -> real `cost_usd`. `client` is injectable for tests.
    An empty prediction short-circuits to incorrect WITHOUT a paid call (an empty
    answer is never correct, and we should not spend to confirm that).
    """
    if not (predicted or "").strip():
        return JudgeResult(
            correct=False,
            reason="empty prediction",
            model=model,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            extra={"short_circuit": "empty_prediction"},
        )
    cli = client or _get_client(api_key)
    prompt = _build_prompt(question, predicted, gold)
    resp = cli.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        getattr(block, "text", "") for block in getattr(resp, "content", [])
    ).strip()
    correct, reason = _parse_verdict(text)
    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    cost_usd = (
        in_tok / 1_000_000.0 * price_in_per_mtok
        + out_tok / 1_000_000.0 * price_out_per_mtok
    )
    return JudgeResult(
        correct=correct,
        reason=reason,
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost_usd,
    )
