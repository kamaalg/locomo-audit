"""answer_gen — pinned Anthropic answer model for the audit protocol.

AUDIT-PROTOCOL.md §3 requires that, instead of scoring a system's raw extractive
top-passage, every system retrieves context and a SINGLE PINNED answer model
GENERATES the answer from (question + retrieved_context). This removes the
"stronger answer model" confound (Fork C) and stops underrating verbose-but-
correct systems (a correct long-context answer scored 0.13 under token_f1 in a
live pilot — the gap this module closes).

Contract:
  * `generate_answer(question, retrieved_context, model, ...)` calls the pinned
    answer model (default `claude-sonnet-4-5`, ADR 0006 / protocol §3) with a
    SMALL max_tokens (answers are short) and returns a GenResult carrying the
    text, real token usage, and a REAL `cost_usd` computed from list pricing.
  * The answer is grounded ONLY in `retrieved_context` — the prompt forbids using
    outside knowledge. If the context is empty, the model is told to say it cannot
    answer (so an empty-retrieval system honestly scores wrong, not hallucinates).
  * Offline-safe IMPORT: importing this module never touches the SDK or a key.
    `generate_answer` builds the client lazily; with no key/SDK it raises
    `SystemUnavailable` (the runner already knows how to log that as a SKIP).
  * Deterministic: temperature=0 so repeated runs on the same (q, context) match.

Honesty: `cost_usd` is derived from the call's REAL input/output token usage at
Sonnet-class list price ($3.00 / Mtok in, $15.00 / Mtok out — the same constants
the longcontext control bills with). No estimate is ever returned as if real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from systems._registry import SystemUnavailable

# Pinned defaults (protocol §3). The one key the PI has -> Claude Sonnet 4.5.
DEFAULT_ANSWER_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 128  # answers are a short fact/phrase

# Sonnet-class list price (USD per 1M tokens); identical to systems/longcontext.py.
PRICE_IN_PER_MTOK = 3.0
PRICE_OUT_PER_MTOK = 15.0


@dataclass
class GenResult:
    """Output of a pinned answer-model generation call."""

    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    extra: dict[str, Any] = field(default_factory=dict)


# Cache one Anthropic client per process so we don't re-instantiate per cell.
_CLIENT: Any = None


def _get_client(api_key: str | None = None) -> Any:
    """Lazily build (and cache) an Anthropic client. Offline-safe to import.

    Raises SystemUnavailable (the runner's known SKIP signal) if the key or SDK
    is missing — never a bare ImportError that would crash a run.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    import os

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemUnavailable(
            "answer_gen needs ANTHROPIC_API_KEY (the pinned answer model is Claude)"
        )
    try:
        import anthropic  # type: ignore
    except Exception as e:  # SDK not installed
        raise SystemUnavailable("answer_gen needs `pip install anthropic`") from e
    _CLIENT = anthropic.Anthropic(api_key=key)
    return _CLIENT


def _build_prompt(question: str, retrieved_context: list[str]) -> str:
    if retrieved_context:
        ctx_block = "\n\n".join(
            f"[{i + 1}] {passage}" for i, passage in enumerate(retrieved_context)
        )
        context_section = f"=== RETRIEVED CONTEXT ===\n{ctx_block}\n\n"
        grounding = (
            "Answer the question using ONLY the retrieved context above. Do not use "
            "outside knowledge. If the context does not contain the answer, reply "
            "exactly: I don't know. Answer concisely — just the fact, name, date, or "
            "short phrase requested, with no preamble."
        )
    else:
        context_section = "=== RETRIEVED CONTEXT ===\n(no context was retrieved)\n\n"
        grounding = (
            "No context was retrieved for this question. Reply exactly: I don't know."
        )
    return (
        f"{context_section}=== QUESTION ===\n{question}\n\n"
        f"=== INSTRUCTIONS ===\n{grounding}\n\n=== ANSWER ==="
    )


def generate_answer(
    question: str,
    retrieved_context: list[str],
    model: str = DEFAULT_ANSWER_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: str | None = None,
    client: Any | None = None,
    price_in_per_mtok: float = PRICE_IN_PER_MTOK,
    price_out_per_mtok: float = PRICE_OUT_PER_MTOK,
) -> GenResult:
    """Generate a concise answer from (question + retrieved_context) via Claude.

    Real Anthropic usage -> real `cost_usd` returned (from list pricing). The
    `client` arg lets tests inject a fake; production passes nothing and a real
    client is built lazily.
    """
    cli = client or _get_client(api_key)
    prompt = _build_prompt(question, retrieved_context)
    resp = cli.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = "".join(
        getattr(block, "text", "") for block in getattr(resp, "content", [])
    ).strip()
    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    cost_usd = (
        in_tok / 1_000_000.0 * price_in_per_mtok
        + out_tok / 1_000_000.0 * price_out_per_mtok
    )
    return GenResult(
        answer=answer,
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost_usd,
        extra={"n_context_passages": len(retrieved_context)},
    )
