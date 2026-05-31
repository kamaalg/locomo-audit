"""longcontext — brute-force long-context control (wired-needs-keys; import-guarded).

No memory system at all: this control concatenates the WHOLE persona corpus into
a single Anthropic (Claude) prompt and asks the question directly. It is the
honest "how far does brute force get you, and at what token cost" baseline
(DESIGN.md §1 Tier B, system 6). It burns the most tokens of any system and
needs only `ANTHROPIC_API_KEY` — the one key the PI has.

Honesty contract (DESIGN.md §1 Tier B):
- DISCOVERABLE with no SDK and no key, so the leaderboard can show it as
  "wired, not yet run". The `anthropic` SDK is imported LAZILY inside
  `build_system()`; if it is not installed, or `ANTHROPIC_API_KEY` is unset,
  `build_system()` raises `SystemUnavailable` with a clear, actionable message.
  The runner catches that, logs a SKIP, and continues — it NEVER crashes the
  run, and the web app never depends on this SDK/key being present.
- It has no incremental memory: `update`/`delete` mutate the in-memory corpus
  copy that gets re-stuffed every query (there is no index to update), and we
  report that honestly rather than faking a cheap update path.
- `cost_usd` and `tokens_used` are computed from the REAL Anthropic usage of the
  call (input+output tokens) so the long-context control's headline cost is
  measured, not estimated. Until a real keyed run happens, this adapter produces
  NO number that is ever written as independent.

mvpStatus: wired-needs-keys. needsKeys: true (ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterable

from systems._registry import SystemUnavailable
from systems.base import Document, MemorySystem, QueryResult

SYSTEM_NAME = "longcontext"

# Vanilla defaults. Sonnet-class model + small output (answers are short).
_DEFAULT_MODEL = "claude-sonnet-4-5"
_DEFAULT_MAX_TOKENS = 512

# Per-Mtok pricing (USD) used only to convert REAL usage tokens -> cost_usd.
# Sonnet-class list price; overridable via config for other models.
_DEFAULT_PRICE_IN_PER_MTOK = 3.0
_DEFAULT_PRICE_OUT_PER_MTOK = 15.0


class LongContextSystem(MemorySystem):
    """Stuff the whole persona corpus into one Claude prompt; answer directly.

    This is a control, not a memory system: it keeps the corpus in a plain list
    and re-sends the relevant persona's documents on every query.
    """

    name = SYSTEM_NAME

    def __init__(self, config: dict[str, Any], client: Any) -> None:
        self.config = config
        self._client = client
        self._model: str = str(config.get("model", _DEFAULT_MODEL))
        self._max_tokens: int = int(config.get("max_tokens", _DEFAULT_MAX_TOKENS))
        self._price_in: float = float(
            config.get("price_in_per_mtok", _DEFAULT_PRICE_IN_PER_MTOK)
        )
        self._price_out: float = float(
            config.get("price_out_per_mtok", _DEFAULT_PRICE_OUT_PER_MTOK)
        )
        self._docs: dict[str, Document] = {}

    # --- MemorySystem interface ------------------------------------------ #

    def ingest(self, documents: Iterable[Document]) -> None:
        for doc in documents:
            self._docs[doc.id] = doc

    def update(self, doc_id: str, new_text: str) -> None:
        # No index to update: the full corpus is re-stuffed each query, so an
        # update is just an in-place text swap of the kept copy. Honest: there
        # is no cheap incremental path here — every query pays full context cost.
        if doc_id in self._docs:
            self._docs[doc_id].text = new_text

    def delete(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)

    def _persona_docs(self, persona_id: str) -> list[Document]:
        docs = [d for d in self._docs.values() if d.persona_id == persona_id]
        # Deterministic order: timestamp then id (no wall-clock / random).
        docs.sort(key=lambda d: (d.timestamp, d.id))
        return docs

    def query(self, query: str, persona_id: str) -> QueryResult:  # pragma: no cover - needs key + SDK
        t0 = time.perf_counter()
        docs = self._persona_docs(persona_id)
        # Concatenate the WHOLE persona corpus into one context block.
        context = "\n\n".join(
            f"[doc {d.id} | {d.type} | {d.timestamp}]\n{d.text}" for d in docs
        )
        prompt = (
            "You are answering a question using ONLY the personal memory documents "
            "below. Answer concisely with just the fact requested.\n\n"
            f"=== MEMORY ({len(docs)} documents) ===\n{context}\n\n"
            f"=== QUESTION ===\n{query}\n\n=== ANSWER ==="
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = "".join(
            getattr(block, "text", "") for block in getattr(resp, "content", [])
        ).strip()

        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        cost_usd = (
            in_tok / 1_000_000.0 * self._price_in
            + out_tok / 1_000_000.0 * self._price_out
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return QueryResult(
            answer=answer,
            supporting_doc_ids=[d.id for d in docs],
            latency_ms=latency_ms,
            tokens_used=in_tok + out_tok,
            cost_usd=cost_usd,
            extra={
                "control": "longcontext",
                "model": self._model,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "n_docs_in_context": len(docs),
                # retrieved_context: longcontext "retrieves" the WHOLE persona
                # corpus — surfaced as the single context block it stuffed, so the
                # audit's generate pipeline can re-answer from it uniformly
                # (protocol §3, §4.1). Note: longcontext ALREADY generated `answer`
                # via Claude, so the runner reuses that directly; this is here for
                # completeness/back-compat with the unified retrieved_context API.
                "retrieved_context": [context] if context else [],
            },
        )

    def stats(self) -> dict[str, Any]:
        return {
            "indexed_docs": len(self._docs),
            "storage_bytes": sum(
                len(d.text.encode("utf-8")) for d in self._docs.values()
            ),
            "control": "longcontext",
            "model": self._model,
        }

    def reset(self) -> None:
        self._docs.clear()


def build_system(config: dict[str, Any] | None = None) -> MemorySystem:
    """Build the long-context control; requires the Anthropic SDK + key.

    With no key/SDK this raises SystemUnavailable (NEVER crashes discovery, the
    runner, or the web app). The runner catches it and logs a clean SKIP.
    """
    config = config or {}
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemUnavailable(
            "set ANTHROPIC_API_KEY to run the longcontext control"
        )
    try:
        import anthropic  # type: ignore
    except Exception as e:  # SDK not installed
        raise SystemUnavailable(
            "pip install anthropic to run the longcontext control"
        ) from e
    client = anthropic.Anthropic(api_key=api_key)  # pragma: no cover - needs key + SDK
    return LongContextSystem(config, client)  # pragma: no cover
