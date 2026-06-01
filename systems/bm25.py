"""bm25 — pure-Python BM25 retriever baseline (real-offline, NO keys, NO SDK).

A transparent, dependency-light memory system we wrote and can audit line-by-line.
It ingests documents, indexes them with a hand-rolled BM25 (Okapi) scorer, and at
query time returns the top passage as an EXTRACTIVE answer (no LLM, cost_usd=0.0).

Honesty (DESIGN.md §1, Tier A):
- `update()` re-indexes the changed doc in place — it CAN update incrementally,
  and it WILL keep returning the stale fact unless the obsolete doc is deleted.
  This is exactly the "stale-fact tax" the forgetting tasks expose; we do not
  hide it.
- `latency_ms` is wall-clock of the query; `tokens_used` is the whitespace token
  count of the retrieved context (there is no LLM to bill); `cost_usd` is 0.0.

mvpStatus: real-offline. needsKeys: false.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Iterable
from typing import Any

from systems.base import Document, MemorySystem, QueryResult

SYSTEM_NAME = "bm25"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _best_sentence(passage: str, query_terms: set[str]) -> str:
    """Return the sentence in `passage` with the most query-term overlap.

    Deterministic, no LLM: splits on sentence punctuation and scores each sentence
    by how many distinct query terms it contains (tie-break: shorter sentence).
    Falls back to the whole passage if it has no sentence breaks.
    """
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if s.strip()]
    if len(sents) <= 1:
        return passage.strip()
    best, best_key = sents[0], (-1, 10**9)
    for s in sents:
        toks = set(_TOKEN_RE.findall(s.lower()))
        overlap = len(toks & query_terms)
        key = (overlap, -len(toks))
        if key > best_key:
            best, best_key = s, key
    return best


class BM25System(MemorySystem):
    """Okapi BM25 over the ingested corpus; extractive top-passage answer."""

    name = SYSTEM_NAME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.k1: float = float(self.config.get("k1", 1.5))
        self.b: float = float(self.config.get("b", 0.75))
        self.top_k: int = int(self.config.get("top_k", 3))
        self._docs: dict[str, Document] = {}
        self._toks: dict[str, list[str]] = {}

    # --- index helpers ---------------------------------------------------- #

    def _index_doc(self, doc: Document) -> None:
        self._docs[doc.id] = doc
        self._toks[doc.id] = _tokenize(doc.text)

    def _corpus_stats(self) -> tuple[dict[str, int], float, int]:
        """Return (document-frequency map, avg doc length, N)."""
        df: dict[str, int] = {}
        total_len = 0
        for toks in self._toks.values():
            total_len += len(toks)
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        n = len(self._toks)
        avgdl = (total_len / n) if n else 0.0
        return df, avgdl, n

    def _score(self, query_terms: list[str], persona_id: str) -> list[tuple[str, float]]:
        df, avgdl, n = self._corpus_stats()
        scored: list[tuple[str, float]] = []
        for doc_id, toks in self._toks.items():
            if self._docs[doc_id].persona_id != persona_id:
                continue
            if not toks:
                scored.append((doc_id, 0.0))
                continue
            tf = Counter(toks)
            dl = len(toks)
            score = 0.0
            for term in query_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                n_q = df.get(term, 0)
                idf = math.log(1 + (n - n_q + 0.5) / (n_q + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * dl / (avgdl or 1))
                score += idf * (f * (self.k1 + 1)) / denom
            scored.append((doc_id, score))
        scored.sort(key=lambda x: (-x[1], x[0]))  # deterministic tie-break by id
        return scored

    # --- MemorySystem interface ------------------------------------------ #

    def ingest(self, documents: Iterable[Document]) -> None:
        for doc in documents:
            self._index_doc(doc)

    def update(self, doc_id: str, new_text: str) -> None:
        if doc_id in self._docs:
            self._docs[doc_id].text = new_text
            self._toks[doc_id] = _tokenize(new_text)

    def delete(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)
        self._toks.pop(doc_id, None)

    def query(self, query: str, persona_id: str) -> QueryResult:
        t0 = time.perf_counter()
        q_terms = _tokenize(query)
        ranked = self._score(q_terms, persona_id)
        top = [doc_id for doc_id, s in ranked[: self.top_k] if s > 0.0]
        if not top:
            top = [doc_id for doc_id, _ in ranked[: self.top_k]]
        passages = [self._docs[d].text for d in top]
        context = "\n".join(passages)
        # Extractive answer = the sentence of the top passage that best overlaps
        # the query terms (a span, not the whole passage). No LLM, deterministic.
        answer = _best_sentence(self._docs[top[0]].text, set(q_terms)) if top else ""
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return QueryResult(
            answer=answer,
            supporting_doc_ids=top,
            latency_ms=latency_ms,
            tokens_used=len(_tokenize(context)),
            cost_usd=0.0,
            # retrieved_context: the passages this retriever used, surfaced so the
            # audit's pinned answer model can generate from them (protocol §3).
            extra={"retriever": "bm25", "k1": self.k1, "b": self.b, "top_k": self.top_k,
                   "retrieved_context": passages},
        )

    def stats(self) -> dict[str, Any]:
        return {
            "indexed_docs": len(self._docs),
            "storage_bytes": sum(len(d.text.encode("utf-8")) for d in self._docs.values()),
            "retriever": "bm25",
        }

    def reset(self) -> None:
        self._docs.clear()
        self._toks.clear()


def build_system(config: dict[str, Any] | None = None) -> MemorySystem:
    return BM25System(config)
