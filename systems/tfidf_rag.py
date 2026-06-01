"""tfidf_rag — TF-IDF retrieval baseline (real-offline, NO keys, NO SDK).

TF-IDF retriever over the ingested corpus + extractive answer from the top-k
concatenation. Uses scikit-learn's `TfidfVectorizer` when available (a declared
dep), and falls back to a PURE-PYTHON TF-IDF + cosine implementation if sklearn
is missing, so this module still runs in a minimal environment.

Honesty (DESIGN.md §1, Tier A): cost_usd=0.0 (no LLM); `update()` re-indexes the
changed doc but keeps the stale fact unless deleted; `tokens_used` = token count
of retrieved context.

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

SYSTEM_NAME = "tfidf_rag"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _best_sentence(passage: str, query_terms: set[str]) -> str:
    """Sentence of `passage` with the most query-term overlap (deterministic, no LLM)."""
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


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


class TfidfRagSystem(MemorySystem):
    """TF-IDF cosine retriever; extractive top-passage answer."""

    name = SYSTEM_NAME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.top_k: int = int(self.config.get("top_k", 3))
        self._docs: dict[str, Document] = {}
        self._use_sklearn = _sklearn_available() and not self.config.get("force_pure_python")

    # --- MemorySystem interface ------------------------------------------ #

    def ingest(self, documents: Iterable[Document]) -> None:
        for doc in documents:
            self._docs[doc.id] = doc

    def update(self, doc_id: str, new_text: str) -> None:
        if doc_id in self._docs:
            self._docs[doc_id].text = new_text

    def delete(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)

    def _persona_docs(self, persona_id: str) -> list[tuple[str, str]]:
        return [(d.id, d.text) for d in self._docs.values() if d.persona_id == persona_id]

    def _rank_sklearn(self, query: str, items: list[tuple[str, str]]) -> list[tuple[str, float]]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = [t for _, t in items] + [query]
        vec = TfidfVectorizer(token_pattern=r"[A-Za-z0-9]+", lowercase=True)
        mat = vec.fit_transform(corpus)
        sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
        ranked = list(zip((i for i, _ in items), (float(s) for s in sims)))
        ranked.sort(key=lambda x: (-x[1], x[0]))
        return ranked

    def _rank_pure(self, query: str, items: list[tuple[str, str]]) -> list[tuple[str, float]]:
        # Pure-python TF-IDF (smoothed idf) + cosine similarity.
        docs_toks = {i: _tokenize(t) for i, t in items}
        n = len(items) or 1
        df: dict[str, int] = {}
        for toks in docs_toks.values():
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}

        def vec(toks: list[str]) -> dict[str, float]:
            tf = Counter(toks)
            return {t: (c / len(toks)) * idf.get(t, math.log(1 + n) + 1.0) for t, c in tf.items()} if toks else {}

        q_vec = vec(_tokenize(query))
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        ranked: list[tuple[str, float]] = []
        for doc_id, toks in docs_toks.items():
            d_vec = vec(toks)
            d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
            dot = sum(q_vec.get(t, 0.0) * v for t, v in d_vec.items())
            ranked.append((doc_id, dot / (q_norm * d_norm)))
        ranked.sort(key=lambda x: (-x[1], x[0]))
        return ranked

    def query(self, query: str, persona_id: str) -> QueryResult:
        t0 = time.perf_counter()
        items = self._persona_docs(persona_id)
        if not items:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return QueryResult("", [], latency_ms, 0, 0.0,
                               {"retriever": "tfidf_rag", "retrieved_context": []})
        ranked = self._rank_sklearn(query, items) if self._use_sklearn else self._rank_pure(query, items)
        top = [doc_id for doc_id, s in ranked[: self.top_k] if s > 0.0]
        if not top:
            top = [doc_id for doc_id, _ in ranked[: self.top_k]]
        passages = [self._docs[d].text for d in top]
        context = "\n".join(passages)
        answer = _best_sentence(self._docs[top[0]].text, set(_tokenize(query))) if top else ""
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return QueryResult(
            answer=answer,
            supporting_doc_ids=top,
            latency_ms=latency_ms,
            tokens_used=len(_tokenize(context)),
            cost_usd=0.0,
            extra={"retriever": "tfidf_rag", "backend": "sklearn" if self._use_sklearn else "pure_python",
                   "top_k": self.top_k, "retrieved_context": passages},
        )

    def stats(self) -> dict[str, Any]:
        return {
            "indexed_docs": len(self._docs),
            "storage_bytes": sum(len(d.text.encode("utf-8")) for d in self._docs.values()),
            "retriever": "tfidf_rag",
            "backend": "sklearn" if self._use_sklearn else "pure_python",
        }

    def reset(self) -> None:
        self._docs.clear()


def build_system(config: dict[str, Any] | None = None) -> MemorySystem:
    return TfidfRagSystem(config)
