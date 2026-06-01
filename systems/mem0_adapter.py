"""mem0 — Mem0 SDK adapter (wired-needs-keys; import-guarded, offline-safe).

This adapter is DISCOVERABLE with no SDK and no key (so the leaderboard can show
it as "wired, not yet run"), but it is NEVER runnable without them. The Mem0 SDK
is imported LAZILY inside `build_system()`; if `mem0ai` is not installed or
`MEM0_API_KEY` is unset, `build_system()` raises `SystemUnavailable` with a clear
message. The runner catches that, logs a SKIP, and continues. No number this
adapter could produce is ever written as independent until a real keyed run.

Operation mapping (DESIGN.md §1 Tier B — Mem0 is the "gorilla" vendor):
  * `ingest`  -> client.add(messages, user_id=persona, metadata={doc_id,...})
  * `update`  -> client.update(memory_id, data=new_text)   (resolve doc_id -> memory_id)
  * `delete`  -> client.delete(memory_id)                   (resolve doc_id -> memory_id)
  * `query`   -> client.search(query, user_id=persona)      -> top memories as context

The Mem0 hosted platform stores its own `id` per memory; we keep a local
{doc_id -> mem0_memory_id} map (populated from the add() response) so that
update/delete can address the right server-side memory. Persona isolation uses
Mem0's `user_id` partition (our `persona_id`). All Mem0 calls happen ONLY after a
real key+SDK are present, so importing this module offline is safe.

Honesty:
  * Mem0 runs an LLM-backed extraction/summarisation pipeline server-side; cost is
    NOT locally observable from the SDK return, so `cost_usd` is reported as 0.0
    and the real spend is tracked by the PI's Mem0 dashboard / the runner budget.
    We flag this explicitly in `extra["cost_observed"]=False` so no one mistakes a
    0.0 here for "free".
  * `latency_ms` is the real wall-clock of the search call.
  * `tokens_used` is the whitespace token count of the retrieved context (a proxy;
    Mem0 does not return server-side token accounting via the search response).

mvpStatus: wired-needs-keys. needsKeys: true.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from typing import Any

from systems._registry import SystemUnavailable
from systems.base import Document, MemorySystem, QueryResult

SYSTEM_NAME = "mem0"


def _tokenize(text: str) -> list[str]:
    return text.split()


class Mem0System(MemorySystem):
    """Adapter over the Mem0 hosted client (`mem0.MemoryClient`).

    Maps the uniform MemorySystem interface onto Mem0's add/update/delete/search.
    Only instantiated by `build_system()` after the SDK + key are confirmed, so
    every method here may assume `self._client` is a live Mem0 client.
    """

    name = SYSTEM_NAME

    def __init__(self, config: dict[str, Any], client: Any) -> None:
        self.config = config or {}
        self._client = client
        self.top_k: int = int(self.config.get("top_k", 5))
        # Local map from our stable doc_id -> Mem0 server-side memory id, so that
        # update()/delete() can address the right memory. Mem0 may split one
        # ingested document into several memories; we keep the list.
        self._doc_to_mem: dict[str, list[str]] = {}
        # Remember which doc belongs to which persona so reset() can scope cleanly.
        self._doc_persona: dict[str, str] = {}

    # --- helpers --------------------------------------------------------- #

    def _extract_ids(self, add_result: Any) -> list[str]:
        """Pull memory ids out of a Mem0 add() response (shape varies by SDK ver).

        Mem0's add() has returned, across versions: a list of dicts with "id",
        or {"results": [{"id": ...}, ...]}. We handle both and ignore anything
        without an id rather than guessing.
        """
        if add_result is None:
            return []
        items = add_result
        if isinstance(add_result, dict):
            items = add_result.get("results", add_result.get("memories", []))
        ids: list[str] = []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get("id"):
                    ids.append(str(it["id"]))
        return ids

    # --- MemorySystem interface ------------------------------------------ #

    def ingest(self, documents: Iterable[Document]) -> None:
        n = 0
        for doc in documents:
            # Mem0's add() takes a messages list (chat-shaped) or a string; the
            # hosted API expects messages. We send the doc text as a single user
            # turn and tag it with our doc_id in metadata for traceability.
            messages = [{"role": "user", "content": doc.text}]
            metadata = {"doc_id": doc.id, "type": doc.type, "timestamp": doc.timestamp}
            result = self._client.add(
                messages,
                user_id=doc.persona_id,
                metadata=metadata,
            )
            self._doc_to_mem[doc.id] = self._extract_ids(result)
            self._doc_persona[doc.id] = doc.persona_id
            n += 1
        # Mem0's hosted add() extracts memories ASYNCHRONOUSLY (an LLM distills
        # facts from each message in the background); a search issued immediately
        # finds nothing. Wait (bounded) for extraction to settle before querying.
        if n:
            import os
            per = float(os.environ.get("MEM0_INGEST_WAIT_S", "1.5"))
            cap = float(os.environ.get("MEM0_INGEST_WAIT_CAP_S", "90"))
            time.sleep(min(n * per, cap))

    def update(self, doc_id: str, new_text: str) -> None:
        # Resolve our doc_id to the Mem0 memory id(s) and update each in place.
        # If we never recorded an id (e.g. add() returned no id), there is nothing
        # to address server-side; we do not silently fabricate one.
        for mem_id in self._doc_to_mem.get(doc_id, []):
            self._client.update(memory_id=mem_id, data=new_text)

    def delete(self, doc_id: str) -> None:
        for mem_id in self._doc_to_mem.get(doc_id, []):
            self._client.delete(memory_id=mem_id)
        self._doc_to_mem.pop(doc_id, None)
        self._doc_persona.pop(doc_id, None)

    def query(self, query: str, persona_id: str) -> QueryResult:
        t0 = time.perf_counter()
        # mem0 SDK >=2.0 requires user_id via filters= (a v2 search call), not a
        # top-level kwarg; older self-hosted clients keep the legacy signature.
        try:
            result = self._client.search(
                query, version="v2", filters={"user_id": persona_id}, limit=self.top_k
            )
        except TypeError:
            result = self._client.search(query, user_id=persona_id, limit=self.top_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # search() returns either a list of memory dicts or {"results": [...]}.
        memories = result
        if isinstance(result, dict):
            memories = result.get("results", result.get("memories", []))
        if not isinstance(memories, list):
            memories = []

        # Build context + supporting doc ids from the returned memories. Each
        # memory may carry our original doc_id in its metadata; fall back to the
        # Mem0 memory id when metadata is absent.
        passages: list[str] = []
        supporting: list[str] = []
        for m in memories:
            if not isinstance(m, dict):
                continue
            text = m.get("memory", m.get("text", "")) or ""
            if text:
                passages.append(text)
            meta = m.get("metadata") or {}
            doc_id = meta.get("doc_id") if isinstance(meta, dict) else None
            supporting.append(str(doc_id or m.get("id", "")))

        context = "\n".join(passages)
        # Mem0's search is a retrieval call; the top memory is the extractive
        # answer span. We do NOT call an LLM here, so this answer is the surfaced
        # memory, mirroring how the offline baselines answer extractively.
        answer = passages[0] if passages else ""
        return QueryResult(
            answer=answer,
            supporting_doc_ids=[s for s in supporting if s],
            latency_ms=latency_ms,
            tokens_used=len(_tokenize(context)),
            cost_usd=0.0,  # NOT locally observable; see module docstring + extra below.
            extra={
                "retriever": "mem0",
                "cost_observed": False,
                "n_memories": len(passages),
                "top_k": self.top_k,
                # retrieved_context: the memories Mem0 surfaced, for the pinned
                # answer model to generate from (protocol §3).
                "retrieved_context": passages,
            },
        )

    def stats(self) -> dict[str, Any]:
        return {
            "indexed_docs": len(self._doc_to_mem),
            "tracked_memories": sum(len(v) for v in self._doc_to_mem.values()),
            "retriever": "mem0",
            "wired": True,
        }

    def reset(self) -> None:
        # Delete every memory we created on the server for the personas we touched,
        # then clear local maps. We only delete ids we ourselves recorded — we never
        # wipe a persona's whole Mem0 store blindly.
        for doc_id, mem_ids in list(self._doc_to_mem.items()):
            for mem_id in mem_ids:
                try:
                    self._client.delete(memory_id=mem_id)
                except Exception:  # best-effort cleanup between persona evals
                    pass
        self._doc_to_mem.clear()
        self._doc_persona.clear()


def build_system(config: dict[str, Any] | None = None) -> MemorySystem:
    """Construct a live Mem0 adapter, or raise SystemUnavailable (offline-safe).

    Order of checks is key-first so the common offline case (no key set) gives the
    most actionable message without even attempting the import.
    """
    config = config or {}
    api_key = os.environ.get("MEM0_API_KEY")
    if not api_key:
        raise SystemUnavailable(
            "mem0 adapter needs MEM0_API_KEY (and `pip install mem0ai`)"
        )
    try:
        from mem0 import MemoryClient  # type: ignore  # lazy: never imported offline
    except Exception as e:  # SDK not installed / import error
        raise SystemUnavailable(
            "mem0 adapter needs `pip install mem0ai`"
        ) from e
    client = MemoryClient(api_key=api_key)  # pragma: no cover - needs key + SDK
    return Mem0System(config, client)  # pragma: no cover - needs key + SDK
