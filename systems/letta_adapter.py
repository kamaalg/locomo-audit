"""letta — Letta / MemGPT adapter (wired-needs-keys; import-guarded).

Maps the harness's MemorySystem ops to Letta's self-editing agent memory: a Letta
agent owns its memory blocks and edits them via tool calls, so `ingest`/`update`/
`delete` push messages/archival passages into the agent and `query` asks the agent.

Like the other Tier-B vendor adapters, this is DISCOVERABLE with no SDK and no key
(so the leaderboard can show it as "wired, not yet run"), but it is NEVER runnable
without them. The SDK is imported LAZILY inside `build_system()`; if the `letta`
client is not installed or no credential is present, `build_system()` raises
`SystemUnavailable` with a clear message. The runner catches that, logs a SKIP, and
continues. No number this adapter could produce is ever written as independent until
a real keyed run.

mvpStatus: wired-needs-keys. needsKeys: true.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

from systems._registry import SystemUnavailable
from systems.base import Document, MemorySystem, QueryResult

SYSTEM_NAME = "letta"


class LettaSystem(MemorySystem):
    """Wired adapter over a live Letta client. Memory ops map to Letta agent ops:
    ingest -> insert archival passages; update -> rewrite a passage; delete -> drop
    a passage; query -> send a message to the agent and read its reply. Bodies are
    stubbed NotImplementedError until a real keyed run wires them against the live
    client; they are never exercised offline (build_system gates on key + SDK)."""

    name = SYSTEM_NAME

    def __init__(self, config: dict[str, Any], client: Any) -> None:
        self.config = config
        self._client = client

    def ingest(self, documents: Iterable[Document]) -> None:  # pragma: no cover - needs key
        raise NotImplementedError("Wired stub: implement against the live Letta client.")

    def update(self, doc_id: str, new_text: str) -> None:  # pragma: no cover - needs key
        raise NotImplementedError

    def delete(self, doc_id: str) -> None:  # pragma: no cover - needs key
        raise NotImplementedError

    def query(self, query: str, persona_id: str) -> QueryResult:  # pragma: no cover - needs key
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:  # pragma: no cover - needs key
        return {"indexed_docs": 0, "wired": True}

    def reset(self) -> None:  # pragma: no cover - needs key
        pass


def build_system(config: dict[str, Any] | None = None) -> MemorySystem:
    config = config or {}
    # Letta runs either as a cloud service (LETTA_API_KEY) or a self-hosted server
    # (LETTA_BASE_URL). Require at least one before attempting a live connection.
    api_key = os.environ.get("LETTA_API_KEY")
    base_url = os.environ.get("LETTA_BASE_URL")
    if not api_key and not base_url:
        raise SystemUnavailable(
            "set LETTA_API_KEY (Letta Cloud) or LETTA_BASE_URL (self-hosted server) "
            "to run the letta adapter"
        )
    try:
        from letta_client import Letta  # type: ignore
    except Exception as e:  # SDK not installed
        raise SystemUnavailable("pip install letta-client to run the letta adapter") from e
    if api_key:  # pragma: no cover - needs key + SDK
        client = Letta(token=api_key)
    else:  # pragma: no cover - needs SDK + self-hosted server
        client = Letta(base_url=base_url)
    return LettaSystem(config, client)  # pragma: no cover
