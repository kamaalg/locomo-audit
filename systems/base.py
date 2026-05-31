"""Uniform interface every evaluated memory system implements.

Fair comparison is the entire point of this project: a baseline that returns
results in a different shape, or that hides its update cost, invalidates the
paper. Every system therefore speaks exactly this interface, and the eval
harness only ever touches systems through it.

See .claude/agents/baseline-implementer.md for the rules each implementation
must follow (vanilla defaults, pinned versions, honest cost measurement).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class Document:
    """A single unit of personal memory ingested into a system."""

    id: str
    text: str
    timestamp: str  # ISO 8601; tasks like T1/T6 depend on this being accurate
    type: str  # "journal" | "email" | "note" | "calendar"
    persona_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """The required output shape for every system's `query`.

    Latency and token/cost fields are not optional: update cost and query cost
    are reported metrics, not afterthoughts.
    """

    answer: str
    supporting_doc_ids: list[str]
    latency_ms: float
    tokens_used: int
    cost_usd: float = 0.0
    # Free-form per-system diagnostics (e.g. Strata's routing distribution).
    extra: dict[str, Any] = field(default_factory=dict)


class MemorySystem(ABC):
    """Abstract base every system in systems/ subclasses.

    Implementations must not special-case the benchmark. If a system does not
    natively return supporting doc ids or a cost figure, instrument it to do so
    rather than fabricating values.
    """

    name: str = "unnamed"
    config: dict[str, Any]

    @abstractmethod
    def ingest(self, documents: Iterable[Document]) -> None:
        """Add documents to memory."""

    @abstractmethod
    def update(self, doc_id: str, new_text: str) -> None:
        """Update an existing document. Critical for T4 (fact updates).

        Systems that cannot update incrementally (e.g. parametric memory) must
        reflect that honestly here — do not fake a cheap update path.
        """

    @abstractmethod
    def delete(self, doc_id: str) -> None:
        """Remove a document. Used for privacy / forgetting evaluation."""

    @abstractmethod
    def query(self, query: str, persona_id: str) -> QueryResult:
        """Answer a query against this persona's memory."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return {storage_bytes, indexed_docs, last_update_ts, ...}."""

    @abstractmethod
    def reset(self) -> None:
        """Wipe all memory. Called between persona evaluations."""
