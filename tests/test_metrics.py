"""Smoke tests for the scoring metrics and the system interface contract.

Keeps CI green from Wave 0 and guards the metric definitions the paper cites.
"""

from eval.metrics import exact_match, normalize_answer, numeric_within_tolerance, token_f1
from systems.base import Document, MemorySystem, QueryResult


def test_normalize_answer_strips_articles_and_punctuation():
    assert normalize_answer("The Dentist's office!") == "dentists office"


def test_exact_match():
    assert exact_match("the cat", "Cat") == 1.0
    assert exact_match("dog", "cat") == 0.0


def test_token_f1_partial_overlap():
    assert token_f1("the quick brown fox", "quick brown fox") > 0.8
    assert token_f1("apples", "oranges") == 0.0


def test_numeric_within_tolerance():
    assert numeric_within_tolerance(102.0, 100.0, rel_tol=0.05) == 1.0
    assert numeric_within_tolerance(120.0, 100.0, rel_tol=0.05) == 0.0


def test_query_result_shape():
    r = QueryResult(answer="x", supporting_doc_ids=["d1"], latency_ms=1.0, tokens_used=10)
    assert r.cost_usd == 0.0 and r.extra == {}


def test_memory_system_is_abstract():
    # Must not be instantiable without implementing the full interface.
    import pytest

    with pytest.raises(TypeError):
        MemorySystem()  # type: ignore[abstract]


def test_document_dataclass():
    d = Document(id="1", text="hi", timestamp="2026-01-01T00:00:00Z", type="note", persona_id="p1")
    assert d.metadata == {}
