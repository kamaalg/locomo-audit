"""Tests for eval/sweep.py — Paper 2 Stage-1 sensitivity sweep (must stay $0).

Additive: these tests exercise the new sweep driver without touching the frozen
audit harness. They use a tiny in-memory corpus + tasks (no real LoCoMo needed) so
they run offline and confirm the invariants that make Stage 1 publishable and free:
  * ZERO cost / ZERO LLM calls,
  * the three free axes (metric / retrieval-k / denominator) are all computed,
  * Stage-2 cells are emitted PENDING (never silently scored).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import sweep


def _write_task(d: Path, sub: str, inst: str, query: str, answer: str,
                cat: int, persona: str, answer_bearing: list[str], answer_type="text"):
    p = d / sub
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{inst}.json").write_text(json.dumps({
        "instance_id": inst,
        "category": "T1" if cat != 5 else "adversarial",
        "persona_id": persona,
        "query": query,
        "answer_bearing_doc_ids": answer_bearing,
        "gold": {"answer": answer, "answer_type": answer_type},
        "metadata": {"locomo_category": cat},
    }))


@pytest.fixture()
def tiny_root(tmp_path: Path) -> Path:
    persona = "p1"
    corpus = tmp_path / "corpus" / persona
    corpus.mkdir(parents=True)
    docs = [
        {"id": "d1", "text": "Caroline researched adoption agencies.", "persona_id": persona},
        {"id": "d2", "text": "The weather was sunny on the trip.", "persona_id": persona},
        {"id": "d3", "text": "They cooked pasta for dinner together.", "persona_id": persona},
    ]
    (corpus / "documents.jsonl").write_text("\n".join(json.dumps(x) for x in docs))
    _write_task(tmp_path, "T1", "p1-0001", "What did Caroline research?",
                "adoption agencies", 1, persona, ["d1"])
    _write_task(tmp_path, "adversarial", "p1-adv-0001", "What did Caroline never mention?",
                "", 5, persona, ["d2"], answer_type="abstention")
    return tmp_path


def test_sweep_is_zero_cost_and_zero_llm(tiny_root: Path):
    r = sweep.run_sweep(tiny_root, ["bm25", "tfidf_rag"], [1, 3])
    m = r["manifest"]
    assert m["total_cost_usd"] == 0.0
    assert m["llm_calls"] == 0
    assert m["zero_cost_asserted"] is True
    # every cell carried zero cost from the retriever query path
    # (run_sweep raises if the cumulative spend is ever non-zero)


def test_sweep_counts_answerable_and_adversarial(tiny_root: Path):
    r = sweep.run_sweep(tiny_root, ["bm25"], [1])
    assert r["manifest"]["n_answerable"] == 1
    assert r["manifest"]["n_adversarial"] == 1


def test_all_three_free_axes_present(tiny_root: Path):
    r = sweep.run_sweep(tiny_root, ["bm25", "tfidf_rag"], [1, 3])
    axes = {row["axis"] for row in r["swing_surface"]}
    assert {"metric", "retrieval_k_recall", "denominator_cat5"} <= axes
    # retrieval_k_score is reported too (it is k-invariant for extractive systems)
    assert "retrieval_k_score" in axes


def test_denominator_fork_dilutes_score(tiny_root: Path):
    """Including cat-5 (abstention, retriever always answers) must not raise the score."""
    r = sweep.run_sweep(tiny_root, ["bm25"], [1])
    for c in r["cells"]:
        assert c["score_denom_incl_cat5"] <= c["score_denom_excl_cat5"] + 1e-9
        assert c["adversarial_abstention_acc"] == 0.0  # retriever never abstains


def test_recall_at_k_is_monotonic_nondecreasing(tiny_root: Path):
    r = sweep.run_sweep(tiny_root, ["bm25"], [1, 3])
    by_k = {c["top_k"]: c["recall_at_k_answerable"]
            for c in r["cells"] if c["metric"] == "token_f1"}
    assert by_k[3] >= by_k[1]


def test_stage2_marked_pending(tiny_root: Path):
    r = sweep.run_sweep(tiny_root, ["bm25"], [1])
    s2 = r["stage2_pending"]
    assert s2["judge_model"] == "PENDING"
    assert "DEFERRED" in s2["status"]


def test_metric_axis_includes_em_and_f1(tiny_root: Path):
    assert "exact_match" in sweep.METRIC_FNS
    assert "token_f1" in sweep.METRIC_FNS
    assert len(sweep.METRIC_FNS) >= 3
