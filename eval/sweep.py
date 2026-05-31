"""eval/sweep.py — Paper 2 Stage-1 sensitivity sweep driver ($0, NO LLM calls).

ADDITIVE-ONLY. This module imports the existing, frozen infra (eval.metrics, the
free retriever adapters via systems._registry, eval.datasets.loaders) and never
edits any of it. It runs ONLY the free, deterministic axes of Paper 2's
"How reproducible are AI-memory benchmark scores?" sensitivity surface:

  (a) metric axis        — exact_match vs token_f1 vs normalized variants on the
                           extractive answers of the free systems.
  (b) retrieval-k axis   — recall@k of the answer-bearing doc for k in {1,3,5,10}.
  (c) denominator axis   — the 84 -> 58.44 fork: score over the answerable set
                           (LoCoMo cats 1-4) vs over answerable + adversarial
                           (cat-5), where cat-5 is scored by ABSTENTION (an empty
                           answer is correct; the retrievers always answer, so this
                           dilutes the headline number — the measured fork).

It makes ZERO Anthropic / LLM calls. The judge-MODEL swap (the headline rank-flip)
is Stage 2 = PAID = DEFERRED and is NOT run here; every Stage-2 cell is emitted as
PENDING so the surface is honest about what has and has not been measured.

Cost accounting: every retriever query reports cost_usd == 0.0; the driver asserts
the cumulative spend is exactly $0.00 and records that assertion in the output.

Usage:

    python -m eval.sweep \
        --tasks   data/external/locomo \
        --systems bm25,tfidf_rag \
        --ks      1,3,5,10 \
        --out     results/sweep

Outputs:
    results/sweep/stage1.json   — full tidy results + per-axis swing surface + manifest
    results/sweep/swing.csv     — compact swing surface (one row per axis sweep)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from eval import metrics
from eval.datasets.loaders import load_corpus
from systems.base import Document

# Free, keyless, deterministic systems only. Stage 1 must spend $0.
DEFAULT_SYSTEMS = ["bm25", "tfidf_rag"]
DEFAULT_KS = [1, 3, 5, 10]

# LoCoMo's official answerable categories are 1-4; category 5 is adversarial
# (official metric = abstention). The denominator fork is whether cat-5 is in the
# denominator (the 84 -> 58.44 spread Paper 1 forensically decomposes).
ANSWERABLE_CATS = {1, 2, 3, 4}
ADVERSARIAL_CAT = 5


# --------------------------------------------------------------------------- #
# Metric axis — all free, no LLM. Each returns a per-(prediction, gold) float.
# --------------------------------------------------------------------------- #


def _exact_match(pred: str, gold: str) -> float:
    """SQuAD-normalized exact match (eval.metrics primitive)."""
    return metrics.exact_match(pred, gold)


def _token_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 over normalized answers (eval.metrics primitive)."""
    return metrics.token_f1(pred, gold)


def _normalized_contains(pred: str, gold: str) -> float:
    """1.0 if the normalized gold appears as a substring of the normalized pred.

    A common 'lenient lexical' variant: credits a verbose answer that contains the
    gold phrase but would score <1 under strict EM and <1 under token_f1. Free,
    deterministic; uses only metrics.normalize_answer.
    """
    n_pred = metrics.normalize_answer(pred)
    n_gold = metrics.normalize_answer(gold)
    if not n_gold:
        return float(not n_pred)
    return float(n_gold in n_pred)


def _normalized_em_first_token(pred: str, gold: str) -> float:
    """1.0 if the first normalized token of pred equals the first of gold.

    A 'short-answer' normalized variant: many LoCoMo golds are a single token, so
    this approximates the strict short-answer grading some harnesses use. Free.
    """
    p = metrics.normalize_answer(pred).split()
    g = metrics.normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    return float(p[0] == g[0])


METRIC_FNS = {
    "exact_match": _exact_match,
    "token_f1": _token_f1,
    "normalized_contains": _normalized_contains,
    "normalized_em_first_token": _normalized_em_first_token,
}


# --------------------------------------------------------------------------- #
# Task loading (answerable cats 1-4 + adversarial cat-5), grouped by persona.
# --------------------------------------------------------------------------- #


def _git_sha() -> str:
    # Resolve the repo root from this file's location, not the shell cwd (sessions
    # often launch one level above the repo, which would yield 'unknown').
    repo_root = Path(__file__).resolve().parent.parent
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _iter_task_files(tasks_root: Path) -> Iterable[Path]:
    # Answerable instances live under T*/; adversarial (cat-5) under adversarial/.
    yield from sorted(tasks_root.glob("T*/*.json"))
    yield from sorted(tasks_root.glob("adversarial/*.json"))


def _load_tasks(tasks_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in _iter_task_files(tasks_root):
        rec = json.loads(path.read_text())
        cat = rec.get("metadata", {}).get("locomo_category")
        gold = rec.get("gold", {})
        out.append(
            {
                "instance_id": rec["instance_id"],
                "persona_id": rec.get("persona_id", "unknown"),
                "query": rec.get("query", ""),
                "locomo_category": cat,
                "is_adversarial": cat == ADVERSARIAL_CAT,
                "gold_answer": gold.get("answer", ""),
                "answer_type": gold.get("answer_type", "text"),
                "answer_bearing_doc_ids": rec.get("answer_bearing_doc_ids", []),
            }
        )
    return out


def _corpus_for_persona(tasks_root: Path, persona_id: str) -> list[Document]:
    return load_corpus(tasks_root / "corpus" / persona_id)


# --------------------------------------------------------------------------- #
# Retrieval (free) — build each free system at a given k, query every task once.
# --------------------------------------------------------------------------- #


def _build_system(name: str, top_k: int):
    """Instantiate a free retriever with the requested top_k. NO LLM, cost 0."""
    from systems._registry import discover

    reg = discover()
    if name not in reg:
        raise KeyError(f"Unknown system {name!r}. Discovered: {sorted(reg)}")
    return reg[name]({"top_k": top_k})


def _retrieve_all(
    system_name: str,
    top_k: int,
    tasks: list[dict[str, Any]],
    tasks_root: Path,
) -> tuple[list[dict[str, Any]], float]:
    """Query every task once at this (system, k). Returns (rows, total_cost_usd).

    Efficient: ingest each persona's corpus ONCE, then query all its tasks. The
    extractive answer for bm25/tfidf is always the top-1 passage's best sentence
    (k-invariant for the answer STRING), while supporting_doc_ids carries the full
    top-k ranking (k-sensitive — this is what the retrieval-k recall axis reads).
    """
    by_persona: dict[str, list[dict[str, Any]]] = {}
    for t in tasks:
        by_persona.setdefault(t["persona_id"], []).append(t)

    rows: list[dict[str, Any]] = []
    total_cost = 0.0
    for persona_id, ptasks in sorted(by_persona.items()):
        system = _build_system(system_name, top_k)
        docs = _corpus_for_persona(tasks_root, persona_id)
        system.reset()
        system.ingest(docs)
        for t in ptasks:
            qr = system.query(t["query"], persona_id)
            total_cost += float(qr.cost_usd)  # must be 0.0 for free systems
            retrieved_ids = list(qr.supporting_doc_ids)
            gold_ids = set(t["answer_bearing_doc_ids"])
            hit = bool(gold_ids & set(retrieved_ids)) if gold_ids else False
            rows.append(
                {
                    "system": system_name,
                    "top_k": top_k,
                    "instance_id": t["instance_id"],
                    "persona_id": persona_id,
                    "locomo_category": t["locomo_category"],
                    "is_adversarial": t["is_adversarial"],
                    "answer": qr.answer,
                    "gold_answer": t["gold_answer"],
                    "answer_type": t["answer_type"],
                    "retrieved_doc_ids": retrieved_ids,
                    "answer_bearing_doc_ids": list(gold_ids),
                    "recall_at_k": float(hit),
                    "cost_usd": float(qr.cost_usd),
                }
            )
    return rows, total_cost


# --------------------------------------------------------------------------- #
# Aggregation — the three free axes.
# --------------------------------------------------------------------------- #


def _adversarial_abstention_score(row: dict[str, Any]) -> float:
    """Official cat-5 metric: correct == the system ABSTAINED (empty answer).

    Our retrievers always emit a non-empty top passage, so they (almost) never
    abstain -> ~0 on cat-5. Including cat-5 in the denominator therefore drags the
    headline down: the measured 'denominator fork'.
    """
    return float(not (row.get("answer") or "").strip())


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _aggregate_cell(rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    """Aggregate one (system, k) cell across the metric + denominator axes."""
    fn = METRIC_FNS[metric_name]
    answerable = [r for r in rows if not r["is_adversarial"]]
    adversarial = [r for r in rows if r["is_adversarial"]]

    # answerable-set score under this metric
    ans_scores = [fn(r["answer"], r["gold_answer"]) for r in answerable]
    # adversarial scored by abstention (official cat-5 metric)
    adv_scores = [_adversarial_abstention_score(r) for r in adversarial]

    score_excl = _mean(ans_scores)  # cat-5 EXCLUDED from denominator
    # cat-5 INCLUDED: answerable graded by metric, adversarial graded by abstention,
    # pooled into one denominator (answerable + adversarial).
    pooled = ans_scores + adv_scores
    score_incl = _mean(pooled)

    return {
        "metric": metric_name,
        "n_answerable": len(answerable),
        "n_adversarial": len(adversarial),
        "score_denom_excl_cat5": round(score_excl, 6),
        "score_denom_incl_cat5": round(score_incl, 6),
        "adversarial_abstention_acc": round(_mean(adv_scores), 6),
        "recall_at_k_answerable": round(_mean([r["recall_at_k"] for r in answerable]), 6),
    }


def _swing(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "range": 0.0}
    return {
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "range": round(max(values) - min(values), 6),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_sweep(
    tasks_root: Path,
    systems: list[str],
    ks: list[int],
) -> dict[str, Any]:
    tasks = _load_tasks(tasks_root)
    n_answerable = sum(1 for t in tasks if not t["is_adversarial"])
    n_adversarial = sum(1 for t in tasks if t["is_adversarial"])

    cells: list[dict[str, Any]] = []
    total_cost = 0.0
    for system_name in systems:
        for k in ks:
            rows, cost = _retrieve_all(system_name, k, tasks, tasks_root)
            total_cost += cost
            for metric_name in METRIC_FNS:
                agg = _aggregate_cell(rows, metric_name)
                agg.update({"system": system_name, "top_k": k})
                cells.append(agg)

    # ---- swing surface: per axis, how far does a single system's score move? ----
    swing_rows: list[dict[str, Any]] = []

    # AXIS 1: metric (fix system, fix k; vary metric) on the answerable set.
    for system_name in systems:
        for k in ks:
            vals = [
                c["score_denom_excl_cat5"]
                for c in cells
                if c["system"] == system_name and c["top_k"] == k
            ]
            sw = _swing(vals)
            swing_rows.append(
                {
                    "axis": "metric",
                    "system": system_name,
                    "held_fixed": f"k={k}, denom=excl_cat5",
                    "varied_over": ",".join(METRIC_FNS),
                    **sw,
                }
            )

    # AXIS 2: retrieval-k (fix system, fix metric; vary k) — score is k-invariant
    # for these extractive systems, but recall@k is NOT — report both.
    for system_name in systems:
        for metric_name in METRIC_FNS:
            score_vals = [
                c["score_denom_excl_cat5"]
                for c in cells
                if c["system"] == system_name and c["metric"] == metric_name
            ]
            swing_rows.append(
                {
                    "axis": "retrieval_k_score",
                    "system": system_name,
                    "held_fixed": f"metric={metric_name}, denom=excl_cat5",
                    "varied_over": ",".join(str(k) for k in ks),
                    **_swing(score_vals),
                }
            )
        # recall@k swing (one per system; metric-independent)
        recall_vals = [
            c["recall_at_k_answerable"]
            for c in cells
            if c["system"] == system_name and c["metric"] == "token_f1"
        ]
        swing_rows.append(
            {
                "axis": "retrieval_k_recall",
                "system": system_name,
                "held_fixed": "denom=excl_cat5",
                "varied_over": ",".join(str(k) for k in ks),
                **_swing(recall_vals),
            }
        )

    # AXIS 3: denominator (fix system, fix metric, fix k; cat-5 excl vs incl).
    for system_name in systems:
        for k in ks:
            for metric_name in METRIC_FNS:
                c = next(
                    cc
                    for cc in cells
                    if cc["system"] == system_name
                    and cc["top_k"] == k
                    and cc["metric"] == metric_name
                )
                vals = [c["score_denom_excl_cat5"], c["score_denom_incl_cat5"]]
                swing_rows.append(
                    {
                        "axis": "denominator_cat5",
                        "system": system_name,
                        "held_fixed": f"metric={metric_name}, k={k}",
                        "varied_over": "excl_cat5,incl_cat5",
                        **_swing(vals),
                    }
                )

    manifest = {
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "stage": "stage1",
        "tasks_root": str(tasks_root),
        "systems": systems,
        "ks": ks,
        "metrics": list(METRIC_FNS),
        "n_answerable": n_answerable,
        "n_adversarial": n_adversarial,
        "llm_calls": 0,
        "total_cost_usd": round(total_cost, 6),
        "zero_cost_asserted": True,
        "note": (
            "Stage 1 is $0: free retrievers + free metric/denominator/retrieval-k "
            "re-aggregation only. The judge-MODEL / judge-prompt / answer-prompt "
            "axes are Stage 2 (PAID, DEFERRED) and are emitted as PENDING below."
        ),
    }
    if total_cost != 0.0:
        raise RuntimeError(
            f"Stage-1 invariant violated: total_cost_usd={total_cost} != 0.0. "
            "A free-only sweep must not spend anything."
        )

    stage2_pending = {
        "judge_model": "PENDING",
        "judge_prompt": "PENDING (eval/prompts/judge_{strict,lenient,cot_rubric}.txt)",
        "answer_prompt": "PENDING (eval/prompts/answer_{concise,verbose}.txt)",
        "seed": "PENDING",
        "rank_stability_kendall_tau": "PENDING",
        "status": "DEFERRED — PAID — not run in this workflow",
    }

    return {
        "manifest": manifest,
        "cells": cells,
        "swing_surface": swing_rows,
        "stage2_pending": stage2_pending,
    }


def _write_csv(swing_rows: list[dict[str, Any]], path: Path) -> None:
    cols = ["axis", "system", "held_fixed", "varied_over", "min", "max", "range"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in swing_rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main() -> int:
    p = argparse.ArgumentParser(description="Paper 2 Stage-1 sensitivity sweep ($0, no LLM)")
    p.add_argument("--tasks", default="data/external/locomo",
                   help="Converted real LoCoMo root (T*/, adversarial/, corpus/).")
    p.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS),
                   help="Comma-separated FREE systems only (bm25,tfidf_rag).")
    p.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS),
                   help="Comma-separated retrieval-k values.")
    p.add_argument("--out", default="results/sweep",
                   help="Output directory for stage1.json + swing.csv.")
    args = p.parse_args()

    tasks_root = Path(args.tasks)
    systems = [s for s in args.systems.split(",") if s.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_sweep(tasks_root, systems, ks)

    (out_dir / "stage1.json").write_text(json.dumps(result, indent=2))
    _write_csv(result["swing_surface"], out_dir / "swing.csv")

    m = result["manifest"]
    print(json.dumps({
        "wrote": [str(out_dir / "stage1.json"), str(out_dir / "swing.csv")],
        "n_answerable": m["n_answerable"],
        "n_adversarial": m["n_adversarial"],
        "total_cost_usd": m["total_cost_usd"],
        "llm_calls": m["llm_calls"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
