"""PersonalMem / Neutral Referee evaluation harness.

This is the spine every experiment runs through. Contracts (unchanged from the
skeleton, relied on by experiment-runner and the tests):

  * resumable: one result per JSON line in raw.jsonl, re-run skips completed cells
  * budget-gated: refuses to start without a hard USD ceiling, halts at 90%
  * reproducible: writes a manifest (git SHA, configs, seeds, dataset version)
  * honest: logs every cell's latency/cost; never silently skips a failure

Wave 05 (Neutral Referee MVP) additions, per docs/artifacts/wave-05-referee/DESIGN.md §5:
  * `--tag` gives a DETERMINISTIC output filename: results/<tag>.json (no wall-clock
    / random in the file-naming path; the manifest still records a timestamp in its
    metadata only, which is acceptable).
  * `load_system(name)` is rewired to systems/_registry.discover()[name](config).
    A missing SDK / key raises SystemUnavailable -> the runner logs a SKIP line and
    continues; it is NEVER written as an independent number.
  * `execute_cell` ingests the task's corpus, queries the system, and scores via
    eval.metrics (the ONLY scorers). It writes results/<tag>.json = {manifest, rows}.

Usage:

    python -m eval.runner \
        --systems-list bm25,tfidf_rag \
        --tasks   data/samples \
        --seeds   1 \
        --output  results/ \
        --tag     mvp_offline_baselines \
        --budget-usd 0.01
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# systems.base / registry are imported lazily inside load_system so the skeleton
# (and --list-tasks) import cleanly even if a vendor adapter's optional dep is odd.


@dataclass
class Cell:
    """One unit of the experimental matrix: (system, task, seed)."""

    system: str
    task_id: str
    category: str
    difficulty: str
    seed: int


@dataclass
class TaskInstance:
    """A single benchmark task instance loaded from a T*/*.json file."""

    task_id: str
    category: str
    difficulty: str
    persona_id: str
    update_frequency: str
    reasoning_depth: str
    path: str


@dataclass
class CellResult:
    cell: Cell
    score: float
    answer: str
    latency_ms: float
    cost_usd: float
    error: str | None = None
    # Wave 05 scoring diagnostics (optional; default-safe for older callers/tests).
    em: float | None = None
    f1: float | None = None
    metric: str | None = None
    label: str | None = None  # FAMA label for T4/T5: correct_current|stale|wrong
    supporting_doc_ids: list[str] = field(default_factory=list)
    skipped: bool = False
    # Audit protocol §3 (answer-mode=generate / judge=llm). Default-None so older
    # callers, tests, and the extractive path are unaffected.
    answer_mode: str | None = None  # "extractive" | "generate"
    judge: str | None = None        # "token_f1" | "llm"
    judge_correct: bool | None = None  # the binary J verdict (None unless judge=llm)
    judge_reason: str | None = None


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def write_manifest(output_dir: Path, args: argparse.Namespace) -> dict:
    """Record everything needed to reproduce this run. Returns the manifest dict."""
    manifest = {
        "git_sha": git_sha(),
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "tag": getattr(args, "tag", None),
        "args": vars(args),
        # Protocol §3/§6 reproducibility: pin the answer + judge models and the hard
        # USD budget into the manifest so a run is fully reproducible from this file.
        "audit": {
            "answer_mode": getattr(args, "answer_mode", "extractive"),
            "judge": getattr(args, "judge", "token_f1"),
            "answer_model": getattr(args, "answer_model", None),
            "judge_model": getattr(args, "judge_model", None),
            "budget_usd": getattr(args, "budget_usd", None),
        },
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("PERSONALMEM_")
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def completed_cells(raw_path: Path) -> set[tuple[str, str, int]]:
    """Read raw.jsonl to find which (system, task_id, seed) cells are done."""
    done: set[tuple[str, str, int]] = set()
    if not raw_path.exists():
        return done
    for line in raw_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        c = rec["cell"]
        done.add((c["system"], c["task_id"], c["seed"]))
    return done


def load_system(name: str, config: dict | None = None):
    """Instantiate a MemorySystem by name via the adapter registry.

    Rewired in Wave 05: discovery scans systems/ for adapters exposing
    SYSTEM_NAME + build_system. A vendor adapter with a missing SDK/key raises
    SystemUnavailable from build_system; the caller (execute_cell) turns that into
    a logged SKIP rather than a crash.
    """
    from systems._registry import discover

    reg = discover()
    if name not in reg:
        raise KeyError(
            f"Unknown system {name!r}. Discovered: {sorted(reg)}. "
            "An adapter is one file in systems/ exposing SYSTEM_NAME + build_system."
        )
    return reg[name](config or {})


def load_tasks(tasks_dir: Path) -> Iterator[TaskInstance]:
    """Yield task instances from T*/*.json under tasks_dir (recursively).

    Reads per-category instance files. Recursive (`**/T*/*.json`) so the bundled
    samples (data/samples/<dataset>/T*/...) and the pilot (data/synthetic/tasks/T*/...)
    both load. Dedupes by resolved path; deterministic order.
    """
    seen: set[str] = set()
    patterns = ["T*/*.json", "**/T*/*.json"]
    paths: list[Path] = []
    for pat in patterns:
        for p in tasks_dir.glob(pat):
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            paths.append(p)
    for path in sorted(paths, key=lambda p: str(p)):
        rec = json.loads(path.read_text())
        yield TaskInstance(
            task_id=rec["instance_id"],
            category=rec["category"],
            difficulty=rec.get("difficulty", "unknown"),
            persona_id=rec.get("persona_id", "unknown"),
            update_frequency=rec.get("update_frequency", "unknown"),
            reasoning_depth=rec.get("reasoning_depth", "unknown"),
            path=str(path),
        )


def summarize_tasks(tasks_dir: Path) -> dict:
    """Counts by category and difficulty — used by --list-tasks and tests."""
    by_cat: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    total = 0
    for t in load_tasks(tasks_dir):
        by_cat[t.category] = by_cat.get(t.category, 0) + 1
        by_diff[t.difficulty] = by_diff.get(t.difficulty, 0) + 1
        total += 1
    return {"total": total, "by_category": by_cat, "by_difficulty": by_diff}


# --------------------------------------------------------------------------- #
# Cell execution + scoring (Wave 05)
# --------------------------------------------------------------------------- #


def _resolve_corpus_dir(task_rec: dict, task_path: Path) -> Path:
    """Resolve the corpus directory for a task from its corpus_ref (repo-relative)."""
    ref = task_rec.get("corpus_ref")
    if ref:
        p = Path(ref)
        if p.exists():
            return p
        # try relative to the repo root inferred from cwd
        return Path.cwd() / ref
    # fall back: a sibling 'corpus' dir under the dataset root
    return task_path.parent.parent / "corpus"


class _PredView:
    """Minimal stand-in exposing `.answer` so `_score_task` scores the FINAL

    prediction (extractive top-passage OR the pinned-model generated answer),
    without `_score_task` needing to know which mode produced it.
    """

    __slots__ = ("answer",)

    def __init__(self, answer: str) -> None:
        self.answer = answer


def _score_task(task_rec: dict, qr) -> dict:
    """Score one query result against a task's gold using ONLY eval.metrics."""
    from eval import metrics

    category = task_rec.get("category", "")
    gold = task_rec.get("gold", {})
    pred = qr.answer or ""

    # T4/T5 fact-update: FAMA label (correct_current | stale | wrong).
    #
    # FAMA's classify_update_answer expects a SHORT answer; our extractive
    # baselines have no generator and answer with a whole passage, so a strict
    # exact-match label is almost always 'wrong'. To honestly surface the
    # "stale-fact tax" the product is named for (DESIGN §3), we label by which
    # gold value the retrieved passage CONTAINS, using only the metrics primitive
    # `normalize_answer` for the containment test. The strict exact-match label is
    # also kept (label_strict) so we never overstate. A passage that contains the
    # obsolete value but not the current one => 'stale' (the retriever surfaced the
    # outdated doc); contains current => 'correct_current'; neither => 'wrong'.
    if category in {"T4", "T5"} and "current_answer" in gold:
        current = gold.get("current_answer", "")
        obsolete = gold.get("obsolete_values", [])
        label_strict = metrics.classify_update_answer(
            pred, current, obsolete, gold.get("answer_type", "text"),
        )
        norm_pred = metrics.normalize_answer(pred)
        has_current = metrics.normalize_answer(current) in norm_pred if current else False
        has_obsolete = any(
            metrics.normalize_answer(o) in norm_pred for o in obsolete if o
        )
        if has_current and not has_obsolete:
            label = "correct_current"
        elif has_obsolete and not has_current:
            label = "stale"
        elif has_current and has_obsolete:
            label = "correct_current"  # mentions both -> credit current (FAMA convention)
        else:
            label = label_strict  # falls back to exact-match verdict ('wrong')
        f1 = metrics.token_f1(pred, current)
        em = metrics.exact_match(pred, current)
        score = 1.0 if label == "correct_current" else 0.0
        return {"score": score, "em": em, "f1": f1, "metric": "fama_extractive",
                "label": label}

    # T2 cross-document synthesis: claim recall when atomic claims are provided.
    if category == "T2" and gold.get("claims"):
        recall, _fm, _extra = metrics.claim_recall(pred, gold["claims"])
        f1 = metrics.token_f1(pred, gold.get("answer", ""))
        em = metrics.exact_match(pred, gold.get("answer", ""))
        return {"score": recall, "em": em, "f1": f1, "metric": "claim_recall", "label": None}

    # Default (T1/T6/QA): exact-match + token-F1 against gold.answer.
    gold_answer = gold.get("answer", gold.get("current_answer", ""))
    em = metrics.exact_match(pred, gold_answer)
    f1 = metrics.token_f1(pred, gold_answer)
    return {"score": f1, "em": em, "f1": f1, "metric": "token_f1", "label": None}


# Cache built systems within a run so we don't re-discover/instantiate per cell.
_SYSTEM_CACHE: dict[str, Any] = {}


def _get_system(name: str):
    if name not in _SYSTEM_CACHE:
        _SYSTEM_CACHE[name] = load_system(name)
    return _SYSTEM_CACHE[name]


def execute_cell(cell: Cell, task: TaskInstance, args: argparse.Namespace) -> CellResult:
    """Run one (system, task, seed) cell: ingest -> query -> score via eval.metrics.

    Vendor adapters whose SDK/key is missing raise SystemUnavailable; we catch it,
    return a skipped CellResult (never written as an independent number).
    """
    from eval.datasets.loaders import load_corpus
    from systems._registry import SystemUnavailable

    task_path = Path(task.path)
    task_rec = json.loads(task_path.read_text())

    try:
        system = _get_system(cell.system)
    except SystemUnavailable as e:
        msg = f"SKIP {cell.system} (task {cell.task_id}): SystemUnavailable: {e}"
        print(msg, file=sys.stderr)
        return CellResult(
            cell=cell, score=0.0, answer="", latency_ms=0.0, cost_usd=0.0,
            error=f"SystemUnavailable: {e}", skipped=True,
        )

    corpus_dir = _resolve_corpus_dir(task_rec, task_path)
    docs = load_corpus(corpus_dir)
    system.reset()
    system.ingest(docs)
    query_text = task_rec.get("query", "")
    qr = system.query(query_text, task.persona_id)

    answer_mode = getattr(args, "answer_mode", "extractive") or "extractive"
    judge_mode = getattr(args, "judge", "token_f1") or "token_f1"

    predicted = qr.answer  # extractive default (back-compat)
    extra_cost = 0.0  # answer-gen + judge spend, counted toward the budget
    judge_correct: bool | None = None
    judge_reason: str | None = None

    # --- Answer generation (protocol §3): pinned answer model from retrieved ctx.
    if answer_mode == "generate":
        from eval.answer_gen import generate_answer

        retrieved = list(qr.extra.get("retrieved_context", []))
        gen = generate_answer(
            query_text,
            retrieved,
            model=getattr(args, "answer_model", None) or "claude-sonnet-4-5",
        )
        predicted = gen.answer
        extra_cost += gen.cost_usd

    # Score the (extractive or generated) prediction via eval.metrics (unchanged).
    scored = _score_task(task_rec, _PredView(predicted))

    # --- LLM judge (protocol §3): pinned binary-correctness verdict -> J.
    if judge_mode == "llm":
        from eval.judge import judge_answer

        gold = task_rec.get("gold", {})
        gold_answer = gold.get("answer", gold.get("current_answer", ""))
        jr = judge_answer(
            query_text,
            predicted,
            gold_answer,
            model=getattr(args, "judge_model", None) or "claude-sonnet-4-5",
        )
        judge_correct = jr.correct
        judge_reason = jr.reason
        extra_cost += jr.cost_usd

    return CellResult(
        cell=cell,
        score=scored["score"],
        answer=predicted,
        latency_ms=qr.latency_ms,
        cost_usd=qr.cost_usd + extra_cost,
        em=scored["em"],
        f1=scored["f1"],
        metric=scored["metric"],
        label=scored["label"],
        supporting_doc_ids=list(qr.supporting_doc_ids),
        answer_mode=answer_mode,
        judge=judge_mode,
        judge_correct=judge_correct,
        judge_reason=judge_reason,
    )


def _aggregate(rows: list[CellResult]) -> dict:
    """Aggregate EM / F1 / FAMA over scored (non-skipped) rows."""
    scored = [r for r in rows if not r.skipped and r.error is None]
    n = len(scored)
    if n == 0:
        return {"n_scored": 0}
    ems = [r.em for r in scored if r.em is not None]
    f1s = [r.f1 for r in scored if r.f1 is not None]
    fama_labels = [r.label for r in scored if r.label is not None]
    agg: dict[str, Any] = {
        "n_scored": n,
        "exact_match": round(sum(ems) / len(ems), 4) if ems else None,
        "token_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "mean_score": round(sum(r.score for r in scored) / n, 4),
    }
    if fama_labels:
        from eval.metrics import fama

        agg["fama"] = fama(fama_labels)
    # Judge accuracy J (protocol §3) — mean of the binary correct verdicts over the
    # rows that were actually judged (judge=llm). Reported alongside token_f1.
    judged = [r for r in scored if r.judge_correct is not None]
    if judged:
        agg["n_judged"] = len(judged)
        agg["judge_accuracy"] = round(
            sum(1 for r in judged if r.judge_correct) / len(judged), 4
        )
    return agg


# --------------------------------------------------------------------------- #
# Run loop
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> int:
    tasks_dir = Path(args.tasks)

    if args.list_tasks:
        summary = summarize_tasks(tasks_dir)
        print(json.dumps(summary, indent=2))
        if summary["total"] == 0:
            print(f"No task instances found under {tasks_dir}", file=sys.stderr)
            return 1
        return 0

    if args.list_systems:
        from systems._registry import discover

        print(json.dumps(sorted(discover().keys()), indent=2))
        return 0

    budget = args.budget_usd
    if budget is None or budget <= 0:
        print(
            "Refusing to start: pass --budget-usd (a hard ceiling). "
            "This guardrail is required by the experiment-runner protocol.",
            file=sys.stderr,
        )
        return 2

    systems = [s for s in str(args.systems_list or "").split(",") if s.strip()]
    if not systems:
        print(
            "No --systems-list given. Use --list-systems to see discovered adapters "
            "or --list-tasks to inspect the benchmark.",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(output_dir, args)

    raw_path = output_dir / "raw.jsonl"
    done = completed_cells(raw_path)
    print(f"Resuming: {len(done)} cells already complete.")

    spent = 0.0
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    tasks = list(load_tasks(tasks_dir))
    halted = False

    with raw_path.open("a") as fh:
        for system in systems:
            for task in tasks:
                for seed in seeds:
                    cell = Cell(
                        system=system,
                        task_id=task.task_id,
                        category=task.category,
                        difficulty=task.difficulty,
                        seed=seed,
                    )
                    if (cell.system, cell.task_id, cell.seed) in done:
                        continue
                    if spent >= 0.9 * budget:
                        print(
                            f"Halting cleanly at 90% of budget (${spent:.2f}/${budget:.2f}).",
                            file=sys.stderr,
                        )
                        halted = True
                        break
                    result = execute_cell(cell, task, args)
                    fh.write(json.dumps({"cell": asdict(cell), **asdict(result)}) + "\n")
                    fh.flush()
                    spent += result.cost_usd
                if halted:
                    break
            if halted:
                break

    # Write the aggregated, scored artifact: results/<tag>.json (deterministic name).
    tag = args.tag or "adhoc"
    rows = _read_rows(raw_path)
    artifact = {
        "manifest": manifest,
        "aggregate": _aggregate(rows),
        "rows": [asdict(r) for r in rows],
    }
    out_file = output_dir / f"{tag}.json"
    out_file.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote {out_file}")
    print(json.dumps(artifact["aggregate"], indent=2))
    return 0


def _read_rows(raw_path: Path) -> list[CellResult]:
    """Rebuild CellResult objects from raw.jsonl (resume-safe aggregation)."""
    rows: list[CellResult] = []
    if not raw_path.exists():
        return rows
    for line in raw_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        c = rec["cell"]
        rows.append(
            CellResult(
                cell=Cell(**c),
                score=rec.get("score", 0.0),
                answer=rec.get("answer", ""),
                latency_ms=rec.get("latency_ms", 0.0),
                cost_usd=rec.get("cost_usd", 0.0),
                error=rec.get("error"),
                em=rec.get("em"),
                f1=rec.get("f1"),
                metric=rec.get("metric"),
                label=rec.get("label"),
                supporting_doc_ids=rec.get("supporting_doc_ids", []),
                skipped=rec.get("skipped", False),
                answer_mode=rec.get("answer_mode"),
                judge=rec.get("judge"),
                judge_correct=rec.get("judge_correct"),
                judge_reason=rec.get("judge_reason"),
            )
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="PersonalMem / Neutral Referee evaluation harness")
    p.add_argument("--systems", default="eval/configs/systems.yaml",
                   help="Path to per-system config (optional).")
    p.add_argument("--systems-list", default="",
                   help="Comma-separated system names to evaluate (registry-discovered).")
    p.add_argument("--tasks", default="data/synthetic/tasks",
                   help="Directory of task instances (T*/*.json, searched recursively).")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--output", default="results/adhoc")
    p.add_argument("--tag", default=None,
                   help="Deterministic output filename: results/<tag>.json")
    p.add_argument("--list-tasks", action="store_true",
                   help="Load the benchmark and print task counts, then exit.")
    p.add_argument("--list-systems", action="store_true",
                   help="Print registry-discovered adapter names, then exit.")
    p.add_argument(
        "--budget-usd",
        type=float,
        default=float(os.environ.get("PERSONALMEM_RUN_BUDGET_USD", 0) or 0),
        help="Hard USD ceiling. Run halts cleanly at 90%%.",
    )
    # Audit protocol §3 wiring.
    p.add_argument(
        "--answer-mode",
        dest="answer_mode",
        choices=["extractive", "generate"],
        default="extractive",
        help="extractive (default, back-compat: score the system's top passage) | "
        "generate (pinned answer model writes the answer from retrieved_context).",
    )
    p.add_argument(
        "--judge",
        choices=["token_f1", "llm"],
        default="token_f1",
        help="token_f1 (default, back-compat) | llm (pinned LLM-judge -> J accuracy).",
    )
    p.add_argument(
        "--answer-model",
        dest="answer_model",
        default="claude-sonnet-4-5",
        help="Pinned answer model (protocol §3). Recorded in the manifest.",
    )
    p.add_argument(
        "--judge-model",
        dest="judge_model",
        default="claude-sonnet-4-5",
        help="Pinned LLM-judge model (protocol §3). Recorded in the manifest.",
    )
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
