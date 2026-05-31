"""Tests for the eval harness task loader (Wave 3 wiring)."""

import json
from pathlib import Path

from eval.runner import load_tasks, summarize_tasks


def _write_task(d: Path, cat: str, diff: str, tid: str) -> None:
    (d / cat).mkdir(parents=True, exist_ok=True)
    (d / cat / f"{tid}.json").write_text(
        json.dumps(
            {
                "instance_id": tid,
                "category": cat,
                "difficulty": diff,
                "persona_id": "p001",
                "update_frequency": "UF1",
                "reasoning_depth": "RD0",
                "query": "q",
                "gold": {},
            }
        )
    )


def test_load_tasks_and_summary(tmp_path):
    _write_task(tmp_path, "T1", "easy", "T1-easy-0001")
    _write_task(tmp_path, "T1", "hard", "T1-hard-0002")
    _write_task(tmp_path, "T8", "medium", "T8-medium-0001")

    insts = list(load_tasks(tmp_path))
    assert len(insts) == 3
    assert {i.category for i in insts} == {"T1", "T8"}
    assert all(i.path.endswith(".json") for i in insts)

    summary = summarize_tasks(tmp_path)
    assert summary["total"] == 3
    assert summary["by_category"] == {"T1": 2, "T8": 1}
    assert summary["by_difficulty"]["easy"] == 1


def test_load_tasks_empty_dir(tmp_path):
    assert list(load_tasks(tmp_path)) == []
    assert summarize_tasks(tmp_path)["total"] == 0


def test_pilot_data_loads_if_present():
    # The committed pilot should load and cover all 8 categories.
    pilot = Path("data/synthetic/tasks")
    if not pilot.exists():
        return
    summary = summarize_tasks(pilot)
    assert summary["total"] >= 16
    assert set(summary["by_category"]) == {f"T{i}" for i in range(1, 9)}
