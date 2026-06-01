"""Dataset loaders: bundled SAMPLE (offline) + real LoCoMo / LongMemEval (PI path).

The harness reads tasks in the repo's `TaskInstance` schema (the same `T*/*.json`
files the runner already globs). This module provides:

- `load_sample(path)`   — load the bundled, committed synthetic look-alike samples
                          under data/samples/. These are format-compatible synthetic
                          stand-ins stamped "SAMPLE": true — NOT the official datasets.
- `load_locomo(path)`   — documented code path to read the REAL LoCoMo dataset from a
- `load_longmemeval(path)` PI-provided local directory (downloaded under its own
                          license; never committed). Same TaskInstance shape, so the
                          only thing that changes between sample and full is the
                          `--tasks` path.

Each loader yields task records (dicts) plus the resolved corpus docs as
`systems.base.Document` objects, so an adapter can ingest them directly.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from systems.base import Document


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load documents.jsonl under a corpus directory into Document objects."""
    docs: list[Document] = []
    jsonl = corpus_dir / "documents.jsonl"
    if not jsonl.exists():
        return docs
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        docs.append(
            Document(
                id=rec["id"],
                text=rec["text"],
                timestamp=rec.get("timestamp", ""),
                type=rec.get("type", "note"),
                persona_id=rec.get("persona_id", "unknown"),
                metadata=rec.get("metadata", {}),
            )
        )
    return docs


def _iter_task_files(tasks_root: Path) -> Iterator[Path]:
    yield from sorted(tasks_root.glob("T*/*.json"))


def load_sample(sample_root: str | Path) -> list[dict[str, Any]]:
    """Load bundled SAMPLE task instances (data/samples/<name>/).

    Returns the raw task dicts. The committed files are stamped `"SAMPLE": true`
    and carry a `source` note marking them synthetic stand-ins.
    """
    root = Path(sample_root)
    out: list[dict[str, Any]] = []
    for path in _iter_task_files(root):
        rec = json.loads(path.read_text())
        rec["_path"] = str(path)
        out.append(rec)
    return out


def _load_real(path: str | Path, dataset: str) -> list[dict[str, Any]]:
    """Shared code path for the real (uncommitted) datasets.

    The PI downloads the official dataset under its own license into `path`. We
    expect the same TaskInstance + corpus layout the sample uses; if a vendor ships
    a different raw schema, the conversion belongs in a `convert_<dataset>.py`
    preprocessing script that emits this layout. This keeps the runner identical
    for sample vs full (`--tasks` path is the only change).
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(
            f"{dataset} not found at {root}. Download the official dataset under its "
            f"own license to a local path and pass it via --tasks; it is never committed."
        )
    return load_sample(root)


def load_locomo(path: str | Path = "data/external/locomo") -> list[dict[str, Any]]:
    """Load the REAL LoCoMo dataset from the converted local layout (never committed).

    `path` is the converted root produced by `eval.datasets.convert_locomo`
    (default `data/external/locomo`), holding `T*/*.json` answerable instances +
    `corpus/<persona>/documents.jsonl`. Adversarial (cat-5) instances live under
    `adversarial/` and are intentionally NOT returned by the runner's `T*/*.json`
    glob (different official metric — abstention, not F1).

    If the converted root is missing but a raw `external/locomo/data/locomo10.json`
    clone exists, we convert it on the fly so the loader is self-bootstrapping.
    LoCoMo is CC BY-NC 4.0; neither raw nor converted data is committed.
    """
    root = Path(path)
    if not (root / "T1").exists():
        raw = Path("external/locomo/data/locomo10.json")
        if raw.exists():
            from eval.datasets.convert_locomo import convert

            convert(raw, root)
    return _load_real(root, "LoCoMo")


def load_longmemeval(path: str | Path) -> list[dict[str, Any]]:
    """Load the REAL LongMemEval dataset from a PI-provided local path (never committed)."""
    return _load_real(path, "LongMemEval")
