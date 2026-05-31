"""Convert the REAL LoCoMo dataset (snap-research/locomo, data/locomo10.json) into
the repo's committed-shape TaskInstance + corpus layout the runner already consumes.

LoCoMo is CC BY-NC 4.0. The raw JSON and the converted instances are NEVER committed
(see .gitignore: data/external/). This script regenerates them from a local clone.

Source schema (data/locomo10.json), per snap-research/locomo README.MD:
  - 10 samples; each = one multi-session conversation between two speakers.
  - sample['conversation']: speaker_a, speaker_b, session_<n>_date_time, session_<n>
    (a list of turns: {speaker, dia_id ("D<sess>:<turn>"), text [, img_url,
    blip_caption, query]}).
  - sample['qa']: list of {question, answer, evidence:[dia_id...], category}.

LoCoMo QA categories (paper §, README/index.html: "single-hop, multi-hop, temporal,
commonsense/world-knowledge, and adversarial"; confirmed against task_eval/evaluation.py):
  1 = multi-hop reasoning
  2 = temporal reasoning
  3 = open-domain / commonsense knowledge (gold answer is "<ans>; <explanation>";
      official eval splits on ';' and scores the first clause)
  4 = single-hop recall
  5 = adversarial / unanswerable (no `answer`; carries `adversarial_answer`; official
      metric is abstention: a prediction saying "no information available" / "not
      mentioned" scores 1, else 0)

Official primary metric (task_eval/evaluation.py): SQuAD-style token-overlap F1
(cats 2,3,4 score the whole phrase; cat 1 splits multi-hop sub-answers; cat 5 is the
abstention check above). The repo's eval.metrics.token_f1 is the same SQuAD F1, so the
honest mapping below routes the answerable cats through the runner's default EM+token_f1
scorer (category "T1") and keeps the true LoCoMo category in metadata.

Mapping decisions (documented for the audit's honesty):
  * Each LoCoMo conversation -> one persona (locomo_p01..p10). Each turn -> one Document.
  * Each ANSWERABLE QA (cats 1-4) -> one TaskInstance with category "T1" so the runner
    uses EM + token_f1 (== LoCoMo's official F1). We do NOT remap onto T2-claims/T4-FAMA:
    those use different scorers that would NOT match LoCoMo's published metric.
  * For cat 3 we store the first ';'-clause as gold.answer (official convention) and keep
    the full original answer in metadata.
  * ADVERSARIAL QA (cat 5) are written under adversarial/ (NOT a T*/ dir) so the runner's
    `T*/*.json` glob skips them — they need an abstention metric our extractive baselines
    cannot satisfy, so including them in an F1 aggregate would be dishonest. They are kept
    for completeness and a future generator-based run.

Usage:
    python -m eval.datasets.convert_locomo \
        --src external/locomo/data/locomo10.json \
        --out data/external/locomo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# LoCoMo category int -> (human label, harness T-code or "adversarial")
LOCOMO_CATEGORIES: dict[int, dict[str, str]] = {
    1: {"label": "multi_hop", "t_code": "T1"},
    2: {"label": "temporal", "t_code": "T1"},
    3: {"label": "open_domain", "t_code": "T1"},
    4: {"label": "single_hop", "t_code": "T1"},
    5: {"label": "adversarial", "t_code": "adversarial"},
}

_SOURCE_NOTE = (
    "REAL LoCoMo (snap-research/locomo, CC BY-NC 4.0); converted, not committed"
)


def _session_keys(conv: dict[str, Any]) -> list[str]:
    keys = [k for k in conv if k.startswith("session_") and k[len("session_"):].isdigit()]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def _turn_text(turn: dict[str, Any], speaker: str) -> str:
    text = turn.get("text", "") or ""
    cap = turn.get("blip_caption")
    if cap:
        text = f"{text} [shared image: {cap}]".strip()
    return f"{speaker}: {text}".strip()


def build_corpus(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a conversation into Document records (one per turn)."""
    conv = sample["conversation"]
    sample_id = sample["sample_id"]
    docs: list[dict[str, Any]] = []
    for skey in _session_keys(conv):
        ts = conv.get(f"{skey}_date_time", "")
        for turn in conv[skey]:
            dia_id = turn.get("dia_id")
            if not dia_id:
                continue
            speaker = turn.get("speaker", "")
            docs.append(
                {
                    "id": f"{sample_id}__{dia_id}",
                    "type": "note",
                    "timestamp": ts,  # LoCoMo session date string ("1:56 pm on 8 May, 2023")
                    "persona_id": None,  # filled by caller
                    "text": _turn_text(turn, speaker),
                    "metadata": {
                        "source": _SOURCE_NOTE,
                        "sample_id": sample_id,
                        "session": skey,
                        "dia_id": dia_id,
                        "speaker": speaker,
                    },
                }
            )
    return docs


def _evidence_doc_ids(sample_id: str, evidence: list[str]) -> list[str]:
    return [f"{sample_id}__{e}" for e in (evidence or []) if e]


def convert(src: Path, out: Path) -> dict[str, Any]:
    data = json.loads(src.read_text())
    out.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {"personas": 0, "documents": 0, "answerable": 0, "adversarial": 0}
    by_cat: dict[str, int] = {}

    for idx, sample in enumerate(data, start=1):
        persona_id = f"locomo_p{idx:02d}"
        sample_id = sample["sample_id"]
        corpus_rel = f"data/external/locomo/corpus/{persona_id}/"

        # --- corpus ---
        corpus_dir = out / "corpus" / persona_id
        corpus_dir.mkdir(parents=True, exist_ok=True)
        docs = build_corpus(sample)
        for d in docs:
            d["persona_id"] = persona_id
        (corpus_dir / "documents.jsonl").write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in docs) + "\n"
        )
        counts["personas"] += 1
        counts["documents"] += len(docs)

        # --- qa -> task instances ---
        for qi, qa in enumerate(sample.get("qa", []), start=1):
            cat = int(qa["category"])
            meta = LOCOMO_CATEGORIES[cat]
            t_code = meta["t_code"]
            by_cat[meta["label"]] = by_cat.get(meta["label"], 0) + 1

            if t_code == "adversarial":
                inst_id = f"{persona_id}-adv-{qi:04d}"
                rec = {
                    "source": _SOURCE_NOTE,
                    "instance_id": inst_id,
                    "category": "adversarial",
                    "difficulty": "adversarial",
                    "update_frequency": "UF1",
                    "reasoning_depth": "RD2",
                    "persona_id": persona_id,
                    "query": qa["question"],
                    "corpus_ref": corpus_rel,
                    "answer_bearing_doc_ids": _evidence_doc_ids(sample_id, qa.get("evidence", [])),
                    "gold": {
                        # No answerable gold: official metric is abstention.
                        "answer": "",
                        "answer_type": "abstention",
                        "adversarial_answer": qa.get("adversarial_answer", ""),
                    },
                    "metadata": {
                        "locomo_category": cat,
                        "locomo_category_label": meta["label"],
                        "sample_id": sample_id,
                        "scoring_note": "official metric = abstention (not F1); excluded from baseline F1 aggregate",
                    },
                }
                cat_dir = out / "adversarial"
                cat_dir.mkdir(parents=True, exist_ok=True)
                (cat_dir / f"{inst_id}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
                counts["adversarial"] += 1
                continue

            # answerable cats 1-4
            raw_answer = qa["answer"]
            gold_answer = str(raw_answer)
            answer_type = "numeric" if isinstance(raw_answer, int) else "text"
            scored_answer = gold_answer
            if cat == 3 and ";" in gold_answer:
                # official: score the first ';'-clause for open-domain/commonsense
                scored_answer = gold_answer.split(";")[0].strip()

            inst_id = f"{persona_id}-{meta['label']}-{qi:04d}"
            rec = {
                "source": _SOURCE_NOTE,
                "instance_id": inst_id,
                "category": t_code,  # T1 -> runner uses EM + token_f1 (== LoCoMo F1)
                "difficulty": meta["label"],
                "update_frequency": "UF1",
                "reasoning_depth": "RD0" if cat == 4 else "RD2",
                "persona_id": persona_id,
                "query": qa["question"],
                "corpus_ref": corpus_rel,
                "answer_bearing_doc_ids": _evidence_doc_ids(sample_id, qa.get("evidence", [])),
                "gold": {"answer": scored_answer, "answer_type": answer_type},
                "metadata": {
                    "locomo_category": cat,
                    "locomo_category_label": meta["label"],
                    "sample_id": sample_id,
                    "original_answer": gold_answer,
                    "evidence": qa.get("evidence", []),
                },
            }
            cat_dir = out / t_code
            cat_dir.mkdir(parents=True, exist_ok=True)
            (cat_dir / f"{inst_id}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
            counts["answerable"] += 1

    (out / "CONVERSION_STATS.json").write_text(
        json.dumps({"counts": counts, "by_locomo_category": by_cat}, indent=2)
    )
    return {"counts": counts, "by_locomo_category": by_cat}


def main() -> int:
    p = argparse.ArgumentParser(description="Convert real LoCoMo -> harness TaskInstance layout")
    p.add_argument("--src", default="external/locomo/data/locomo10.json")
    p.add_argument("--out", default="data/external/locomo")
    args = p.parse_args()
    stats = convert(Path(args.src), Path(args.out))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
