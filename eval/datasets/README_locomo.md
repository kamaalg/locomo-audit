# LoCoMo data + loader (Track 1)

**LoCoMo is CC BY-NC 4.0** (snap-research/locomo). Neither the raw dataset nor the
converted instances are committed. `external/` and `data/external/` are gitignored.

## Acquire + convert (no keys, no spend)

```bash
# 1. clone the official repo OUTSIDE the committed tree (ships the data in-repo)
git clone --depth 1 https://github.com/snap-research/locomo.git external/locomo
#    -> data file: external/locomo/data/locomo10.json

# 2. convert REAL LoCoMo -> the harness TaskInstance + corpus layout
python -m eval.datasets.convert_locomo \
    --src external/locomo/data/locomo10.json \
    --out data/external/locomo
```

`load_locomo()` in `loaders.py` auto-bootstraps: if `data/external/locomo/T1/`
is missing but the raw clone exists, it converts on first call.

## Source dataset structure (`data/locomo10.json`)

- **10 samples**, each one multi-session conversation between two speakers.
  - `conversation`: `speaker_a`, `speaker_b`, and per session
    `session_<n>_date_time` + `session_<n>` (a list of turns
    `{speaker, dia_id "D<sess>:<turn>", text [, img_url, blip_caption, query]}`).
  - `qa`: `{question, answer, evidence:[dia_id...], category}`.
  - Also `observation`, `session_summary`, `event_summary` (RAG DBs / the event
    task — not used by this QA baseline).

### QA categories (confirmed against `task_eval/evaluation.py`)

| cat | meaning | count | official scoring |
|----:|---------|------:|------------------|
| 1 | multi-hop | 282 | F1 with multi-hop sub-answer split |
| 2 | temporal | 321 | SQuAD token F1 |
| 3 | open-domain / commonsense | 96 | F1 on first `;`-clause of the answer |
| 4 | single-hop | 841 | SQuAD token F1 |
| 5 | adversarial / unanswerable | 446 | abstention (1 if "no information available"/"not mentioned", else 0); has `adversarial_answer`, no `answer` |

Primary metric in the paper = **SQuAD-style token-overlap F1**, which is exactly
`eval.metrics.token_f1`.

## Converted layout (`data/external/locomo/`, uncommitted)

- `corpus/locomo_p01..p10/documents.jsonl` — one Document per turn
  (`id = "<sample_id>__<dia_id>"`, `text = "Speaker: utterance"`, image turns
  append `[shared image: <blip_caption>]`, `timestamp` = session date string).
- `T1/*.json` — the **1540 answerable** QA (cats 1-4) as TaskInstances. All use
  `category: "T1"` so the runner scores with EM + `token_f1` (== LoCoMo F1); the
  true LoCoMo category is in `metadata.locomo_category(_label)` and surfaced as
  `difficulty`. Cat-3 gold is the first `;`-clause (official convention).
- `adversarial/*.json` — the **446 adversarial** QA. Deliberately NOT under a
  `T*/` dir, so the runner's `T*/*.json` glob skips them: their official metric is
  abstention, which the keyless extractive baselines cannot satisfy, so scoring
  them on F1 would be dishonest. Kept for a future generator-based run.

## Honesty caveats

- The keyless baselines (`bm25`, `tfidf_rag`) answer **extractively** — they return
  the best-overlapping sentence span of the top-ranked turn, with no LLM
  generation. LoCoMo gold answers are short generated phrases, so **EM is ~0** and
  token-F1 is low: this is a *retrieval-floor* baseline, not a system score. It is
  honest and useful as a no-spend floor and a partial answer to the "only
  synthetic" critique — but it is NOT comparable to Mem0/Zep's generation-based
  LoCoMo numbers.
