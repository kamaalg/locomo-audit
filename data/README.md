# PersonalMem Data

This directory holds the PersonalMem benchmark corpus, task instances, and the ethics/provenance
documents. The **pilot** (3 synthetic personas) is present; full generation (Step B, ~50–100
personas) is gated separately on PI approval of the pilot cost estimate.

## Layout

```
data/
├── README.md                     # this file
├── CONSENT.md                    # "no consented data this round" (ADR 0003)
├── LICENSES.md                   # source + license plan (synthetic + public-licensed)
├── REDACTION.md                  # PII redaction pipeline (public tier, Step B)
├── SYNTHETIC_GENERATION.md       # prompts, models, seeds, corpus→tasks procedure
├── T7_INFERENCE_ALLOWLIST.md     # FROZEN: income/commute/household; protected classes excluded
├── T8_SENSITIVE_TAXONOMY.md      # FIXED taxonomy — PI REVIEW REQUIRED before Step B
├── statistics.json               # pilot stats (tokens by source, instances by category×difficulty)
├── synthetic/
│   ├── personas/persona-00k.json # persona specs incl. T7 ground-truth + T8 sensitive items
│   ├── corpus/persona-00k/        # one JSON per document (journal/email/note/calendar)
│   │   └── documents.jsonl        # all docs for the persona, one per line
│   └── tasks/T1../T8/             # task instances (benchmark-spec schema)
├── public/                       # public-licensed tier (Step B; empty in pilot)
└── consented/                    # empty by policy (no consented data this round)
```

`persona_id` values are `p001`, `p002`, `p003`. A persona's spec file is
`personas/persona-001.json`; its corpus directory is `corpus/persona-001/`. Each task instance's
`corpus_ref` points at the persona's corpus directory.

## Loading

```python
import json, pathlib
DATA = pathlib.Path("data/synthetic")

def load_corpus(persona_id):          # e.g. "p001"
    n = persona_id[1:]                 # "001"
    p = DATA / "corpus" / f"persona-{n}" / "documents.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

def load_tasks(category=None):         # category e.g. "T4" or None for all
    root = DATA / "tasks"
    cats = [category] if category else sorted(d.name for d in root.iterdir() if d.is_dir())
    out = []
    for c in cats:
        for f in sorted((root / c).glob("*.json")):
            out.append(json.loads(f.read_text()))
    return out
```

Each document conforms to the `systems.base.Document` contract:
`{id, text, timestamp (ISO 8601), type ∈ {journal,email,note,calendar}, persona_id, metadata}`.
Each task conforms to the common wrapper in `docs/benchmark-spec.md` §3 plus its category schema in
`docs/benchmark-examples/Tn/`.

## Splits

- Pilot: no train/test split (validation set for the pipeline).
- Step B: frozen `test.json` + held-out `dev.json` (10%) under `data/splits/` (benchmark-spec §0.4).

## Licensing

- Data: **CC-BY-4.0**. Code: **MIT**. See `LICENSES.md`.
- Privacy-preserving release: only synthetic-split examples are published verbatim; real/licensed
  subsets (Step B) release aggregate metrics + derived labels only. T8 sensitive query–answer pairs
  are never published verbatim.

## Ethics

Synthetic + public-licensed only (ADR 0003). T7 restricted to a frozen non-protected allowlist with
an evidence-ablation control. T8 sensitive content is synthetic, from a fixed PI-reviewed taxonomy,
scored programmatically. See the four ethics docs above.
