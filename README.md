# The LoCoMo Audit — An Independent, Reproducible Evaluation of AI-Memory Systems

> An independent, reproducible harness for evaluating LLM long-term-memory systems — and a vendor-audited re-evaluation of LoCoMo showing there is no single "LoCoMo score."

[![License: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/synthetic_data-CC--BY--4.0-green.svg)](#license)

**Status:** Preprint + open harness. Solo, independent project (no institutional affiliation). The forensic decomposition and the keyless $0 reproducible floor are done and reproducible today; the keyed vendor-comparison table is pending a budget-gated paid run (see [What's pending](#whats-pending-the-keyed-run)).

---

## What this is

This repository is two things:

1. **An independent, reproducible evaluation harness for AI-memory systems.** Any memory system — vector RAG, BM25, a knowledge-graph memory, a commercial memory API, or a long-context model — implements one small Python interface and is then scored by exactly the same pipeline: the system retrieves context, a single **pinned answer model** generates the answer from `(question + retrieved context)`, and a single **pinned LLM judge** scores the generated answer against the gold answer. Runs are budget-gated, resumable, and write a git-SHA + config + seed reproducibility manifest.

2. **The LoCoMo Audit** — a sourced forensic re-evaluation of the public dispute over LoCoMo, the de-facto benchmark for long-term conversational memory.

## Headline finding: there is no single "LoCoMo score"

Commercial memory vendors report wildly divergent LoCoMo accuracies — Zep ~84% (later corrected to 58.44%, then counter-claimed at 75.14%), Mem0 67.13% and later ~92.5%, with each vendor publicly disputing the others' methodology. **These numbers do not diverge because anyone is lying.** Each headline is the output of a *different protocol*, often run by an *interested party on a competitor's configuration*. The audit:

- **Decomposes the dispute** into four nameable methodological forks (denominator/adversarial handling; who-configured-the-loser; judge- and answer-model drift; benchmark-identity confusion), each traced to a primary source.
- **Corrects a provenance error**: the famous "84%" does *not* appear in Zep's arXiv paper (arXiv:2501.13956 evaluates only DMR and LongMemEval — no LoCoMo). It originates in a separate Zep evaluation artifact and is attributed accordingly.
- **Specifies one frozen, fair, fully-reproducible protocol** that neutralizes all four forks (consistent denominator, vendor-sign-off configs, one answer model + one judge model, one benchmark per table, ≥5 seeds), implemented and verified end-to-end in this harness.

### The $0 keyless reproducible floor

To prove the pipeline runs on **real** (non-synthetic) LoCoMo data with no API keys and no spend, we report a **keyless retrieval floor** over the real `snap-research/locomo` corpus:

| System | Metric | Score | N | Cost |
|---|---|---|---|---|
| BM25 | token-F1 | 0.0596 | 1540 | $0.00 |
| TF-IDF | token-F1 | 0.0520 | 1540 | $0.00 |

This is explicitly a **retrieval floor**, not a system score — it exercises the full pipeline on real data at zero cost so anyone can reproduce the harness's plumbing before spending a cent. The keyed vendor comparison (Mem0, Zep, Letta, long-context, under the generate-then-judge protocol) is **pending the budget-gated paid run**.

---

## Evaluate your memory system in one command

Add your system as a single file `systems/<yourname>.py` exposing two symbols — `SYSTEM_NAME: str` and `build_system(config) -> MemorySystem`. It is auto-discovered; you touch no shared registry, no YAML, no `__init__`.

Your class implements this interface (`systems/base.py`):

```python
from systems.base import MemorySystem, Document, QueryResult

class MyMemory(MemorySystem):
    name = "my_memory"

    def ingest(self, documents: Iterable[Document]) -> None: ...
    def update(self, doc_id: str, new_text: str) -> None: ...   # honest update cost
    def delete(self, doc_id: str) -> None: ...                  # forgetting / privacy
    def query(self, query: str, persona_id: str) -> QueryResult: ...
    def stats(self) -> dict: ...
    def reset(self) -> None: ...

SYSTEM_NAME = "my_memory"
def build_system(config: dict) -> MemorySystem:
    return MyMemory(config)
```

`QueryResult` requires `answer`, `supporting_doc_ids`, `latency_ms`, and `tokens_used` (cost/latency are reported metrics, not afterthoughts). If your SDK or key is missing, `build_system` raises `SystemUnavailable` and the runner logs a SKIP and continues — a missing system is never silently scored as a number.

Then run it:

```bash
python -m eval.runner \
    --systems-list my_memory \
    --tasks   data/external/locomo \
    --seeds   1 \
    --output  results/ \
    --tag     my_memory_run \
    --budget-usd 0.01
```

Inspect what's available without running anything:

```bash
python -m eval.runner --list-systems   # discovered adapters
python -m eval.runner --list-tasks     # benchmark task counts by category/difficulty
```

The runner **refuses to start without `--budget-usd`** (a hard ceiling) and halts at 90% of it. `--tag` gives a deterministic output filename so results are diff-able and reproducible.

---

## Reproduce our numbers

The keyless floor reproduces today at **$0.00**, no API keys required:

```bash
# 1. install (Python 3.12+; uv recommended)
uv sync

# 2. obtain the real LoCoMo data locally (CC BY-NC 4.0 — NOT redistributed here;
#    see data/external/README for the one-line fetch from snap-research/locomo)
#    -> populates data/external/locomo/

# 3. run the keyless retrieval floor on real LoCoMo
python -m eval.runner \
    --systems-list bm25,tfidf_rag \
    --tasks   data/external/locomo \
    --seeds   1 \
    --output  results/real_locomo_baselines \
    --tag     real_locomo_baselines \
    --budget-usd 0.01
```

Expected: bm25 token-F1 ≈ 0.0596 and tf-idf ≈ 0.0520 over N=1540 answerable questions, at $0.00. The run writes a manifest (git SHA, configs, seeds, dataset version) alongside the results.

For the keyed generate-then-judge protocol, set `ANTHROPIC_API_KEY` in `.env` (copy `.env.example`), choose `--answer-mode generate`, pin `--answer-model` and `--judge-model`, and raise `--budget-usd` to your ceiling. See `paper/` (the audit draft) §10 for the full keyed protocol.

---

## What's pending (the keyed run)

Honest current state:

- **Done & reproducible now:** the harness (`eval/runner.py`, `eval/metrics.py`, `eval/answer_gen.py`, `eval/judge.py`), the keyless $0 real-LoCoMo floor, the forensic decomposition, and the frozen protocol (implemented and verified end-to-end on real data with BM25 retrieval under a tiny budget cap).
- **Pending the budget-gated keyed run (~$250–400):** every vendor cell (Mem0, Zep, Letta, long-context) under the generate-then-judge protocol. These are marked **"pending — keyed run"** in the paper and are not reported as numbers until the run lands.
- **Not claimed:** this is a preprint and an open harness, not a "widely adopted standard." No adoption, citation, or press claims are made.

---

## License

This project uses a **split license**:

- **Code** (harness, metrics, judge, adapters, scripts): **Apache-2.0** — see `LICENSE`.
- **Original synthetic benchmark data and the audit's derived artifacts**: **CC BY 4.0**.
- **Third-party datasets are NOT redistributed here.** Real LoCoMo (`snap-research/locomo`) is **CC BY-NC 4.0**; you fetch it yourself and it stays local. See `data/LICENSES.md` for every source and its terms.

## How to cite

Citation metadata is in [`CITATION.cff`](CITATION.cff). Until the arXiv DOI is assigned:

```bibtex
@misc{gurbanov2026locomoaudit,
  title  = {There Is No Single LoCoMo Score: A Standardized, Vendor-Audited
            Re-evaluation of LLM Memory Systems},
  author = {Gurbanov, Kamal},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/<owner>/<repo>}
}
```

(Replace the URL/DOI placeholders once the public repo and arXiv preprint exist.)

## Acknowledgments

LoCoMo: Maharana et al., ACL 2024 (`snap-research/locomo`, CC BY-NC 4.0, arXiv:2402.17753). The audit cites and attributes every vendor number to its primary source; see the paper draft for the full provenance table.
