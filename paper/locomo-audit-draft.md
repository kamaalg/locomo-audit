# There Is No Single LoCoMo Score: A Standardized, Vendor-Audited Re-evaluation of LLM Memory Systems

**Kamal Gurbanov**
PersonalMem Project · Independent
`kamalgurbanov916@gmail.com`

> **DRAFT — archival paper, v0.2 (2026-05-31).** Track 1 (LoCoMo Audit). NOT LEGAL ADVICE.
> Status of each section is marked inline as **[DRAFTED]** or **[PENDING-DATA]**. All vendor
> result cells are marked **"PENDING — keyed pilot"** (free-tier 1-persona pilot imminent; full
> 5-seed run after). Built from `DISPUTE-FORENSICS.md`,
> `AUDIT-PROTOCOL.md`, `KEYED-RUN-SPEC.md`, `HARNESS-UPGRADE-REPORT.md`, `T1-DATA-REPORT.md`.
>
> **Provenance/confidence legend** (carried from the forensics): **[V]** verified from a
> primary source; **[P]** partially / secondary-verified; **[U]** unverified — primary source
> not located.
>
> **Reproducibility at a glance.** Public harness (`eval/runner.py`, `eval/metrics.py`,
> `eval/answer_gen.py`, `eval/judge.py`); pinned answer + judge model `claude-sonnet-4-5`;
> dataset `snap-research/locomo` @ `3eb6f2c`; ≥5 seeds per cell; deterministic `--tag`
> output filenames; budget-gated, resumable runs with a git-SHA + config + seed manifest.
> The keyless real-data floor (Table 3) reproduces today at \$0.00. See §10.

---

## Abstract  *[DRAFTED]*

Long-term conversational memory systems for large language models report wildly divergent
accuracies on LoCoMo, the de-facto benchmark for very-long-term dialog memory: Zep ~84% (later
corrected to 58.44%, then counter-claimed at 75.14%), Mem0 67.13% and subsequently ~92.5%, with
each vendor publicly disputing the others' methodology. We show that these numbers do not diverge
because any party is lying. **There is no single "LoCoMo score."** Each headline number is the
output of a *different protocol*, run by an *interested party on a competitor's configuration*,
and several widely-cited head-to-head comparisons (e.g. "Zep 63.8 vs Mem0 49.0") are
*cross-source splices* that were never produced under one harness. We make four contributions.
First, a **sourced forensic decomposition** of the dispute into four nameable methodological
forks — (A) denominator / adversarial-category handling, (B) who-configured-the-loser, (C)
judge- and answer-model drift, and (D) benchmark-identity confusion — each traced to a primary
URL. Second, a **provenance correction**: the famous "84%" does *not* appear in Zep's arXiv paper
(arXiv:2501.13956 reports only DMR and LongMemEval and contains no LoCoMo evaluation); it
originates in Zep's separate evaluation repository / blog, and we attribute it accordingly.
Third, **one frozen, fully-specified, auditable protocol** that neutralizes all four forks and is
**now live in the harness**: every system retrieves context, a single **pinned answer model**
(Claude) *generates* the answer from (question + retrieved context), and a single **pinned LLM
judge** (Claude) scores generated-answer-vs-gold correctness — the credible, vendor-comparable
**judge-accuracy J**. (Scoring the old extractive top-passage by token-F1 underrated real systems:
a correct verbose long-context answer scored 0.13 in a live pilot; the generate-then-judge pipeline
closes exactly that gap.) The protocol fixes Categories 1–4 only with consistent denominators,
vendor-sign-off configurations, one answer + one judge model, one benchmark per table, and ≥5
seeds, executed under a budget-gated, resumable, third-party-reproducible harness with a vendor
pre-disclosure / fair-play window.
Fourth, a **keyless real-data floor** computed on the real `snap-research/locomo` corpus at
\$0.00 — bm25 token-F1 0.0596 and tf-idf 0.0520 over N=1540 answerable questions — that exercises
the full pipeline on non-synthetic data and is explicitly labeled a *retrieval floor*, not a
system score. The generation-based vendor results (Mem0, Zep, Letta, long-context) execute the
same protocol and are reported in this draft as **pending the keyed run**. Our central
contribution is methodological infrastructure: a fair, citable, reproducible re-evaluation that
converts a vendor shouting match into an auditable measurement.

---

## 1. Introduction  *[DRAFTED]*

Personal AI assistants are judged increasingly on memory: the ability to recall, across months of
conversation, what a user said, when they said it, and how it relates to everything else they
said. The benchmark that has come to stand in for this capability is **LoCoMo** (Maharana et al.,
ACL 2024), ten multi-session conversations of roughly 300 turns and 9K tokens each, with
questions tagged into single-hop, multi-hop, open-domain, temporal, and adversarial categories.
Commercial memory vendors compete openly on their LoCoMo accuracy. The trouble is that the
reported accuracies are mutually irreconcilable.

The public record reads as a dispute. Zep's evaluation artifact reported roughly **84%**. Mem0
re-ran Zep and reported **58.44% ± 0.20**, attributing the gap to an inconsistent denominator.
Zep rebutted with **75.14% ± 0.17**, attributing *that* gap to Mem0 having mis-configured Zep's
adapter. Mem0's own paper reports **67.13%**, and a later Mem0 announcement reports **~92.5%**.
Independent systems land elsewhere again — Memobase 75.78%, MemMachine 91.69%. A reader trying to
answer the simple question "which memory system is best on LoCoMo?" finds no answer, only a
crossfire.

Our claim is that this crossfire is **not a disagreement about architecture** — it is a
disagreement about *measurement that no one fixed*. There is no single LoCoMo score because every
number above was produced under a different protocol, and in several cases by the *competitor* of
the system being scored. Once the protocols are named and held constant, the 84 / 58.44 / 75.14 /
92.5 spread resolves into **four methodological forks**, not four architecture verdicts.

We do four things. (1) We **forensically decompose** the dispute, tracing every number to a
primary source with an explicit confidence flag. (2) We **correct the provenance** of the most-
cited number, the "84%," which is widely but wrongly attributed to Zep's arXiv paper. (3) We
specify **one frozen, fair, fully-reproducible protocol** that neutralizes all four forks and run
it on the **real** LoCoMo data, with a pre-disclosure window so each vendor sees its own number
before publication. (4) We report a **keyless real-data baseline floor** now, at zero cost, and
present the vendor results table as **pending the budget-gated keyed run** that will fill it.

The paper's strength is the forensics plus a protocol that is **implemented and verified**, not
just specified: the generate-then-judge pipeline (system retrieves → pinned Claude generates →
pinned Claude judges → J) runs end-to-end in the public harness, validated cheaply on real LoCoMo
with bm25 retrieval under a tiny budget cap. The vendor J numbers are the empirical payoff once the
keyed run lands; until then every vendor cell in this draft is marked explicitly as pending.

---

## 2. Background & Related Work  *[DRAFTED]*

**LoCoMo** (Maharana et al., ACL 2024; `snap-research/locomo`; CC BY-NC 4.0; arXiv:2402.17753).
Ten multi-session conversations between two speakers, ~300 turns and ~9K tokens each, with
question–answer pairs tagged by category. The repository's official primary metric is **SQuAD
token-overlap F1**. Standard practice evaluates **Categories 1–4** (single-hop, multi-hop,
open-domain/commonsense, temporal) and **excludes Category 5 (adversarial)**, whose ground-truth
answers are not usable under the LLM-judge protocol. **[V]** (Mem0 paper, §eval).

**Mem0** (arXiv:2504.19413). A memory layer that extracts and consolidates salient facts. Its
paper reports LoCoMo under a **gpt-4o-mini** answer-and-judge regime, Categories 1–4 only, with
±std over multiple seeds — the most fully-specified vendor protocol in the dispute, and the one
this audit treats as the methodological reference point. Mem0 overall J = 67.13 ± 0.65; its graph
variant 65.71 ± 0.45; it re-runs Zep at 65.99 ± 0.16. **[V]**

**Zep** (arXiv:2501.13956, "A Temporal Knowledge Graph Architecture for Agent Memory"). A
temporal knowledge-graph memory. **Critically for this audit, Zep's paper evaluates only two
benchmarks — DMR (94.8% gpt-4-turbo / 98.2% gpt-4o-mini) and LongMemEval (63.8 gpt-4o-mini / 71.2
gpt-4o) — and contains no LoCoMo evaluation and no "84%" figure.** **[V]** The LoCoMo numbers
attributed to Zep come from a *separate* artifact (the `getzep/zep-papers`
`kg_architecture_agent_memory/locomo_eval` evaluation code and an accompanying blog post). We
return to this in §3.

**Memora / FAMA** (arXiv:2604.20006). Forgetting-aware memory accuracy. Cited here as adopted
prior art for the project's Track-2 metric work (`eval/metrics.fama`), not reinvented; it is a
methodological neighbor, not a system tabulated against LoCoMo here.

**Independent LoCoMo reports.** Memobase (75.78 overall; temporal 85.05; **gpt-4o** judge) and
MemMachine (91.69; **gpt-4.1-mini** answer model) are valuable precisely because they are *not*
parties to the Mem0/Zep dispute — and they land at yet different operating points, which §4.C
shows is largely a model-generation effect.

**Adjacent benchmarks we never co-tabulate: LongMemEval and DMR.** Both measure long-context /
memory ability but are *different datasets with different protocols*. A recurring source of public
confusion (§4.D) is quoting a LongMemEval or DMR number as though it were a LoCoMo number. We
treat one benchmark per table as a hard rule.

---

## 3. The Dispute  *[DRAFTED]*

We first establish, from primary sources, **who claims what, under which protocol, configured by
whom.** Lead finding (the provenance correction): the famous "Zep claimed 84% on LoCoMo" did
**not** come from Zep's arXiv paper. **[V]** The Zep paper (arXiv:2501.13956) evaluates Zep on
**only DMR and LongMemEval** and contains **no LoCoMo evaluation and no 84% figure at all.** The
84% originates from a separate Zep benchmarking artifact — the `getzep/zep-papers`
`.../locomo_eval` evaluation code and an accompanying blog — which is exactly what Mem0's GitHub
issue #5 corrects. This matters for fairness: critics who say "Zep's *paper* claimed 84% and was
wrong" conflate two artifacts. We attribute the 84% to the **eval repo / blog**, not the paper.

**Table 1. Claims-as-published (provenance-annotated; NOT single-harness results).**

| # | Claim | Claimed by | Benchmark | Answer model | Judge | Adversarial cat? | Configured by | Conf |
|---|---|---|---|---|---|---|---|---|
| 1 | **Zep ~84%** | Zep (eval repo / blog) | LoCoMo | unspecified | unspecified | **in numerator, excluded from denominator** (per Mem0) | Zep (self) | **[P]** |
| 2 | **Zep 58.44% ± 0.20** | Mem0 (re-running Zep), May 8 2025 | LoCoMo | gpt-4o-mini | Mem0 harness | excluded (corrected) | Mem0 | **[V]** |
| 3 | **Zep 65.99% ± 0.16** ("previous Zep algorithm") | Mem0 paper T1 | LoCoMo | gpt-4o-mini | gpt-4o-mini | excluded | Mem0 | **[V]** |
| 4 | **Zep 75.14% ± 0.17** | Zep rebuttal blog | LoCoMo | unspecified | unspecified | excluded | Zep (own fix) | **[V]** |
| 5 | **Mem0 67.13 ± 0.65** (overall); **Mem0^g 65.71 ± 0.45** | Mem0 paper T1 | LoCoMo | gpt-4o-mini | gpt-4o-mini | excluded | Mem0 (self) | **[V]** |
| 6 | **Mem0 92.5** (new algo, Apr 2026) | Mem0 | LoCoMo | unspecified | unspecified | inclusion not stated | Mem0 (self) | **[V]** number, **[U]** protocol |
| 7 | **Mem0 94.4** | Mem0 | LongMemEval | unspecified | unspecified | n/a | Mem0 (self) | **[V]** number, **[U]** protocol |
| 8 | **Zep 63.8 / 71.2** | Zep paper | LongMemEval | gpt-4o-mini / gpt-4o | LME | n/a | Zep (self) | **[V]** |
| 9 | **"Zep 63.8 vs Mem0 49.0 (GPT-4o)"** | secondary aggregators | LongMemEval | mixed | mixed | n/a | not one harness | **[P]/[U]** — splice, §4.D |
| 10 | **Zep 94.8 / MemGPT 93.4** | Zep paper | **DMR (not LoCoMo)** | gpt-4-turbo | DMR | n/a | Zep | **[V]** |
| — | **Memobase 75.78; temporal 85.05** | Memobase (independent) | LoCoMo | unspecified | **gpt-4o** | excluded | Memobase | **[V]** |
| — | **MemMachine 91.69** | MemMachine (independent) | LoCoMo | **gpt-4.1-mini** | unspecified | excluded | MemMachine | **[P]** |

*Comparable Mem0-paper per-category J cells* **[V]**: Mem0 overall 67.13; multi-hop 51.15 ± 0.31;
open-domain 72.93 ± 0.11; temporal 55.51 ± 0.34. Zep in the same table: single-hop 61.70 ± 0.32,
multi-hop 41.35 ± 0.48, open-domain 76.60 ± 0.13, temporal 49.31 ± 0.50. *(Caveat: an automated
extractor mislabeled the Mem0 67.13 row as "single-hop"; 67.13 is the **overall** J. Re-confirm
the exact cell layout against the PDF before final typesetting — honesty-ledger item, §8.)*

The pattern is now visible: every headline LoCoMo number is a party scoring **its competitor's**
configuration, under a protocol it chose, with a judge/answer model that is often unstated. That
is not a leaderboard; it is a set of incommensurable measurements.

---

## 4. Why the Numbers Diverge: Four Methodological Forks  *[DRAFTED]*

### Fork A — Denominator / adversarial-category handling  *(explains 84 → 58.44, the largest gap)*
**[V]** Per Mem0's issue #5, Zep's 84% counted Category-5 (adversarial) correct answers in the
**numerator** while the **denominator excluded** Category 5 — an inconsistent normalization that
inflates the score by ~25.56 points. Re-scoring Zep on a consistent Categories-1–4-only basis
yields **58.44% ± 0.20**. This is an **arithmetic/normalization fork, not an architecture fork.**
*Status:* Mem0's *description* of the inflation is verified; we have **not** independently re-
executed Zep's original script to confirm the exact arithmetic — that re-execution is precisely
the audit's no-spend reproduction target (§8 honesty ledger).

### Fork B — Who configured the loser  *(explains 58.44 ↔ 75.14, the same system)*
Every headline number is a party scoring its competitor's system. **[V]** Zep's rebuttal claims
Mem0's harness mis-set Zep three ways: (i) used a single-user graph but assigned **both** speakers
the user role; (ii) passed timestamps by **appending to message text** instead of Zep's dedicated
`created_at` field (which specifically hurts the temporal category); (iii) ran searches
**sequentially** rather than in parallel (this inflates *latency*, not accuracy). Fixing (i)–(ii),
Zep reports **75.14% ± 0.17**. So **58.44 and 75.14 are the same Zep system under hostile vs
friendly configuration** — neither is "the" Zep score. The fair mitigation is **vendor-blessed
adapters**: each vendor configures (or signs off on) its own adapter, and every config is
published verbatim.

### Fork C — Judge and answer-model drift  *(silently shifts every number)*
LoCoMo's J is **LLM-as-judge**, so the judge and answer model are part of the measurement. **[V]**
Mem0: *"All language model operations utilized gpt-4o-mini."* **[V]** Memobase judged with
**gpt-4o**. **[P]** MemMachine answered with **gpt-4.1-mini** and reports 91.69. **[U]** Mem0's
new 92.5 and Zep's 75.14 do **not** state their judge/answer model on the pages read. Both
stronger answer models and more lenient judges raise J. The ~92.5 / 91.69 cluster (newer, stronger
models) versus the ~58–75 cluster (gpt-4o-mini era) is therefore **largely a model-generation
effect, not purely an architecture win.** The mitigation is **now implemented, not merely
proposed**: the harness pins one answer model and one judge model across all systems (§5.2,
`eval/answer_gen.py` + `eval/judge.py`) and reports a second judge for robustness, so judge leniency
is visible rather than baked silently into a headline.

### Fork D — Benchmark-identity confusion  *(the "63.8 vs 49.0" splice)*
**[V]** The "Zep 63.8 vs Mem0 49.0 on LongMemEval (GPT-4o)" pairing that circulates in
aggregators is a **cross-source splice**: 63.8 is Zep's own paper number for **Zep + gpt-4o-mini**
(Zep + **gpt-4o** is **71.2**, where the full-context baseline 71.2 actually ties/edges Zep),
while the **49.0 for Mem0 comes from a third party (Vectorize), not a shared head-to-head
harness** — and the model labels don't even match. Separately, Zep's **DMR 94.8%** is routinely
quoted *as if* it were a LoCoMo number. **Any Zep-vs-Mem0 pairing not produced by a single run is
unverified.** **[U]** for the 49.0 specifically. The mitigation: **one benchmark per table**, and
splices appear only in a provenance-annotated claims table (Table 1), never as results.

**Table 2. Fork → mitigation traceability.**

| Fork | Symptom | Protocol kill-switch |
|---|---|---|
| **A** Denominator / adversarial | 84 → 58.44 (+25.56) | Cat 1–4 only; Cat-5 out of *both* numerator and denominator; per-category denominators published |
| **B** Who configured the loser | 58.44 ↔ 75.14 (same system) | Vendor-blessed / sign-off configs; all configs published verbatim |
| **C** Judge / answer-model drift | 58–75 vs 92.5 clusters | One pinned answer + judge model; second judge for robustness; unstated-model cells flagged |
| **D** Benchmark-identity confusion | "63.8 vs 49.0" splice; DMR quoted as LoCoMo | One benchmark per table; splices shown only as provenance-annotated claims |

---

## 5. A Standardized, Auditable Protocol (Methods)  *[DRAFTED]*

We re-run **all** memory systems on the **real** `snap-research/locomo` data (commit `3eb6f2c`,
CC BY-NC 4.0; 10 conversations, 5882 turn-Documents) under a **single frozen harness**
(`eval/runner.py` + `eval/metrics.py`). Each design choice maps to the fork it neutralizes.

**5.1 Frozen dataset (kills Fork A).** Source `snap-research/locomo` @ `3eb6f2c`, data file
`external/locomo/data/locomo10.json` (ships in-repo, no HF gate). **Scored set: Categories 1–4
only, N=1540** (single_hop 841, temporal 321, multi_hop 282, open_domain 96). **Category 5 (446
adversarial) is EXCLUDED from both numerator and denominator** of the headline and reported
separately as an abstention diagnostic. Per-category denominators are published verbatim; no
question may be counted correct in a numerator while excluded from its denominator.

**5.2 The generate-then-judge pipeline (the measurement, AS IMPLEMENTED).** This is the core of
the protocol and is now live in the harness (`eval/answer_gen.py`, `eval/judge.py`, wired into
`eval/runner.py` via `--answer-mode generate --judge llm`). For each (system, question, seed) cell:

  1. **Retrieve.** The system under test returns its retrieved context in a uniform field,
     `QueryResult.extra["retrieved_context"]` (a `list[str]`): top-k passages for `bm25`/`tfidf_rag`,
     surfaced memories for `mem0`, graph facts + node summaries for `zep`, and the stuffed persona
     corpus for the `longcontext` control. The retrieval step is the *only* part that differs across
     systems — everything downstream is held identical.
  2. **Generate (pinned answer model).** A single pinned answer model — **`claude-sonnet-4-5`**
     (temperature 0, `max_tokens` 128) — generates a concise answer grounded **only** in that
     retrieved context. The prompt forbids outside knowledge; if the context is empty the model is
     instructed to abstain ("I don't know") rather than hallucinate, so an empty-retrieval system
     honestly scores wrong. Identical generation capacity for every system is what removes the
     "stronger answer model" confound (Fork C).
  3. **Judge (pinned LLM judge).** A single pinned judge model — **`claude-sonnet-4-5`**
     (temperature 0) — scores the generated answer against gold as a binary correct/incorrect in the
     LoCoMo/Memora style: tolerant of paraphrase, formatting, extra words, and equivalent
     dates/units, strict about the actual fact. An empty prediction short-circuits to *incorrect*
     with **no paid call**. The cell's **J (judge-accuracy)** is the mean of these correct verdicts.

**J (judge-accuracy) is the headline, vendor-comparable metric** — it is the same quantity the
disputed vendor numbers (58.44 / 65.99 / 67.13 / 75.14 / 92.5) report, and it credits a correct-
but-verbose answer that token-F1 underrates (a live pilot scored a correct long-context answer
0.13 under token_f1; the judge marks it correct). Alongside J we still report **token_f1** (SQuAD
token-overlap F1, `eval/metrics.token_f1`, the LoCoMo-official metric — the one number comparable
across *all* systems including the no-generation keyless baselines) and **exact_match**. The
answer-gen and judge calls return **real** token usage and **real** `cost_usd` ($3/Mtok in,
$15/Mtok out) that count against the budget gate; no estimate is ever returned as if real. A
per-category breakdown (single / multi / open / temporal) is always reported; temporal is the
fork-sensitive category (Fork B-ii). **≥5 seeds** per (system, dataset) cell, reporting **mean ±
std** to be line-comparable with Mem0's ± bands. No metric is computed outside `eval/metrics.py`,
`eval/answer_gen.py`, and `eval/judge.py`.

*Verification (no vendor spend).* The full generate-then-judge pipeline was exercised end-to-end on
a tiny real-LoCoMo slice with `bm25` retrieval and a hard \$0.60 cap: it produced J and token_f1,
billed real cost \$0.0086, and the budget gate halted cleanly at 90% on a \$0.006 re-run. The
worked case that motivates the design: gold *"self-care is important"*, bm25-grounded generation
*"Melanie realized that self-care is really important,"* token_f1 = 0.6 but **judge = correct** —
the judge credits the fact the token metric penalizes (`HARNESS-UPGRADE-REPORT`).

**5.3 Pinned models (kills Fork C).** Answer model and primary judge: **`claude-sonnet-4-5`** (the
one key the PI holds; ADR 0006) — identical generation capacity for every system removes the
"stronger answer model" confound. A **robustness judge** (a second model; gpt-4o-mini if a second
key is authorized, to bridge to the gpt-4o-mini-era vendor numbers) is reported alongside.
**Disclosure rule:** every table states its answer and judge model; where a vendor's number used
an unstated model (Mem0's 92.5, Zep's 75.14, both `[U]`), that vendor cell is marked
"model-unstated (vendor-reported)" and is **not** presented as comparable to our pinned-model
cells. We never imply a model-effect gap is an architecture gap.

**5.4 Frozen system / config spec (kills Forks B and D).** Systems under test: keyless real-data
floor (`bm25`, `tfidf_rag`, no spend, runnable now, reported as a *retrieval floor*, explicitly
NOT comparable to generation-based vendor numbers); long-context control (`longcontext` /
`claude-sonnet-4-5`, the brute-force token-cost anchor); vendor systems (`mem0`, `zep`, `letta`,
PI-gated keyed run). **Vendor-config sign-off:** use each vendor's own published/blessed adapter
config as default; send each vendor its config + number + reproduce command in the pre-disclosure
window (§5.6); publish every adapter config verbatim, including Zep's three contested settings
(single- vs dual-user role, `created_at` vs appended timestamp, sequential vs parallel search —
the last affecting latency, not accuracy, and reported as such). **One benchmark per table:** a
LoCoMo cell never shares a table with LongMemEval or DMR; the "63.8 vs 49.0" splice is never
reproduced as a head-to-head.

**5.5 Category-5 handling.** Excluded from the headline (Fork A); systems that emit abstentions
are scored on Cat-5 with an abstention / over-answer diagnostic in its own table, never folded
into the F1/J headline.

**5.6 Pre-disclosure / fair-play protocol (the credibility step).** Before any public release: (1)
freeze the protocol and run it; (2) notify each vendor (Mem0, Zep, Letta) with their adapter
config, their number(s) under this protocol, the exact reproduce command, and the git SHA +
dataset commit; (3) open a **comment window** (suggested 10 business days) for the vendor to flag a
misconfiguration or supply a blessed config — reproducible corrections are adopted and credited
with a published changelog; (4) publish the protocol, all configs, all seeds, the manifest, and
per-cell results so a third party can re-run any single number. Unverified honesty-ledger items
(§8) are listed as such, never silently dropped.

**5.7 Reproducibility contract.** Every run launches from this protocol + a config; every cell
carries a seed; the runner writes a manifest (git SHA, configs, seeds, dataset commit) to
`results/<tag>.json`; runs are **budget-gated** (hard USD ceiling, halt at 90%) and **resumable**
(same `--tag` continues without re-spending completed cells); raw CC-BY-NC artifacts stay under
gitignored `results/` and `data/external/` — only code, configs, and aggregate numbers are
committed.

---

## 6. Experimental Setup  *[DRAFTED]*

Harness: `eval/runner.py` (budget-gated, resumable, manifest-emitting), driving the
generate-then-judge pipeline of `eval/answer_gen.py` + `eval/judge.py` under
`--answer-mode generate --judge llm` (the audit setting; `--answer-mode extractive --judge token_f1`
remains the back-compatible default for the keyless floor). Metrics: `eval/metrics.py` (single
source of truth for *how* token_f1/exact_match are computed; the protocol owns *which* set they are
computed over; J is owned by `eval/judge.py`). The runner records the pinned `answer_model`,
`judge_model`, `answer_mode`, `judge`, and `budget_usd` under `manifest.audit`, and aggregates
**`judge_accuracy`** (J) alongside token_f1. Dataset commit `3eb6f2c`; data file
`external/locomo/data/locomo10.json`; corpus 5882 turn-Documents at
`corpus/locomo_p01..p10/documents.jsonl`. Conversion via `eval/datasets/convert_locomo.py` →
committed-shape `TaskInstance` + corpus layout; `load_locomo()` auto-converts on first call. Pinned
answer + judge model `claude-sonnet-4-5`; ≥5 seeds per cell; a Sonnet-class robustness judge
reported alongside (per §5.3); cost envelope and exact reproduce command per `KEYED-RUN-SPEC`
(de-risk pass at 1 seed first, then the full 5-seed run under a \$6,000 ceiling). License
compliance: raw CC-BY-NC data is not redistributed (clone + commit-pin instructions are provided
instead).

---

## 7. Results  *[PENDING-DATA for all vendor cells]*

**7.1 Keyless real-data baseline floor — DONE, \$0.00.** We loaded the real LoCoMo corpus and ran
`bm25` and `tfidf_rag` over the N=1540 answerable (Cat 1–4) questions: **3080 cells scored, cost
\$0.00.** These extractive baselines return the best query-overlapping **sentence span** of the
top-ranked conversation turn with **no LLM generation**; LoCoMo gold answers are short generated
phrases, so **EM is exactly 0** and token-F1 is a low partial-overlap **retrieval floor** —
**NOT comparable** to Mem0/Zep's generation-based accuracies (~58–92%). Its value: a *real-data*
(non-synthetic), no-spend baseline that exercises the full pipeline end-to-end and directly
answers the "only synthetic data" critique.

**Table 3. Keyless retrieval floor (real LoCoMo, Cat 1–4, N=1540, \$0.00, seed=1).**

| System | EM | token_f1 (LoCoMo F1) | single_hop | open_domain | multi_hop | temporal |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.0000 | **0.0596** | 0.0898 | 0.0244 | 0.0291 | 0.0179 |
| tfidf_rag | 0.0000 | **0.0520** | 0.0783 | 0.0279 | 0.0242 | 0.0146 |
| combined | 0.0000 | 0.0558 | — | — | — | — |

Per-category N: single_hop 841, temporal 321, multi_hop 282, open_domain 96. **These are a
retrieval floor, not system scores.**

**7.2 Vendor + long-context results — PENDING the keyed pilot, then the full keyed run.** The main
vendor comparison is structured around **J (judge-accuracy)**, the credible vendor-comparable metric
produced by the generate-then-judge pipeline of §5.2; token_f1 is carried as the cross-system floor
column. All cells execute the identical frozen protocol (Cat 1–4; `claude-sonnet-4-5` answer +
primary judge; Sonnet-class robustness judge; ≥5 seeds, mean ± std). **Every vendor and
long-context cell is marked "PENDING — keyed pilot"** and is filled in two stages: an **imminent
free-tier, single-persona (p01) pilot** that de-risks the adapters and surfaces config breakage at
~1 seed, then the **full 5-seed keyed run** under the budget-gated ceiling (`KEYED-RUN-SPEC`). No
number below is invented; pending means pending.

**Table 4. Standardized single-harness results (one benchmark = LoCoMo; Cat 1–4; J primary, token_f1 floor).**

| System | Config source | **J (sonnet-4-5)** | J (robustness judge) | token_f1 | single | multi | open | temporal | Status |
|---|---|---|---|---|---|---|---|---|---|
| bm25 (keyless floor) | — | n/a (no gen) | n/a | **0.0596** | 0.0898 | 0.0291 | 0.0244 | 0.0179 | **DONE (\$0)** |
| tfidf_rag (keyless floor) | — | n/a (no gen) | n/a | **0.0520** | 0.0783 | 0.0242 | 0.0279 | 0.0146 | **DONE (\$0)** |
| longcontext (sonnet-4-5) | brute-force | PENDING — keyed pilot | PENDING — keyed pilot | PENDING — keyed pilot | — | — | — | — | **PENDING** |
| mem0 | vendor-blessed | PENDING — keyed pilot | PENDING — keyed pilot | PENDING — keyed pilot | — | — | — | — | **PENDING** |
| zep | vendor-blessed | PENDING — keyed pilot | PENDING — keyed pilot | PENDING — keyed pilot | — | — | — | — | **PENDING** |
| letta | vendor-blessed | PENDING — keyed pilot | PENDING — keyed pilot | PENDING — keyed pilot | — | — | — | — | **PENDING** |

*The keyless floor rows carry token_f1 only (`J = n/a`): they perform no generation, so there is no
generated answer to judge. They are a real-data retrieval floor, never comparable to the
generation-based J cells.* The provenance-annotated "claims-as-published" comparison (84 / 58.44 /
65.99 / 67.13 / 75.14 / 92.5 / 94.4, each with model + judge + config + URL + [V/P/U]) is Table 1 in
§3 and is explicitly **NOT** a single-harness result; it must never be read as a column of Table 4.

**Table 5 (PENDING — keyed pilot). Category-5 abstention diagnostic.** Reported separately from the
headline (Fork A): for each system, the abstention / over-answer rate on the 446 adversarial
questions, never folded into J or token_f1. Filled by the same keyed run.

---

## 8. Threats to Validity / Limitations  *[DRAFTED]*

- **Small dataset → wide variance.** LoCoMo is 10 conversations; we report ±std over ≥5 seeds.
- **Free-tier scale of the first vendor numbers.** The imminent pilot is a **single-persona (p01),
  ~1-seed free-tier run** whose purpose is adapter de-risking, not a leaderboard; it must be read as
  a smoke test, and the headline J cells are reported only after the **full 5-seed, 10-persona keyed
  run** lands. We label every preliminary cell accordingly and never promote a 1-persona pilot
  number to a paper claim.
- **Judge-model dependence.** J is LLM-as-judge, so the judge model is *part of the measurement*
  (this is Fork C, the very confound the audit names). Pinning one judge makes our numbers
  internally consistent but not numerically identical to the gpt-4o-mini-era vendor cells; a
  single judge can also encode its own leniency. Mitigations: a second **robustness judge** with
  both J values reported per cell, and an explicit "model-unstated (vendor-reported)" flag on any
  vendor cell whose judge/answer model is unknown (Mem0's 92.5, Zep's 75.14) so it is never silently
  treated as comparable. **We never present a model-effect gap as an architecture gap.**
- **Single-provider answer + judge.** Both the answer model and the primary judge are Anthropic
  `claude-sonnet-4-5` — the one key the PI holds (ADR 0006). This is a deliberate confound-control
  (identical generation capacity for every system) but it is also a *provider monoculture*: a
  systematic Claude-as-judge bias would shift all cells together. The cross-provider robustness judge
  (gpt-4o-mini, if a second key is authorized — which also bridges to the historical vendor regime)
  is the intended check; until then, single-provider dependence is a stated limitation, not a hidden
  one.
- **Vendor-config dependence.** Vendor adapters depend on vendor-blessed configs; a vendor
  declining sign-off is reported as such, with the default published config used and labeled.
- **Synthetic-free scope of this draft.** The only computed numbers here are the *real-data*
  keyless floor; everything generation-based is pending the keyed run — this draft contains no
  synthetic-data results.
- **What we did NOT verify (honesty ledger, from forensics §6):**
  - **[U]** judge/answer model behind Mem0's **92.5** and Zep's **75.14** — not stated on pages
    read; to be raised with vendors in the pre-disclosure window.
  - **[P]** the exact 84 → 58.44 inflation arithmetic — verified *as described* by Mem0, **not**
    independently re-executed (the audit's reproduction target).
  - **[U]** the Mem0 **49.0** LongMemEval figure — found only in secondary aggregators; locate a
    primary run or drop it.
  - Mem0 paper **Table-1 cell layout** — re-read from the PDF (the auto-extractor mislabeled the
    67.13 row as single-hop; it is the overall J).
  - Whether Zep replied *inside* issue #5 or only on its blog — comments not captured; re-read.

---

## 9. Ethics & Fair-Play  *[DRAFTED]*

This audit's entire value is being **correct and fair**. (1) **Pre-disclosure:** no vendor learns
its number first from a public post — each receives its config, number, reproduce command, git SHA,
and dataset commit, with a comment window to flag misconfiguration or supply a blessed config;
reproducible corrections are adopted and credited with a published changelog. (2) **Fairness cuts
both ways:** we correct the record *for* Zep (the "84%" is not in its paper) as readily as we
surface inconsistencies; an unstated-model number is flagged, never silently treated as comparable.
(3) **Data ethics:** LoCoMo is CC BY-NC 4.0; we do not redistribute raw data, providing clone +
commit-pin instructions instead. (4) **No legal claims:** this is a measurement audit, **NOT LEGAL
ADVICE**, and makes no assertion of bad faith by any party — the thesis is explicitly that the
spread is a *measurement artifact*, not deception.

---

## 10. Reproducibility Statement & Release  *[DRAFTED]*

**Reproducibility statement.** The audit is reproducible by construction. (1) **Public harness:**
`eval/runner.py` (budget-gated, resumable, manifest-emitting), the generate-then-judge pipeline
`eval/answer_gen.py` + `eval/judge.py`, the scorers `eval/metrics.py`, and every system adapter
(`systems/{bm25,tfidf_rag,longcontext,mem0_adapter,zep_adapter,letta_adapter}.py`) are released
open-source. (2) **Pinned models:** answer + primary judge `claude-sonnet-4-5`, recorded in
`manifest.audit` of each result file, so there is no silent model drift. (3) **Seeds:** every cell
carries a seed; we run ≥5 and report mean ± std. (4) **Frozen data:** `snap-research/locomo` @
`3eb6f2c` (CC BY-NC 4.0); we do **not** redistribute raw data but ship clone + commit-pin
instructions. (5) **Deterministic outputs:** `--tag` fixes the result filename; re-invoking the same
tag resumes without re-spending completed cells. (6) **Exact commands:** the keyless floor (Table 3)
reproduces today at \$0.00 (`KEYED-RUN-SPEC §3c`); the keyed pilot and full run have published
commands (`§3a`, `§3b`). Any single cell is independently re-runnable by a third party. The
companion open-source harness is the durable, citable artifact.

---

## 11. Conclusion  *[DRAFTED]*

The LoCoMo dispute is a **measurement artifact, not an architecture verdict.** There is no single
LoCoMo score because every headline number was produced under a different protocol, often by the
competitor of the system being scored, and several famous comparisons are cross-source splices.
Named and held constant, the 84 / 58.44 / 75.14 / 92.5 spread resolves into four methodological
forks. A single frozen, vendor-sign-off, fully-reproducible protocol — run on the real data, with
a pre-disclosure fair-play window — is the durable, citable contribution. The keyless real-data
floor is reported now; the standardized vendor numbers are the empirical payoff once the keyed run
lands.

---

## Venue Plan  *[DRAFTED]*

- **Now (forensics + protocol + keyless floor + implemented pipeline):** arXiv preprint —
  establishes the citable artifact and the O-1 scholarly-article evidence and timestamps the
  provenance correction (the "84%" is not in Zep's paper) and the frozen protocol. The forensics and
  the live, reproducible generate-then-judge harness already stand on their own; the keyed numbers
  upgrade the empirical payoff but are not a precondition for the preprint.
- **After the keyed run (full vendor J table + pre-disclosure window closed):** **ACL/EMNLP 2026**
  resource/benchmark or short-paper track, or **NeurIPS 2026 Datasets & Benchmarks / Evaluation &
  Disclosure**; **fallback NeurIPS E&D 2027.** The pre-disclosure window and any adopted vendor
  corrections are timed to precede camera-ready.
- **Companion:** open-source harness + project page — the moat and the most-citable artifact, and
  the strongest single piece of O-1 "original contribution of major significance" evidence.

---

## References (real identifiers; verify formatting at typeset)  *[DRAFTED]*

- Maharana et al., *Evaluating Very Long-Term Conversational Memory of LLM Agents* (LoCoMo), ACL
  2024. arXiv:2402.17753. Repo: `snap-research/locomo` (CC BY-NC 4.0), commit `3eb6f2c`.
- Mem0 (memory layer), arXiv:2504.19413. HTML: arxiv.org/html/2504.19413v1.
- Zep, *A Temporal Knowledge Graph Architecture for Agent Memory*, arXiv:2501.13956 (DMR +
  LongMemEval only; **no LoCoMo**). HTML: arxiv.org/html/2501.13956v1.
- Memora / FAMA, arXiv:2604.20006 (adopted for `eval/metrics.fama`).
- Mem0 issue correcting 84 → 58.44: github.com/getzep/zep-papers/issues/5.
- Zep LoCoMo eval code (origin of 84%): github.com/getzep/zep-papers/tree/main/kg_architecture_agent_memory/locomo_eval.
- Zep rebuttal (75.14%): blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/.
- Mem0 new algo (92.5 / 94.4): mem0.ai/research.
- Memobase independent LoCoMo (75.78; temporal 85.05; gpt-4o judge): github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md.
- MemMachine independent LoCoMo (91.69; gpt-4.1-mini): memmachine.ai/blog/2025/12/memmachine-v0.2-delivers-top-scores-and-efficiency-on-locomo-benchmark/.
- Secondary aggregators (the 63.8/49.0 splice, treated as unverified): vectorize.io/articles/mem0-vs-zep ; atlan.com/know/zep-vs-mem0/.
- LongMemEval, DMR — adjacent benchmarks, never co-tabulated with LoCoMo.
