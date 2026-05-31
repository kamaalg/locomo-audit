# eval/prompts — Stage-2 alternate judge / answer templates (TEXT ONLY, NOT RUN HERE)

These are the controlled **judge-prompt** and **answer-prompt** wordings for Paper 2's
PAID Stage-2 sweep (the judge-model / judge-prompt / answer-prompt axes). They are
**additive artifacts**: Stage 1 makes **zero** LLM calls, so nothing in this folder is
invoked by `eval/sweep.py`. The orchestrator wires these into the existing
`eval/judge.py` / `eval/answer_gen.py` *at Stage 2* (PI-gated, paid).

Drop-in contract (so Stage 2 needs no edit to the pinned modules):

- **Judge templates** expose a `judge_id`, an `INSTRUCTIONS` block, and a `build(question,
  predicted, gold)` shape that mirrors `eval/judge.py::_build_prompt` (same QUESTION /
  GOLD ANSWER / PREDICTED ANSWER fields, same single-line JSON verdict
  `{"correct": true|false, "reason": "..."}`). Only the *instruction wording* changes —
  that is the controlled judge-prompt axis (RQ1) and feeds the rank-flip result (RQ2).
- **Answer templates** expose an `answer_prompt_id`, an `INSTRUCTIONS` block, and a
  `build(question, retrieved_context)` shape that mirrors
  `eval/answer_gen.py::_build_prompt` (same RETRIEVED CONTEXT / QUESTION / INSTRUCTIONS /
  ANSWER sections, same ONLY-the-context grounding rule). Only the *style/verbosity*
  instruction changes — the answer-prompt axis (Regime B, re-generation spend).

Files:

- `judge_strict.txt`    — terse, fact-must-match grader (closest to the pinned default).
- `judge_lenient.txt`   — paraphrase/partial-credit-friendly grader (the "generous judge").
- `judge_cot_rubric.txt`— chain-of-thought rubric grader (reason before verdict).
- `answer_concise.txt`  — short fact/phrase only (closest to the pinned default).
- `answer_verbose.txt`  — full-sentence, context-citing answer (the verbose-correct probe).

Stage-2 grid (DEFERRED, PAID): 2 judge models × {strict, lenient, cot_rubric} judge prompts
× {concise, verbose} answer prompts. Every Stage-2 cell is marked PENDING until run.
ZERO of this is executed by the Stage-1 workflow.
