# LLM-judge prompts — usage and guardrails

PersonalMem minimizes LLM-judge use by design (see `docs/benchmark-spec.md` §4). Of the eight
categories, **only T2 (cross-document synthesis)** unavoidably touches a judge, and only as a
**tie-breaker** for atomic-claim entailment in the ambiguous band `0.3 < token_f1 < 0.6`. Every
other category is scored programmatically (EM / token-F1 / numeric / date / multiple-choice /
held-out stylometric classifier / programmatic withhold-check).

## Mandatory guardrails for ANY judge use
1. **3 judges + majority vote.** No majority -> conservative 0 (claim not entailed) in production;
   escalate to a human during calibration.
2. **Cross-family judges.** The judge panel MUST NOT include a model from the same family as the
   system being scored on that cell. (Anti-pattern explicitly forbidden: a model judging its own
   family.) Recommended panel: three distinct families; rotate so no system is ever judged by its
   own family.
3. **Adversarial gate.** Each prompt ships adversarial test cases (below). The panel must produce
   the expected verdicts on ALL of them before a run is considered valid. A panel that fails an
   adversarial case is mis-calibrated; fix the prompt or swap models.
4. **Calibration gate (spec §2.2).** Judge-majority vs human score, Pearson r > 0.85 over 100
   instances. If r <= 0.85, DROP the judge and fall back to pure token-F1 claim matching.

## Files
- `t2_claim_entailment.md` — the only production judge prompt, plus its adversarial cases.

Judges output strict JSON so parsing is deterministic. Temperature 0. No chain-of-thought in the
returned field (reasoning may be in a separate `rationale` field, not used for scoring).
