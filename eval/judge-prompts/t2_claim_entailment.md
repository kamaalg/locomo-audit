# T2 claim-entailment judge prompt

Used ONLY for gold atomic claims where programmatic `token_f1(prediction, claim)` falls in the
ambiguous band `0.3 < f1 < 0.6`. Outside that band the score is purely programmatic and the judge
is NOT called. Run as a 3-judge cross-family panel with majority vote (see README guardrails).

## System prompt

```
You are a strict entailment checker for a benchmark. You are given a SYSTEM ANSWER and a single
GOLD CLAIM. Decide ONLY whether the GOLD CLAIM is entailed by (explicitly stated or unambiguously
implied by) the SYSTEM ANSWER.

Rules:
- "entailed" = a careful reader of the SYSTEM ANSWER would agree the GOLD CLAIM is asserted or
  unambiguously implied by it. Paraphrase is fine; synonyms are fine.
- "not_entailed" = the SYSTEM ANSWER does not assert the claim, contradicts it, or only weakly
  hints at it without asserting it.
- Judge ONLY entailment of THIS claim. Do not reward fluency, extra correct claims, or style.
- Do NOT use outside knowledge. Judge solely from the SYSTEM ANSWER text.
- If the SYSTEM ANSWER asserts the OPPOSITE of the claim, that is "not_entailed".
- Partial/hedged assertion of the claim ("maybe", "possibly") with no commitment = "not_entailed".

Output STRICT JSON only, no prose outside it:
{"verdict": "entailed" | "not_entailed", "rationale": "<=20 words"}
```

## User prompt template

```
GOLD CLAIM:
{claim}

SYSTEM ANSWER:
{prediction}
```

## Scoring
- Per judge: `verdict`. Panel verdict = majority of 3. `entailed` -> claim counts toward
  claim-recall; `not_entailed` or no-majority -> does not count.
- Temperature 0. Parse the JSON; a malformed response counts as `not_entailed` for that judge
  (conservative) and is logged.

## Adversarial test cases (panel must pass ALL before a run is valid)

### A1 — paraphrase, should be `entailed`
- claim: `"mood lifts on weekends"`
- prediction: `"On Saturdays and Sundays your entries are noticeably more upbeat and positive."`
- expected: `entailed` (weekend = Sat/Sun; lifts = more upbeat). Tests that paraphrase is accepted.

### A2 — opposite assertion, must be `not_entailed`
- claim: `"you quit caffeine"`
- prediction: `"You drink coffee every single morning without fail."`
- expected: `not_entailed`. Tests that a confident OPPOSITE is rejected (catches lenient judges).

### A3 — topical overlap but claim not asserted, must be `not_entailed`
- claim: `"your mood is linked to exercise"`
- prediction: `"You write a lot about your mood, and separately you mention going to the gym."`
- expected: `not_entailed` (co-mention is not a causal/associative link assertion). Tests that mere
  keyword overlap (high-ish token-F1) does not trick the judge into entailment.

### A4 — hedged, must be `not_entailed`
- claim: `"you moved to Lisbon"`
- prediction: `"You might be considering a move, possibly to Lisbon at some point."`
- expected: `not_entailed` (hedged, not asserted). Tests rejection of non-committal hints.
