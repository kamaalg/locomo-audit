# PII Redaction Pipeline

Applies to the **public-licensed tier only** (Enron, public diaries, blogs). The synthetic
tier contains **no real PII by construction** — synthetic names/places/numbers are invented and
never map to a real individual, so redaction is a no-op there (a provenance assertion is recorded
instead). This document is the procedure for **Step B** public ingestion.

## Goal

Remove or pseudonymize personally-identifying information from public-source documents before
they enter the corpus, so that (a) no real individual is re-identifiable and (b) the published
synthetic split remains the only verbatim-published text (benchmark-spec §0.6).

## PII categories targeted

| Category | Action | Method |
|---|---|---|
| Person names | Pseudonymize (consistent alias per entity within a persona) | NER (spaCy `en_core_web_trf`) + alias map |
| Email addresses | Replace with `<EMAIL_n>` | regex `[\w.+-]+@[\w-]+\.[\w.-]+` |
| Phone numbers | Replace with `<PHONE_n>` | regex (E.164 + common US/intl formats) |
| Street addresses | Generalize to city-level or `<ADDRESS>` | NER `LOC`/`GPE` + address regex |
| Government IDs (SSN, etc.) | Drop the whole document | regex; document quarantined, not ingested |
| Financial account numbers / cards | Drop the whole document | Luhn check + regex; quarantined |
| URLs with personal handles | Strip handle, keep domain if non-identifying | regex |
| Dates of birth (exact) | Coarsen to year | NER `DATE` + DOB heuristic |

## Pipeline steps

1. **Ingest** raw document with its provenance (`source, license, url, retrieved_at`).
2. **Detect** PII with the NER + regex layer above. Log every span detected.
3. **Quarantine** documents containing hard-identifiers (SSN, financial account, card) — they are
   NOT ingested; a count is kept.
4. **Pseudonymize / mask** the remaining categories. Aliases are consistent **within** a persona
   (so "Dr. Alvarez" stays one entity) but **independent across** personas.
5. **Second-pass review:** human spot-checks a 5% sample per source; if any miss is found the
   regex/NER config is tightened and the source is re-run.
6. **Store** the redacted text + a `redaction_id` referencing the redaction log
   (`data/public/redaction_log/<redaction_id>.json`: spans, actions, reviewer, timestamp).
7. **Provenance stamp:** `metadata.provenance.redaction_id` set on the stored document.

## Protected-class note (ties to T7)

Redaction does **not** annotate or extract protected-class attributes (race, religion, health
status, sexuality, national origin, disability). T7 inference targets are restricted to the frozen
non-protected allowlist (`T7_INFERENCE_ALLOWLIST.md`); the redaction pipeline must never produce a
protected-class label as a side effect.

## Reproducibility

The redaction config (NER model version, regex set, alias seed) is versioned in
`data/public/redaction_config.yaml` and cited in the paper. Re-running the pipeline on the same raw
input with the same config reproduces the redacted output byte-for-byte (aliases are seeded).
