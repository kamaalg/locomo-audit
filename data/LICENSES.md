# Data Sources & Licenses

PersonalMem corpus composition target (ADR 0003): **~70% synthetic, ~30% public-licensed.**
The **pilot** (this dispatch) is **100% synthetic** — no public source is touched until Step B.

## Release license

- **Benchmark data + spec:** CC-BY-4.0 (per benchmark-spec §0.6).
- **Code (harness, metrics, judge prompts, generation/QC scripts):** Apache-2.0 — see `LICENSE`.
- Synthetic personas and their corpora are original works authored for this project and
  released under the data license (CC-BY-4.0). No third-party rights attach to synthetic text.

## Synthetic tier (preferred, ~70%)

| Source | License | Notes |
|---|---|---|
| LLM-generated personas + corpora | Original work, CC-BY-4.0 | Prompts/models/seeds in `SYNTHETIC_GENERATION.md`. Zero consent risk. |

## Public-licensed tier (~30%, **STEP B ONLY — not in pilot**)

Every public source MUST have a clear license recorded here before ingestion. Reject anything
without one. PII redaction (`REDACTION.md`) is applied to all public-tier text before use.

| Source | License / Terms | Status | Redaction required | Notes |
|---|---|---|---|---|
| Enron Email Corpus | Public domain (released by FERC; CMU distribution) | PLANNED (Step B) | YES (names, emails, phones) | Journal/email-like personal text. Cite CMU CALO distribution. |
| Project Gutenberg diaries / autobiographies | Public domain (US) | PLANNED (Step B) | LIGHT (modern PII unlikely) | Respect Gutenberg ToS; US-PD works only. |
| Public blogs (robots-respecting) | Per-site; only explicitly-licensed (CC) or ToS-permitting | CANDIDATE (Step B) | YES | robots.txt honored; reject ambiguous licenses. |
| Reddit | Per Reddit ToS + API terms | UNDER REVIEW (Step B) | YES | ToS review required before any pull; may be dropped. |

### Rules (enforced at ingestion, Step B)

1. **No license → no ingestion.** Ambiguous license == no license.
2. **robots.txt + ToS respected** for any web source.
3. **No private-platform data** (Slack/Discord/WhatsApp exports) even if accessible (`data-ethics.md` §"What we will NOT do").
4. **PII redaction before storage** for every public-tier document; redaction log retained.
5. **Privacy-preserving release:** real/licensed subsets release **aggregate metrics + derived labels only**,
   never raw personal text. Only the **synthetic split** is published verbatim (benchmark-spec §0.6).
6. **Cite source + license in the paper** for every subset.

## Provenance manifest (Step B)

Each ingested public document will carry `metadata.provenance = {source, license, url, retrieved_at, redaction_id}`.
The pilot's synthetic documents carry `metadata.provenance = {source: "synthetic", generator, seed}`.
