# Released results (aggregates only)

The aggregate, reproducible numbers behind the two papers. No per-question text —
only system/config identifiers, counts, scores, and cost manifests.

- `sweep/stage1.json`, `sweep/swing.csv` — Stage-1 ($0, 0 LLM calls) sensitivity
  surface: retrieval-k recall, metric sensitivity, Cat-5 denominator fork.
- `stage2_rejudge/summary.json` — Stage-2 judge model/prompt sensitivity
  (3 systems x 5 judge configs, pilot p01): j_table, rankings, rank_stability (Kendall tau), cost.
- `pilot_p01/manifest.json` — p01 pilot run manifest (spend, models, config).

## Withheld for LoCoMo CC BY-NC 4.0 compliance
Per-question raw files (`pilot_p01/raw.jsonl`, `stage2_rejudge/cells.jsonl`) embed
LoCoMo questions/gold answers and are NOT redistributed. Regenerate them by cloning
`snap-research/locomo` @ 3eb6f2c and running the harness; the aggregates here reproduce.
