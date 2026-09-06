# Current Domain-CL results

All retained runs use seed 42. Background-inclusive Dice is the requested
headline metric; foreground-only Dice remains visible because background Dice
is approximately 0.99 and can hide foreground failure.

| Method | Run | Status | Epochs/task | Foreground Dice | Background-inclusive Dice | Role |
|---|---|---|---:|---:|---:|---|
| ZS-GPM | `m7v2q` | Complete, 6/6 stages | 150 | 0.3247 | 0.6576 | Continual baseline |
| ZS-DER++ | `t4m7b` | Complete, 6/6 stages | 80 | 0.5437 | 0.7687 | Formal result |
| ZS-Joint | `y9h4m` | Complete short run | 5 | 0.5628 | 0.7760 | Convergence diagnostic only |

The formal ZS-DER++ final checkpoint produced background-inclusive Dice
`0.8273 / 0.8721 / 0.7994 / 0.6156 / 0.6826 / 0.8154` on domains A-F. Its
foreground A-Dice is `0.543748`, BWTR is `-0.204754`, and E-FWT is `0.261256`.

The Joint row is not a formal upper bound because it uses only five pooled
epochs. Superseded low-score artifacts from an earlier configuration and a
divergent standalone Joint entry were removed from the current tree; Git
history retains them for recovery.

See `results/final_metrics.csv` for the compact table and each result directory
for manifests, matrices, summaries, and paired metric outputs.
