# ZS independent scribble tuning and formal training

Status: tuning launched on 2026-09-06 at 21:09 CST; formal training is configured to start automatically after validation-only selection.

## Objective

Tune the three additive ZS losses for independent scribble-supervised training:

`L = w_pce * L_pce + w_global * L_global + w_spatial * L_spatial`

Domain-CL and Class-CL select separate shared coefficient sets because their class spaces and loss scales differ. The tuning representatives are Domain C and Class T2. Every candidate uses seed 42, 20 epochs, batch size 4, and spatial warm-up through epoch 5.

## Candidate grid

| Candidate | PCE | Global | Spatial |
|---|---:|---:|---:|
| c01 | 1.0 | 0.1 | 0.01 |
| c02 | 1.0 | 0.5 | 0.01 |
| c03 | 1.0 | 1.0 | 0.01 |
| c04 | 1.0 | 2.0 | 0.01 |
| c05 | 1.0 | 0.5 | 0.05 |
| c06 | 1.0 | 1.0 | 0.05 |
| c07 | 0.5 | 1.0 | 0.05 |
| c08 | 2.0 | 1.0 | 0.05 |

Domain selection maximizes background-inclusive validation Dice. Class selection maximizes foreground validation Dice. Tuning runs use `--independent-skip-test`, so no test split is evaluated or used for coefficient selection.

## Formal continuation

After all 16 tuning runs complete, the coordinator writes `tuning_summary.json`, selects one coefficient tuple per scenario, and immediately launches:

- Domain A--F: six independent 80-epoch ZS scribble runs.
- Class T1--T3: three independent 80-epoch ZS scribble runs.
- Spatial warm-up through epoch 20 for the 80-epoch runs.
- Per-run validation-selected `best.pt`, separate `last.pt`, and final test metrics.

Large tuning and checkpoint artifacts are stored on `data_nas` and are not committed. The launch check observed eight concurrent tuning processes across GPUs 4--7, approximately 7.7 GB allocated per GPU, and 87%--100% instantaneous utilization.
