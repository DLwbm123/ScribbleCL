# ZS scribble independent-training results

The coefficient search and all nine formal independent runs completed on 2026-09-07. Search runs used 20 epochs and seed 42; formal runs used 80 epochs and seed 42. Domain and Class checkpoint selection both maximized foreground validation Dice, while the result tables report both foreground and background-inclusive Dice.

## Completion audit

- The coefficient search completed 16/16 runs without using the test split for selection.
- Formal training completed Domain A--F and Class T1--T3: 9/9 runs reached epoch 79 and produced `best.pt`, `last.pt`, `manifest.json`, and `independent_scores.json`.
- No training process remained active and GPUs 4--7 were idle at the completion check.
- The nine formal artifact directories occupy 1.8 GB under `/data_nas/jiangsuiyang/ScribbleCL/zs_independent_search_seed42_20260906/formal`; checkpoints and full logs are not published to GitHub.

## Selected coefficients

| Scenario | Candidate | PCE | Global | Spatial | Search validation foreground | Search validation inclusive |
|---|---|---:|---:|---:|---:|---:|
| Domain | c07 | 0.5 | 1.0 | 0.05 | 0.6262 | 0.7994 |
| Class | c05 | 1.0 | 0.5 | 0.05 | 0.4719 | 0.6443 |

The formal runs used a 20-epoch spatial-loss warmup. Domain coefficients were selected on task C and Class coefficients on task T2, then shared across all tasks in the corresponding scenario.

## Formal test results

| Scenario | Task | Foreground Dice | Inclusive Dice | Best epoch |
|---|---|---:|---:|---:|
| Domain | A | 0.2565 | 0.6041 | 54 |
| Domain | B | 0.3045 | 0.6346 | 54 |
| Domain | C | 0.4589 | 0.7156 | 74 |
| Domain | D | 0.5427 | 0.7674 | 59 |
| Domain | E | 0.5016 | 0.7405 | 39 |
| Domain | F | 0.3018 | 0.6401 | 24 |
| **Domain mean** |  | **0.3944** | **0.6837** |  |
| Class | T1 | 0.6804 | 0.7572 | 64 |
| Class | T2 | 0.6686 | 0.7757 | 19 |
| Class | T3 | 0.7101 | 0.8059 | 74 |
| **Class mean** |  | **0.6864** | **0.7796** |  |

## Comparison with the PCE scribble reference

| Scenario | Metric | PCE scribble | Tuned ZS scribble | Change |
|---|---|---:|---:|---:|
| Domain | Foreground Dice | 0.2182 | 0.3944 | +0.1762 |
| Domain | Inclusive Dice | 0.5723 | 0.6837 | +0.1114 |
| Class | Foreground Dice | 0.5496 | 0.6864 | +0.1367 |
| Class | Inclusive Dice | 0.6837 | 0.7796 | +0.0959 |

ZS improves the mean foreground and inclusive test Dice over PCE scribble in both scenarios. Domain gains are uneven: D and E are strongest, while A, B, and F remain substantially below the full-supervision references. Class gains are more consistent, and T3 has the largest improvement over its PCE scribble reference.

The historical Domain PCE checkpoint was selected by inclusive validation Dice, whereas this corrected ZS run was selected by foreground validation Dice. Therefore the Domain comparison is useful as a result summary but is not a perfectly selection-matched ablation. Exact per-run values and exact search scores are available in the accompanying CSV files.
