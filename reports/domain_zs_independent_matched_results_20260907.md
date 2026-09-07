# Domain ZS independent matched-loss rerun

All six Domain A--F runs completed on 2026-09-07. Each run used scribble supervision, seed 42, 80 epochs, learning rate 0.03, and foreground validation Dice for checkpoint selection. The loss configuration was matched to the base ZS objective used by the earlier sequential runs: PCE 1.0, global consistency 1.0, and spatial loss 0.0.

## Completion audit

- Six of six runs reached epoch 79 and produced `best.pt`, `last.pt`, `manifest.json`, and `independent_scores.json`.
- No scoped process remained active, GPUs 4--7 were idle, and the logs contained no matched runtime-error signature.
- The 1.2 GB of checkpoints and full logs remain under `/data_nas/jiangsuiyang/ScribbleCL/zs_independent_matched_seed42_20260907` and are not published to GitHub.

## Results

| Domain | Foreground Dice | Inclusive Dice | Predicted foreground fraction | Best epoch |
|---|---:|---:|---:|---:|
| A | 0.0761 | 0.4194 | 0.3933 | 59 |
| B | 0.2598 | 0.6080 | 0.0970 | 9 |
| C | 0.3509 | 0.6538 | 0.0980 | 4 |
| D | 0.1255 | 0.5133 | 0.1858 | 4 |
| E | 0.2288 | 0.5711 | 0.1790 | 4 |
| F | 0.1103 | 0.5150 | 0.1561 | 4 |
| **Mean** | **0.1919** | **0.5468** |  |  |

The matched-loss rerun is 0.2025 foreground Dice below the tuned ZS independent run (0.3944) and 0.0263 below the PCE-only independent reference (0.2182). It therefore falsifies the hypothesis that spatial loss alone caused the low independent scores.

The main observable failure is excessive foreground prediction: five domains predict foreground on roughly 9.7%--39.3% of pixels, far above the approximately 1.1%--3.8% seen on the earlier sequential diagonals. A small sparse PCE loss only confirms that annotated pixels were fitted; it does not constrain the large unlabeled area sufficiently.

The earlier sequential values came from ZS-GPM and ZS-DER++, not from a plain ZS sequential control. Their current-task diagonal means were 0.6465 and 0.6716 foreground Dice, respectively. Consequently this rerun is a completed negative control, not a valid final independent reference. The next isolating comparison should run task A through the ordinary continual code path with `zs-sequential`, `zs-gpm`, and `zs-derpp` separately before launching another six-task sweep.
