# Independent full-vs-scribble training results

All 18 runs completed on 2026-09-06. Each task was initialized from scratch, trained for 80 epochs with seed 42, and evaluated from its validation-selected `best.pt`. Full and scribble runs share the backbone, PCE objective, optimizer, augmentation, and data split; only the training-label source differs.

## Completion audit

- Domain A--F and Class T1--T3 each have one full-supervision run and one scribble-supervision run.
- Every run has 80 epoch records, `manifest.status=complete`, `best.pt`, `last.pt`, and `independent_scores.json`.
- No training process remains active. GPUs 4--7 returned to idle.
- Checkpoints and full logs remain on `data_nas`; only the compact, public-safe metric table is published here.
- Domain checkpoints were selected by background-inclusive validation Dice. Class checkpoints were selected by foreground validation Dice. Both test variants are reported below.

The first Domain scribble dispatch used the parent sparse-annotation directory instead of its `domain` subdirectory, so those jobs stopped at the preflight input check before training. The path was corrected and only the seven missing runs (six Domain scribble runs and Domain E full) were launched. Completed artifacts were neither overwritten nor counted twice.

## Domain-CL

The background-inclusive Dice is the primary value under the current Domain audit convention.

| Task | Full inclusive | Scribble inclusive | Full - Scribble | Full foreground | Scribble foreground |
|---|---:|---:|---:|---:|---:|
| A | 0.9413 | 0.4627 | +0.4786 | 0.8844 | 0.0991 |
| B | 0.9316 | 0.5784 | +0.3532 | 0.8652 | 0.2142 |
| C | 0.9496 | 0.5920 | +0.3576 | 0.9014 | 0.2572 |
| D | 0.8877 | 0.6203 | +0.2674 | 0.7785 | 0.2725 |
| E | 0.9397 | 0.6001 | +0.3396 | 0.8822 | 0.2663 |
| F | 0.9294 | 0.5804 | +0.3490 | 0.8600 | 0.1997 |
| **Mean** | **0.9299** | **0.5723** | **+0.3576** | **0.8620** | **0.2182** |

Full supervision is higher on every Domain task. The largest inclusive gap is A (+0.4786), and the smallest is D (+0.2674). The scribble-only PCE reference is therefore a weak standalone baseline on these prostate domains, especially A; this does not by itself indicate an execution failure because every run passed the artifact and completion gates.

## Class-CL

Foreground Dice is the checkpoint-selection value for Class-CL; background-inclusive Dice is included for direct comparison.

| Task | Full foreground | Scribble foreground | Full - Scribble | Full inclusive | Scribble inclusive |
|---|---:|---:|---:|---:|---:|
| T1 | 0.8560 | 0.5678 | +0.2882 | 0.8913 | 0.6701 |
| T2 | 0.8539 | 0.6190 | +0.2349 | 0.9014 | 0.7414 |
| T3 | 0.7971 | 0.4622 | +0.3350 | 0.8642 | 0.6396 |
| **Mean** | **0.8357** | **0.5496** | **+0.2860** | **0.8857** | **0.6837** |

Full supervision is again higher on all three tasks. T2 is the strongest scribble-trained Class task (0.6190 foreground Dice), while T3 shows the largest full-vs-scribble foreground gap (+0.3350).

## Interpretation boundary

These numbers quantify independent per-task training, not continual-learning retention, forgetting, forward transfer, or final-sequence average Dice. They are suitable as independently trained reference scores for later CL metric calculations. The published CSV contains the exact test values, best validation values, selected epoch, and selected iteration for every run.
