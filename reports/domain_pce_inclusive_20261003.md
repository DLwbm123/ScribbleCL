# Domain-CL PCE-Sequential: background-inclusive checkpoint reevaluation

This reevaluation uses the existing six-stage `x7p2a` run and its original independent PCE reference `x7p2r` (seed 42, 150 epochs per task). It changes only the evaluation class set to background 0 and foreground 1. It does not retrain, tune, select new checkpoints, or change the original test split.

## Results

| Metric | Original foreground only | Including background |
| --- | ---: | ---: |
| A-Dice | 0.237069 | 0.607314 |
| BWTR | -0.565429 | -0.191535 |
| RMA | 0.775513 | 0.917630 |
| E-FWT | 0.092679 | 0.350387 |

| Final domain | Foreground Dice | Background Dice | Inclusive Dice |
| --- | ---: | ---: | ---: |
| A | 0.208717 | 0.954734 | 0.581726 |
| B | 0.334230 | 0.974530 | 0.654380 |
| C | 0.306015 | 0.982564 | 0.644289 |
| D | 0.071851 | 0.988182 | 0.530016 |
| E | 0.070573 | 0.977622 | 0.524098 |
| F | 0.431030 | 0.987714 | 0.709372 |

## Calculation and verification

Dice is calculated per patient and per class, then averaged equally over patients and the two classes. The final A-Dice averages the six domains equally. BWTR averages the relative final-versus-acquisition change over A–E. RMA averages acquisition Dice divided by the corresponding independent PCE reference over B–F. E-FWT averages the 15 upper-triangle differences against the corresponding same-seed random-model score. All components use the same inclusive definition.

All 48 task evaluations passed foreground parity against the preserved original records: 36 stage/task cells, six random-model scores and six independent-reference scores. The maximum absolute foreground difference was 1.11022302463e-16. All four recomputed foreground aggregate metrics also matched the original values within 1e-4. The inclusive aggregates were independently checked from the saved scalar matrix.

Evaluation took 575.4 seconds on one RTX 3090; peak reserved CUDA memory was 1062.0 MiB. Runtime: Python 3.10.6 and PyTorch 2.2.1+cu121.

The previously transcribed thesis A-Dice 0.2480 could not be linked to an original run. These results are tied to `x7p2a`, whose preserved foreground A-Dice is 0.23706938696044358. They replace that manuscript entry with a traceable evaluation; they are not an algebraic conversion of 0.2480. This report does not reevaluate other methods or establish matched-budget comparisons with them.

## Artifacts and reproduction

- [Complete scalar matrices, references and checks](../results/domain_pce_inclusive_20261003/results.json)
- [Preserved original foreground metrics and manifests](../results/domain_pce_inclusive_20261003/original_foreground.json)
- [Evaluation script](../tools/domain_pce_inclusive/run.py)

Copy the script into a new private working directory as `run.py`. Provide a `plan.json` containing `code` (the original compatible runner checkout), `data_root`, `run_root` (x7p2a), `reference_root` (x7p2r), `seed: 42`, and `batch_size: 4`. Select a GPU through `CUDA_VISIBLE_DEVICES`, then execute `python -u run.py` from that directory. Original run inputs must include `metrics.json`, the six selected checkpoints, and the independent reference scores/checkpoints. Evaluation fails if foreground scores do not reproduce.

Private medical data, model weights, absolute server paths and raw logs are excluded from the public artifacts.
