# Organ-CL completed comparisons — 2026-09-11

All five comparisons finished by **01:30:22 Asia/Shanghai**. The 10:27 completion audit confirmed zero per-job/pipeline exit codes, all **60/60/10/10 epochs**, finite epoch losses and four nonempty selected stage checkpoints per run. Campaign duration: approximately 2 hours 28 minutes. No new experiment was launched.

Scores are **foreground-only per-case macro Dice (%)** after T4. BWTR is the mean relative change from each old task's acquisition Dice to final Dice, expressed as percent; closer to zero is better. Epsilon-sized scores round to 0.00.

| Method | T1 | T2 | T3 | T4 | Mean Dice | BWTR (%) | T1–T3 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZSDERpp (provisional) | 63.43 | 57.29 | 79.22 | 81.95 | 70.47 | -6.18 | 66.64 |
| zs-gpm | 15.07 | 7.64 | 3.73 | 75.58 | 25.51 | -85.18 | 8.81 |
| zs-ewc | 8.33 | 1.93 | 0.00 | 80.57 | 22.71 | -94.03 | 3.42 |
| zs-sequential | 0.00 | 3.45 | 0.00 | 81.74 | 21.30 | -97.69 | 1.15 |
| dense-sequential | 0.00 | 0.00 | 0.00 | 85.00 | 21.25 | -100.00 | 0.00 |
| pce-sequential | 0.00 | 0.22 | 20.24 | 77.59 | 24.51 | -91.20 | 6.82 |

The provisional main result exceeds ZS-GPM, the highest baseline mean, by **44.96 percentage points**. Its old-task mean is **66.64%**. The gap concentrates in retention: ZS-Sequential acquires T3 at 81.95% but finishes near zero on T3, while its T4 Dice (81.74%) is close to the main method (81.95%). Dense-Sequential acquires T1/T2/T3 at 86.83/82.10/90.15% and finishes T4 at 85.00%, but loses virtually all earlier foreground performance. Dense labels alone do not prevent forgetting in this setup.

The low baseline final scores appear in saved task matrices despite completed execution and finite losses. They are not process failures. This summary does not establish the sole cause of forgetting or constitute a full implementation audit.

## Protocol and interpretation limits

See [launch protocol](organ_comparisons_20260910.md). Controls start from scratch with seed 42, the same fixed Organ subsets, architecture, task order, validation selection, epoch/LR schedule and non-replay transition controls. ZS uses Spatial .01/.01/0/0, first active at epoch 30 in T1/T2. Dense uses full labels and is a different-supervision reference.

The main row is the previously accepted continuation campaign with inherited T1/T2 and validation-selected T3/T4 model/replay states, not a fresh main-method rerun. It was selected after exploratory comparisons that included observing test results. These are provisional single-seed reporting results, **not an untouched-test confirmatory comparison or a statistically significant superiority claim**. They do not isolate replay as the sole cause of the gap. Foreground scores must remain separate from the background-inclusive Domain thesis metrics.

RMA is unreported because existing independent references use different budgets. GPM DRR remains unset pending the representation-count convention; Sequential/EWC DRR is zero. All five methods have MPE=0.000005202012386845555 as a ratio. Missing metrics are not invented.

## Artifacts

- [Task matrices and provisional reference](../results/organ_comparisons_20260910/comparison.json)
- [Final metric CSV](../results/organ_comparisons_20260910/final_metrics.csv)
- [Completion audit](../results/organ_comparisons_20260910/completion_check.json)
- Exact deployed Python source: organ_comparisons_runtime/ in the public release.
- Private root: /data_nas/jiangsuiyang/ScribbleCL/organ_comparisons_20260910.

Publication includes source, protocol, aggregate metrics and completion receipts. Images, labels, patient-level results, replay/source IDs, numerical traces and checkpoints remain private on NAS.
