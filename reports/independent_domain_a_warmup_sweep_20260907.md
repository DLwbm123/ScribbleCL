# Domain-A spatial warmup sweep at 80 epochs

Status: **completed at 2026-09-07 17:26:47 Asia/Shanghai**. All four new candidates completed 80 epochs and 6,080 optimizer steps; all four exit codes and the coordinator exit code were 0. Validation selected **10 warmup epochs, spatial enabled from the 11th epoch**. B-F formal training subsequently started using this selected configuration.

## Completed results

| First spatial-active epoch | Best foreground validation Dice | Best checkpoint epoch (one-based) |
|---:|---:|---:|
| 6 (reused reference) | 0.6365137028 | 43 |
| **11** | **0.6759455315** | **74** |
| 21 | 0.5972746958 | 24 |
| 41 | 0.6525318701 | 61 |
| 61 | 0.6726428838 | 77 |

The 11th-epoch candidate exceeded the 61st-epoch candidate by only 0.0033026477 validation Dice. This one-seed result does not establish a robust advantage for one activation time.

After validation selection was frozen, the selected `warmup10/best.pt` checkpoint (epoch index 73, iteration 5,600) was evaluated once on A test using [evaluate_selected_warmup.py](../evaluate_selected_warmup.py): **foreground Dice 0.6531867577**, background-inclusive Dice **0.8232875366**. This is below the prior five-warmup-epoch reference's test Dice 0.7575264500. The warmup sweep improved validation selection score, but did not improve A test performance or meet 0.7261 with its selected checkpoint. No additional candidate was tested or re-selected based on this test outcome. The original 0.7575 artifacts remain preserved.

Public evidence: [selection and all candidates](../results/independent_domain_a_warmup_sweep_20260907/selection.json), [completion and exit codes](../results/independent_domain_a_warmup_sweep_20260907/pipeline.json), and [selected checkpoint test result](../results/independent_domain_a_warmup_sweep_20260907/selected_test.json).

## Protocol and provenance

The previous coefficient sweep fixed five warmup epochs. That was 25% of a 20-epoch run but only 6.25% of an 80-epoch run. This follow-up compares the actual 80-epoch schedules; a 20-epoch pilot cannot test activation at epochs 41 or 61.

| GPU | Warmup epochs | First spatial-active epoch (one-based) | Runner warmup argument |
|---:|---:|---:|---:|
| completed reference | 5 | 6 | 4 |
| 4 | 10 | 11 | 9 |
| 5 | 20 | 21 | 19 |
| 6 | 40 | 41 | 39 |
| 7 | 60 | 61 | 59 |

Each new candidate is a fresh seed-42 independent Domain-A run: 80 epochs, 6,080 steps, PCE/global/spatial 1/1/0.01, batch size 4, LR 0.03 with polynomial decay, SGD momentum 0.9, optimizer decay 0 plus manual gradient decay 1e-4, historical pattern_f5_b10 scribbles, 8 workers and OMP_NUM_THREADS=4. Each uses the shared stage-training implementation, with no checkpoint initialization or replay.

The completed five-warmup-epoch reference contributes only its foreground validation Dice, 0.6365137028, to selection. All four new runs disable test evaluation. The coordinator selects the greatest best foreground validation Dice across all five candidates; exact ties prefer the shorter warmup. The baseline's already-known test result is not used by this selector. This is a one-seed comparison and does not remove CUDA nondeterminism.

Source: [run_independent_a_warmup_sweep.py](../run_independent_a_warmup_sweep.py), runtime revision `e4e0142`. The existing [training helper](../run_independent_a_spatial_sweep.py) now accepts a warmup argument and checks the expected activation epoch and active-epoch count for each run.

Runtime root: `/data_nas/jiangsuiyang/ScribbleCL/independent_A_warmup_sweep80_20260907`. Tmux: `independent-a-warmup80`. Its sibling `_launch.sh` records the exact command, `_coordinator.log` records orchestration, and `_coordinator.exitcode` records completion. `pipeline.json` records source/protocol/status; `selection.json` is written only after all four new candidates pass their completion checks. Each candidate has command, log, exit code, compact result, manifest, train log, and best/last checkpoint artifacts.

Before this sweep, the selected-task shared entry was extended from A to B-F. A six-epoch, one-batch-per-epoch B smoke check passed: 168 B training slices, stage 0 throughout, finite losses, spatial active only at index 5 (raw loss 0.8374210596), no test evaluation. The multi-GPU domain queue's runnable self-check passed complete coverage, failure reporting, and forwarding the selected warmup argument.

## B-F formal continuation

The user's authorization to run all domains on available GPUs 4-7 remains in effect. After this sweep, fresh B-F runs were launched with PCE/global/spatial 1/1/0.01, 80 epochs, ten warmup epochs (`--zs-spatial-warmup-epochs 9`), seed 42, and the same shared stage-training implementation and historical scribbles. No per-domain hyperparameter sweep is implied. Each domain selects its checkpoint by its own foreground validation Dice and evaluates that checkpoint once on its own test split.

Verified initial assignment: B on GPU 5, C on GPU 6, D on GPU 7, E on GPU 4; F waits for the first available worker. GPU 4 briefly handled the selected A test evaluation before starting E. Runtime source revision: `dc81893`. Tmux: `independent-bf-formal80`. Runtime root: `/data_nas/jiangsuiyang/ScribbleCL/independent_BF_formal80_warmup10_20260907`; its sibling `_launch.sh` records the exact command. The queue uses [run_independent_domains_formal.py](../run_independent_domains_formal.py) with `--warmup-epochs 10` and `--completed-a` pointing to the selected `warmup10` directory. These B-F runs are ongoing; no formal B-F results are claimed yet.

Data, annotations, checkpoints, and full runtime logs remain on experiment storage. Only source code, protocol, compact metrics, and this report are published.
