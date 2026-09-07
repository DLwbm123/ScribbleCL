# Domain-A spatial warmup sweep at 80 epochs

Status: launched on 2026-09-07. The user requested testing spatial activation time before proceeding with the other domains' formal experiments. GPU 4-7 were checked idle before launch and now each run one candidate. B-F formal training has not started.

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

Before this sweep, the selected-task shared entry was extended from A to B-F. A six-epoch, one-batch-per-epoch B smoke check passed: 168 B training slices, stage 0 throughout, finite losses, spatial active only at index 5 (raw loss 0.8374210596), no test evaluation. Formal B-F jobs remain deferred while activation time is compared. The multi-GPU domain queue's runnable self-check passed complete coverage and failure reporting.

Data, annotations, checkpoints, and full runtime logs remain on experiment storage. This report records an active launch, not completed sweep results.
