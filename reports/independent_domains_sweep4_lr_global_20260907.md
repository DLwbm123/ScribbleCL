# Four-candidate per-domain LR/global sweep, then selected formal training

Status: **completed at 2026-09-08 01:30:26 Asia/Shanghai**, after launch at 2026-09-07 22:00:32 (about 3 hours 30 minutes). All **24 new 20-epoch sweep runs plus 6 fresh 80-epoch formal runs** completed, with 30 training exit codes and the coordinator exit code all 0. This meets the user's limit of four new sweep candidates per domain. Earlier experiments are preserved.

The previously authorized platform-demo protocol continues: each domain's original test split is the selection split. All scores are explicitly labeled demo selection scores and are not held-out test estimates. This batch does not use a shared cross-domain winner.

## Completed sweep scores

| Domain | s0: LR .03 / Global 1 | s1: LR .01 / Global 1 | s2: LR .03 / Global .1 | s3: LR .01 / Global .1 |
|---|---:|---:|---:|---:|
| A | 0.7475 | 0.7000 | **0.7592** | 0.6174 |
| B | **0.5516** | 0.4953 | 0.5094 | 0.4745 |
| C | 0.5168 | **0.6034** | 0.5803 | 0.4908 |
| D | **0.4594** | 0.4048 | 0.3917 | 0.2964 |
| E | 0.6297 | **0.6318** | 0.6224 | 0.4707 |
| F | 0.5510 | **0.6448** | 0.5822 | 0.6046 |

Each cell is that candidate's best original-test foreground demo selection Dice during 20 epochs. B, C, and F's selected sweep checkpoints occurred before spatial activation; their selection cannot establish the effectiveness of the spatial component. E's s1 lead over s0 is only 0.0021633752 and should not be described as a robust advantage from one seed.

## Fresh 80-epoch formal results

| Domain | Selected LR | Selected Global | Sweep winner Dice | New formal Dice | Previous formal Dice | Change (percentage points) |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.03 | 0.1 | 0.7592 | **0.5843** | 0.7366 | **-15.24** |
| B | 0.03 | 1.0 | 0.5516 | 0.5524 | 0.5414 | +1.10 |
| C | 0.01 | 1.0 | 0.6034 | 0.6462 | 0.6413 | +0.49 |
| D | 0.03 | 1.0 | 0.4594 | 0.5370 | 0.4804 | +5.67 |
| E | 0.01 | 1.0 | 0.6318 | 0.7054 | 0.6899 | +1.54 |
| F | 0.01 | 1.0 | 0.6448 | 0.6787 | 0.6628 | +1.59 |
| Unweighted mean | | | | **0.6173** | **0.6254** | **-0.81** |

Five domains improved relative to the [previous demo batch](independent_domains_demo_test_selection_20260907.md), but A regressed enough that the formal six-domain mean decreased. B/D retained the same LR/global settings, while evaluation frequency changed from every 200 steps to every epoch for all domains. These comparisons therefore combine checkpoint sampling, training variability, and (where changed) hyperparameters; improvements cannot be attributed solely to the sweep.

A's failure to transfer from 0.7591552963 in the 20-epoch sweep to 0.5842726165 in the fresh 80-epoch run is material. The formal run did not resume the sweep checkpoint; it used a different-length polynomial LR schedule and a new CUDA training trajectory. Completion checks confirmed the selected LR/global parameters were forwarded correctly. This result demonstrates the limitation of short-budget selection; it does not establish a specific causal explanation for the drop. No additional training or candidate testing was launched during this closeout.

## Platform checkpoint choices from this batch

For the already-authorized demonstration use, the strongest available checkpoint in this batch is **A_s2/best.pt** for A (20-epoch sweep, score **0.7591552963**, selected at epoch 17). For B-F, the strongest available checkpoints are their new `<domain>_formal80/best.pt` files, selected at epochs 42, 71, 60, 74, and 51 respectively. This mixed sweep/formal collection has an unweighted mean demo selection Dice of **0.6464683087**. It must not be called the six-domain 80-epoch formal result. No platform deployment or checkpoint replacement was performed.

Public artifacts: [all 30 runs](../results/independent_domains_sweep4_lr_global_20260908/all_runs.csv), [formal comparison](../results/independent_domains_sweep4_lr_global_20260908/formal_comparison.csv), [checkpoint choices with phase provenance](../results/independent_domains_sweep4_lr_global_20260908/demo_checkpoint_choices.csv), [completed protocol](../results/independent_domains_sweep4_lr_global_20260908/pipeline_complete.json), [exact commands](../results/independent_domains_sweep4_lr_global_20260908/commands.json), [manifests](../results/independent_domains_sweep4_lr_global_20260908/manifests.json), and [selection curves](../results/independent_domains_sweep4_lr_global_20260908/selection_curves.csv).

## Candidate grid

| Candidate | Learning rate | Global loss weight | PCE weight | Spatial weight | First spatial-active epoch |
|---|---:|---:|---:|---:|---:|
| s0 | 0.03 | 1.0 | 1 | 0.01 | 11 |
| s1 | 0.01 | 1.0 | 1 | 0.01 | 11 |
| s2 | 0.03 | 0.1 | 1 | 0.01 | 11 |
| s3 | 0.01 | 0.1 | 1 | 0.01 | 11 |

This is a 2-by-2 comparison of learning rate and global-consistency strength, the two controls not previously swept independently per domain. It includes the preceding configuration as s0. No additional candidates or automatic retries are scheduled.

Each candidate trains from scratch for 20 epochs with seed 42, batch size 4, SGD momentum 0.9, optimizer weight decay 0 plus manual gradient decay 1e-4, the historical pattern_f5_b10 scribbles, 8 workers, and OMP_NUM_THREADS=4. Spatial is disabled for ten epochs and enabled for the remaining ten. The same ten-epoch warmup is used for formal training. Other model, loss, and augmentation implementation details remain those of the shared stage loop.

Evaluation now occurs **every epoch**, using each domain's batches-per-epoch interval: A 76, B 42, C 87, D 42, E 146, F 176. Input slice counts were checked before launch. Both sweep and formal training use this frequency, avoiding the former approximately five-epoch gap between evaluations for B/D. A final evaluation also runs at completion.

## Selection and formal phase

The coordinator first completes all 24 sweep jobs. Each domain is selected using the largest best foreground demo selection Dice among its own four completed candidates. Exact ties prefer the lower candidate index. Missing, duplicate, failed, or non-finite candidate results block formal launch rather than silently selecting from an incomplete set.

After all six domain selections are written to `selection.json`, six new 80-epoch formal runs are started automatically. Each uses its own selected LR/global pair, initializes from scratch with seed 42, and selects `best.pt` on the original test split. No sweep checkpoint is used as initialization. `last.pt` is retained separately. The polynomial LR schedule has exponent 0.9 and uses the actual total optimizer-step budget for each run.

Twenty epochs provide a bounded initial screen. Candidate rankings can change at 80 epochs, and identical seeds do not make the CUDA training trajectory fully deterministic. Formal results must be reported separately from sweep maxima. No claim of guaranteed improvement is made at launch.

## Execution and evidence

The existing GPU-memory queue is reused for GPUs **4-7**, allowing shared GPUs when at least 12,288 MiB is free. It retains the 60-second startup reservation and does not stop other processes. At most eight queue workers execute simultaneously. Training runs under tmux and is independent of the SSH or Codex session. No ongoing Codex polling or scheduled monitor is configured.

- Runtime code revision: `c25b9b7`.
- Entrypoint: [run_per_domain_sweep4.py](../run_per_domain_sweep4.py).
- Tmux session: `independent-af-sweep4`.
- Output root: `/data_nas/jiangsuiyang/ScribbleCL/independent_AF_sweep4_lr_global20_formal80_20260907`.
- Coordinator log: the output-root path followed by `_coordinator.log`.
- Exact launch script and final exit code: sibling `_launch.sh` and `_coordinator.exitcode`.
- Runs: `A_s0` through `F_s3`, then `A_formal80` through `F_formal80`.
- Phase/results: `pipeline.json`, `sweep_results.json`, and `selection.json`.

The NAS mount and free capacity were checked, followed by a small create/write/read/remove probe. Six input slice counts and annotation-file availability passed. Checkpoints are expected to occupy roughly 6 GiB; at least 20 GiB of free capacity was required before launch. All runtime output stays under the NAS project directory.

The existing queue self-check passed after reuse. The new coordinator's self-check verifies exactly four candidates for every domain, separate winners per domain, all 24 candidates finishing before any formal run, forwarding each selected pair into its own formal run, and blocking formal launch after a failed candidate. It uses simulated training, so it adds no sweep experiments. Code syntax and whitespace checks passed.

At closeout, all 30 summaries/manifests were checked against their expected epoch/step counts, zero exit codes, demo-selection labels, result metrics, and selected parameters. All 60 best/last checkpoint files were present and nonempty. Metrics were exported without re-evaluation or file hashing. Source, configurations, aggregate metrics, curves, and this report are published; data, annotations, model checkpoints, credentials, and full runtime logs remain excluded from the public repository. Model files stay under the NAS output root above.
