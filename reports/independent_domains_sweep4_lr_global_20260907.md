# Four-candidate per-domain LR/global sweep, then selected formal training

Status: **launched at 2026-09-07 22:00:32 Asia/Shanghai**. The user requested an independent sweep for every domain with at most four sweep experiments per domain, followed by formal training using each domain's selected parameters. This batch contains exactly **24 new 20-epoch sweep runs plus 6 fresh 80-epoch formal runs**. Earlier experiments are preserved.

The previously authorized platform-demo protocol continues: each domain's original test split is the selection split. All scores are explicitly labeled demo selection scores and are not held-out test estimates. This batch does not use a shared cross-domain winner.

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

This report documents an active batch. Completed metrics will be checked and published when completion is subsequently queried. Data, annotations, model checkpoints, credentials, and full runtime logs are not included in the public repository.
