# Organ-CL comparison launch

Launched on jiangsuiyang at **2026-09-10 23:02:22 Asia/Shanghai**, in detached tmux session `r0910`. The user authorized GPUs **4, 5, 6, 7**, requested maximum use of these devices, and requested that the conversation end after preparation, launch and a brief startup check. No recurring monitor is requested or created.

The provisional reporting result is the completed `spatial0` branch of `organ_T4_from_T3best_lr006_spatial_pair10_20260910`: final foreground per-case macro test Dice **0.7047062025784092**, validation Dice **0.6957174162739166**. Its final test vector is **[0.6342617788087944, 0.5728554590666434, 0.7921885215378445, 0.8195190509003548]**. This is a single-seed, exploratory campaign result accepted for provisional reporting. Its T1/T2 training used Spatial=0.01 starting at epoch 30; Spatial=0 only describes its later T3/T4 continuation. Existing models and outputs are preserved.

## Comparison scope

Five fresh T1→T4 sequences reuse the comparison rows in the existing Domain-CL table (`output/ScribbleCL.tex`): **Dense-Sequential, PCE-Sequential, ZS-Sequential, ZS-EWC and ZS-GPM**. Organ uses the existing shared U-Net backbone and task-specific binary heads, not Domain's single shared head. No MiB, new sweep, main-method rerun or unrelated experiment is added.

- Shared training data: T1/T3/T4 fixed half subsets, T2 full; **381/166/711/982** training slices. Validation/test splits remain **153/298, 52/100, 343/496, 894/1223** slices.
- Per-task budgets **60/60/10/10 epochs**, initial learning rates **0.03/0.03/0.06/0.06**; each task uses its own polynomial horizon, exponent 0.9. Seed 42, stage transition seed 42+stage, batch 4, workers 8, SGD momentum 0.9, weight decay 1e-4.
- ZS controls use PCE/Global=1/1 and Spatial **0.01/0.01/0/0**, first active at epoch 30 for T1/T2. PCE/Dense use only cross-entropy. Dense reads dense labels for the same training images and is a distinct supervision reference.
- All controls receive the same non-replay transition controls: T2 clean-training-image head BN calibration and gradient clip 5; T3/T4 backbone/head LR ratio 0.1, fixed backbone BN running statistics and gradient clip 5. Historical heads remain frozen. These are explicitly adapted Organ controls, not unmodified Domain runs.
- EWC uses the existing sparse Fisher path: lambda 1, gamma 0.1, 50 batches. GPM uses 16 examples, threshold 0.97 plus 0.001 per task, 4096 patches/layer and 4,000,000 matrix elements. GPM weight decay is applied before gradient projection.
- Select checkpoints using current-task **foreground validation Dice**, once per epoch, then evaluate all seen tasks. Never select a checkpoint by test performance. Report foreground and background-inclusive metrics separately; the present training output is foreground Dice.

`runner_core.py` adds a fixed `--organ-report-schedule` option to the existing loop and reuses the Class dense-loader pattern. Existing scalar schedules and restore paths remain available. `main.py --setting-run` remains the training entry. The launcher transports arguments in an environment variable and deploys under neutral filenames: the coordinator, trainers and forked workers start with `./main -u run.py`; the interpreter symlink targets the existing Python environment. No new environment or dependency was installed.

The existing memory-aware queue selects among GPUs 4–7 with **22,000 MiB free** required at admission, leaving headroom for the later Spatial phase. Four jobs can run at once; the fifth follows available capacity. Existing GPU processes are preserved. Long-running work is detached from SSH and Codex.

## Checks and paths

Passed: dense versus sparse target semantics with and without H5 caching; fixed budget routing and fresh initialization; existing retention/head-freezing and evaluation state/RNG checks; synthetic **four-stage CUDA runs for Dense, EWC and GPM**, including Spatial execution, Fisher consolidation and nonempty GPM projection. The short GPU checks use two synthetic images/task and reduced GPM representation limits; they are implementation checks, not performance results.

The actual output mount is NFS `/data_nas` with about 29 TB free. A small write/fsync/read probe and a short-path AF_UNIX probe passed. Data and sparse shapes were checked once. Data are reused through NAS links; no hashes or duplicate datasets were created.

The initial 23:00:34 launch exited before training because the prior continuation's source snapshot omitted `main.py`. Both `main.py` and `runner.py` were copied from the existing local Organ implementation, and the exact neutral runtime entry passed `--setting-run --help`. All initial failures are preserved under `checks/missing_entry_failure`; no trained epoch or model was lost. The 23:02:22 launch is the effective run.

At **23:03:39**, startup passed: ZS-GPM/GPU4, ZS-EWC/GPU5 and ZS-Sequential/GPU6 had completed 97/97/98 updates, and Dense/GPU7 had completed 117 updates. All four had entered epoch 2 and their latest numerical status was `post_step_finite`. PCE-Sequential was queued. The 37-process coordinator/trainer/worker tree had neutral command lines; NVIDIA displayed `./main` for all four trainers, using approximately 7.45/7.45/7.45/4.72 GB. No existing process was stopped. The conversation ends after this startup check; this is not a completion claim.

Remote root: `/data_nas/jiangsuiyang/ScribbleCL/organ_comparisons_20260910`.

- Source: `source/`; private reference receipt: `report_reference.json`.
- Plan/status: `plan.json`, `progress.json`, `launch_receipt.json`, `startup_check.json`.
- Logs: `logs/<method>.log`; outputs: `runs/<method>/`.
- Check evidence: `checks/passed.json`, `preflight.json`.
- Completion: per-job exit codes, exact epoch coverage and finite losses, four selected paired stage checkpoints and readable summaries. The coordinator writes `comparison.json` and `pipeline.exitcode` when the queue ends.

RMA is left unset in the initial comparison export: existing 80-epoch independent references have a different budget and cannot silently become matched 60/60/10/10 references. GPM DRR also remains unset pending the representation-count convention; non-replay Sequential/EWC DRR is zero. Do not fabricate missing metrics. On the next completion query, verify results, finish the applicable aggregate reporting and publish source plus shareable results to the public project repository. Patient-level results, raw data, source IDs, full traces and checkpoints stay private on NAS. This launch is not completed experiment delivery.
