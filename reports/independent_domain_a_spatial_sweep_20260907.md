# Independent Domain-A spatial-loss sweep: 20 epochs, then 80 epochs

Status: **complete at 2026-09-07 15:43:39 Asia/Shanghai; target achieved**. All six 20-epoch candidates completed successfully. Validation selected spatial coefficient **0.01**, and the fresh 80-epoch independent Domain-A run achieved **foreground test Dice 0.7575264500**, exceeding the historical target 0.7261413307 by **0.0313851194 (3.14 percentage points)**. Formal training and the active coordinator both exited with code 0.

The user authorized a spatial-loss hyperparameter sweep on Domain A followed by formal training of the best configuration. The target is foreground test Dice around or above 0.7261413307. The initial launch used GPUs 6 and 7. The user subsequently authorized GPU 4; the three-GPU continuation uses GPUs 6, 7, and 4.

## Fixed protocol and candidates

Reuse the independent interface and shared stage-training implementation from `dd490ae`. Keep the corrected historical `sparse_annotations_pattern_f5_b10/domain` scribbles, seed 42, PCE/global weights 1/1, batch size 4, LR 0.03 with polynomial decay, SGD momentum 0.9, optimizer weight decay 0 plus manual gradient decay 1e-4, 8 workers, and `OMP_NUM_THREADS=4`. No old checkpoint is loaded.

| Candidate | Spatial coefficient | GPU | Sweep epochs |
|---|---:|---:|---:|
| sweep_s00 | 0 | 6 | 20 |
| sweep_s01 | 0.01 | 7 | 20 |
| sweep_s02 | 0.05 | 4 | 20 |
| sweep_s03 | 0.1 | 6 | 20 |
| sweep_s04 | 0.3 | 7 | 20 |
| sweep_s05 | 1.0 | 4 | 20 |

Each GPU runs its two candidates sequentially, with one training process per GPU. Each sweep run has 301 training slices, 76 batches per epoch, and 1,520 optimizer steps. Validation occurs every 200 steps plus the final validation.

The runner's default spatial warmup would disable spatial loss for every epoch in a 20-epoch sweep. Set `--zs-spatial-warmup-epochs 4`: the existing condition is `epoch > 4`, so spatial loss starts at zero-based epoch 5, after five warmup epochs. Positive-weight candidates have 15 spatial-active epochs. Keep this same five-epoch warmup in formal training; do not silently change the chosen schedule between sweep and formal.

## Selection and automatic formal training

All six runs must exit successfully, finish 20 epochs, and produce validation-only results. The coordinator checks that spatial loss actually activated and was nonzero for positive-weight candidates. It ranks the six configurations by their **best foreground validation Dice**, breaking exact ties in favor of the smaller coefficient. Sweep runs use `--independent-skip-test`; no test metrics are available to the selector.

After writing `sweep_summary.json`, the coordinator automatically starts a **fresh seed-42, 80-epoch** independent Domain-A run on GPU 6, with the selected coefficient and otherwise unchanged controls. This is a new 6,080-step training schedule, not a continuation of the 20-epoch checkpoint. It selects its checkpoint by foreground validation Dice, then evaluates the selected checkpoint once on test and reports foreground and inclusive Dice. No B–F or other-scenario training is launched.

Training and ranking reuse the existing runner; the only new code is a small subprocess coordinator, [run_independent_a_spatial_sweep.py](../run_independent_a_spatial_sweep.py). Its `--self-check` verifies validation-only ranking, exact-tie handling, and rejection of incomplete candidate sets. The spatial-enabled smoke passed before launch: with six one-batch epochs and coefficient 1.0, spatial loss activated only at epoch index 5, produced raw spatial loss 0.8234404 and finite total loss 1.4170318, and did not evaluate test data. [Smoke evidence](../results/independent_domain_a_spatial_sweep_20260907/spatial_smoke.json).

## Reproduction and artifacts

### Completed results

| Spatial coefficient | Best foreground validation Dice (20 epochs) |
|---:|---:|
| 0 | 0.5116036280 |
| **0.01** | **0.6088808106** |
| 0.05 | 0.4794269843 |
| 0.1 | 0.5024610864 |
| 0.3 | 0.5220990595 |
| 1.0 | 0.5423627867 |

All sweep results are validation-only. The 0.1 candidate's best score came from the final validation at iteration 1,520; the exported curve includes this result.

| Formal run field | Result |
|---|---:|
| Completed epochs / optimizer steps | 80 / 6,080 |
| PCE / global / spatial weights | 1 / 1 / 0.01 |
| First spatial-active epoch | 6th (zero-based index 5) |
| Spatial-active epochs | 75 |
| Selected checkpoint epoch / iteration | 43rd (index 42) / 3,200 |
| Best foreground validation Dice | 0.6365137028 |
| Selected checkpoint foreground test Dice | **0.7575264500** |
| Selected checkpoint background-inclusive test Dice | 0.8764818904 |
| Formal elapsed time | 2,274.56 seconds (37.9 minutes) |

The formal run's best validation score also exceeded the sweep winner's 0.6088808106. Formal training restarted from initialization with an 80-epoch polynomial learning-rate schedule, so its intermediate trajectory need not match the 20-epoch sweep. Spatial activation was unchanged: the first five epochs disabled it, and the sixth enabled it.

Public evidence: [sweep summary](../results/independent_domain_a_spatial_sweep_20260907/sweep_summary.json), [formal metrics](../results/independent_domain_a_spatial_sweep_20260907/formal_summary.json), [completion and protocol](../results/independent_domain_a_spatial_sweep_20260907/pipeline_complete.json), [formal manifest](../results/independent_domain_a_spatial_sweep_20260907/formal_manifest.json), [exact formal command](../results/independent_domain_a_spatial_sweep_20260907/formal_command.json), and [validation curves](../results/independent_domain_a_spatial_sweep_20260907/validation_curves.csv).

The selected checkpoint remains at `/data_nas/jiangsuiyang/ScribbleCL/independent_A_spatial_sweep20_formal80_20260907/formal80/best.pt`; `last.pt` is retained alongside it. Both files were present at 103,132,993 bytes at completion verification. The runner evaluated the validation-selected checkpoint once on test; no additional test replay was performed for this report.

### Launch command

```bash
cd /home/jiangsuiyang/ScribbleCL_independent_A_20260907
OMP_NUM_THREADS=4 /home/jiangsuiyang/anaconda3/envs/py38/bin/python -u \
  run_independent_a_spatial_sweep.py \
  --gpus 6 7 4 \
  --data-root /home/jiangsuiyang/medical_continual_segmentation_domain_fastlane/data \
  --sparse-root /home/jiangsuiyang/medical_continual_segmentation_domain_gptpro/data/sparse_annotations_pattern_f5_b10/domain \
  --output /data_nas/jiangsuiyang/ScribbleCL/independent_A_spatial_sweep20_formal80_20260907
```

The original source revision is `530d93c3da927f9daf5d855fcf58a4a73ea529c8`; the GPU-4 continuation changes the coordinator only. Its revision is recorded in `pipeline.json`, alongside the original source revision. The output must not already exist. `pipeline.json` records the source revision, fixed protocol, phase, and completion or failure. Every candidate has a command record, complete log, exit code, train/validation log, best/last checkpoint, and compact result. The chosen configuration is recorded before formal training starts. Data, annotations, checkpoints, and full runtime logs remain on experiment storage; only code and compact public-safe metrics and reports are published.

This is a one-seed result, not a multi-seed robustness claim. Only the validation-selected candidate received formal 80-epoch training, so this does not establish that it would beat every candidate at that budget. The target was achieved for this run without test-based re-selection. The improvement over the prior 150-epoch baseline cannot be attributed solely to spatial loss: the training budget also changed, and CUDA training is not fully deterministic.

## Adding GPU 4 without restarting active training

The original coordinator was stopped with SIGSTOP while its two training children continued. The updated coordinator accepts `--gpus 6 7 4 --adopt-running sweep_s00=1856491 sweep_s01=1856492 --adopt-parent 1856487`. It validates each retained process identity, waits for its exit status through Linux `/proc` while the old coordinator is stopped, and applies the same completion checks. The two running candidates keep their training state, output directories, and GPU assignments. GPU 4 starts the previously unlaunched 0.05 candidate, then 1.0.

The new tmux session is `independent-a-spatial-sweep-gpu4`; its coordinator log and exit-code marker use the suffix `_coordinator_gpu4`. The old coordinator is retired after the sweep completes; its eventual killed exit status denotes supersession, not a training failure. Selection still requires all six candidates, and the 80-epoch formal run remains on GPU 6. The Linux self-check verifies that adoption preserves and reads a nonzero child exit status. No training loop or candidate hyperparameter changed.

Live verification confirmed three training processes on GPUs 4, 6, and 7. The retained 0 and 0.01 runs finished all 20 epochs and passed the adopted-exit checks before the next candidates started. For these two adopted runs, the original coordinator's `elapsed_seconds` values used a log modification time and are not reliable training durations; use command/manifest timestamps when exporting timing. The coordinator source now uses the command timestamp for future adoptions. This metadata correction does not affect candidate ranking.
