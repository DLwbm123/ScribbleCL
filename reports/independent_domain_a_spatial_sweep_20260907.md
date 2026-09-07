# Independent Domain-A spatial-loss sweep: 20 epochs, then 80 epochs

Status: **sweep launched at 2026-09-07 14:37:52 CST** in tmux session `independent-a-spatial-sweep`. The first two candidates have completed epoch 0 / iteration 76 on GPUs 6 and 7, with finite losses and no test evaluation. The coordinator will select after all six 20-epoch candidates complete and automatically start the 80-epoch formal run. No final result is claimed yet.

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

```bash
cd /home/jiangsuiyang/ScribbleCL_independent_A_20260907
OMP_NUM_THREADS=4 /home/jiangsuiyang/anaconda3/envs/py38/bin/python -u \
  run_independent_a_spatial_sweep.py \
  --data-root /home/jiangsuiyang/medical_continual_segmentation_domain_fastlane/data \
  --sparse-root /home/jiangsuiyang/medical_continual_segmentation_domain_gptpro/data/sparse_annotations_pattern_f5_b10/domain \
  --output /data_nas/jiangsuiyang/ScribbleCL/independent_A_spatial_sweep20_formal80_20260907
```

The original source revision is `530d93c3da927f9daf5d855fcf58a4a73ea529c8`; the GPU-4 continuation changes the coordinator only. Its revision is recorded in `pipeline.json`, alongside the original source revision. The output must not already exist. `pipeline.json` records the source revision, fixed protocol, phase, and completion or failure. Every candidate has a command record, complete log, exit code, train/validation log, best/last checkpoint, and compact result. The chosen configuration is recorded before formal training starts. Data, annotations, checkpoints, and full runtime logs remain on experiment storage; only code and compact public-safe metrics and reports are published.

This is a one-seed, short-budget hyperparameter search. A 20-epoch winner may not remain the strongest configuration after 80 epochs, and the prior aligned run showed that stronger validation performance does not guarantee the target test Dice. No test-based re-selection or target achievement is assumed.

## Adding GPU 4 without restarting active training

The original coordinator was stopped with SIGSTOP while its two training children continued. The updated coordinator accepts `--gpus 6 7 4 --adopt-running sweep_s00=1856491 sweep_s01=1856492 --adopt-parent 1856487`. It validates each retained process identity, waits for its exit status through Linux `/proc` while the old coordinator is stopped, and applies the same completion checks. The two running candidates keep their training state, output directories, and GPU assignments. GPU 4 starts the previously unlaunched 0.05 candidate, then 1.0.

The new tmux session is `independent-a-spatial-sweep-gpu4`; its coordinator log and exit-code marker use the suffix `_coordinator_gpu4`. The old coordinator is retired after the sweep completes; its eventual killed exit status denotes supersession, not a training failure. Selection still requires all six candidates, and the 80-epoch formal run remains on GPU 6. The Linux self-check verifies that adoption preserves and reads a nonzero child exit status. No training loop or candidate hyperparameter changed.
