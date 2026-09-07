# A-F independent training for platform demonstration

Status: **launched 2026-09-07 18:19:36 Asia/Shanghai**. The user explicitly requested fresh training on all six domains with the original test split used as validation to select the best checkpoint, for platform demonstration rather than academic reporting. The user also authorized concurrent jobs whenever the authorized GPUs have sufficient free memory.

These scores are **demo selection scores on the original test split**, not held-out test estimates. The protocol is explicitly recorded as `result_usage=platform_demo`, `selection_split=test`, and `test_for_selection=true`. The final summary sets `held_out_test_evaluated=false`. Each record uses `best_selection`, not `best_validation`; the separate held-out `test` field is null. No unchanged-data held-out result is claimed.

## Training and selection

All A-F models are initialized afresh with seed 42 and trained independently for 80 epochs through the shared stage-training loop. This includes A; neither its prior model nor another domain's model is loaded. Training remains on each domain's original scribble-supervised train split. Only the checkpoint-selection split changes to the original test split.

Fixed settings: historical `sparse_annotations_pattern_f5_b10/domain` annotations, PCE/global/spatial weights 1/1/0.01, ten warmup epochs (spatial starts at epoch index 10, the 11th epoch), batch size 4, LR 0.03 with polynomial decay, SGD momentum 0.9, optimizer weight decay 0 plus manual gradient decay 1e-4, 8 workers, and OMP_NUM_THREADS=4. Foreground Dice on the original test split is evaluated every 200 optimizer steps and at the end. `best.pt` retains the greatest score among these checkpoints; `last.pt` retains the final model. There is no additional held-out evaluation after selection.

## GPU scheduling and artifacts

Only GPUs 4, 5, 6, and 7 are used. The queue admits a job when at least 12,288 MiB is free, independently of GPU utilization or existing jobs. Existing runs have used approximately 10 GB per job. A 60-second per-GPU startup reservation prevents admitting another job before CUDA's initial allocations become visible. The short smoke run successfully shared GPU 4 with an existing training process. Free memory is checked again before each queued launch; jobs continue to share compute as long as the memory gate permits admission.

Verified initial assignment: A on GPU 5, B on GPU 6, C on GPU 7, D on GPU 4. E and F are queued for sufficient memory on any authorized GPU. Existing standard-protocol B-F jobs and their artifacts are retained; they were not terminated. New demo manifests were checked for `selection_split=test`, `test_for_selection=true`, and 80 epochs. No completed A-F demo result is claimed at launch.

Runtime source revision: `ffea586`. Tmux session: `independent-af-demo80`. Output root: `/data_nas/jiangsuiyang/ScribbleCL/independent_AF_demo_test_selection80_20260907`. The sibling `_launch.sh` contains the exact command; `_coordinator.log` and `_coordinator.exitcode` record orchestration and termination. Each domain writes a command record, exit code, compact result, manifest, train/evaluation log, and best/last checkpoints.

```bash
OMP_NUM_THREADS=4 /home/jiangsuiyang/anaconda3/envs/py38/bin/python -u \
  run_independent_domains_formal.py \
  --data-root /home/jiangsuiyang/medical_continual_segmentation_domain_fastlane/data \
  --sparse-root /home/jiangsuiyang/medical_continual_segmentation_domain_gptpro/data/sparse_annotations_pattern_f5_b10/domain \
  --output /data_nas/jiangsuiyang/ScribbleCL/independent_AF_demo_test_selection80_20260907 \
  --warmup-epochs 10 --demo-test-selection --min-free-memory-mib 12288
```

Run from `/home/jiangsuiyang/ScribbleCL_independent_A_20260907`, with a new output path for any subsequent reproduction.

## Verification

[check_independent_demo.py](../check_independent_demo.py) passed a two-epoch, one-batch-per-epoch integration check on the actual training implementation. It verified that the evaluator reads A's three original test cases (not its two original validation cases), the summary selects the maximum observed foreground Dice, best/last checkpoint files are produced, the explicit demo labels are present, and spatial activates at the requested smoke-test boundary. Its output is diagnostic, not an A formal result. The existing queue self-check covers all six demo domains, standard-mode preservation, warmup forwarding, and failure reporting.

Source code and this public-safe launch report are published. Data, annotations, checkpoints, full runtime logs, and credentials are excluded from GitHub.
