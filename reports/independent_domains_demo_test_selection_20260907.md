# A-F independent training for platform demonstration

Status: **complete at 2026-09-07 20:54:23 Asia/Shanghai**, following launch at 18:19:36. All six domains completed 80 epochs; all six training exit codes and the coordinator exit code were 0. The user explicitly requested fresh training on all six domains with the original test split used as validation to select the best checkpoint, for platform demonstration rather than academic reporting. The user also authorized concurrent jobs whenever the authorized GPUs have sufficient free memory.

These scores are **demo selection scores on the original test split**, not held-out test estimates. The protocol is explicitly recorded as `result_usage=platform_demo`, `selection_split=test`, and `test_for_selection=true`. The final summary sets `held_out_test_evaluated=false`. Each record uses `best_selection`, not `best_validation`; the separate held-out `test` field is null. No unchanged-data held-out result is claimed.

## Completed results

| Domain | Foreground demo selection Dice | Background-inclusive Dice | Best epoch (one-based) | Best iteration |
|---|---:|---:|---:|---:|
| A | 0.7366443473 | 0.8657135338 | 22 | 1,600 |
| B | 0.5413882084 | 0.7647328573 | 58 | 2,400 |
| C | 0.6412886439 | 0.8160824342 | 44 | 3,800 |
| D | 0.4803634779 | 0.7374507131 | 62 | 2,600 |
| E | 0.6899351196 | 0.8406771360 | 74 | 10,800 |
| F | 0.6628073631 | 0.8293936480 | 64 | 11,200 |

The unweighted mean across the six domain foreground scores is **0.6254045267**; this is a mean of domain scores, not a pooled-patient score. A's new demo score 0.7366 exceeds 0.7261 but is below the preserved earlier validation-selected run's 0.7575. This fresh run does not replace or erase that earlier result.

Public evidence: [results CSV](../results/independent_domains_demo_test_selection_20260907/results.csv), [completed pipeline](../results/independent_domains_demo_test_selection_20260907/pipeline_complete.json), [per-domain manifests](../results/independent_domains_demo_test_selection_20260907/manifests.json), [exact training commands](../results/independent_domains_demo_test_selection_20260907/commands.json), and [periodic selection curves](../results/independent_domains_demo_test_selection_20260907/selection_curves.csv). The CSV and pipeline include final selected metrics, including final-validation candidates if applicable.

All six `best.pt` and six `last.pt` files were present at completion verification, each 103,132,993 bytes. For platform use, the selected model for each domain is `<output root>/<domain>/best.pt`; `last.pt` is the final training state and need not have the best selection score. Metrics were checked against each domain's completed summary and manifest; no additional test replay or hashing was performed.

## Training and selection

All A-F models are initialized afresh with seed 42 and trained independently for 80 epochs through the shared stage-training loop. This includes A; neither its prior model nor another domain's model is loaded. Training remains on each domain's original scribble-supervised train split. Only the checkpoint-selection split changes to the original test split.

Fixed settings: historical `sparse_annotations_pattern_f5_b10/domain` annotations, PCE/global/spatial weights 1/1/0.01, ten warmup epochs (spatial starts at epoch index 10, the 11th epoch), batch size 4, LR 0.03 with polynomial decay, SGD momentum 0.9, optimizer weight decay 0 plus manual gradient decay 1e-4, 8 workers, and OMP_NUM_THREADS=4. Foreground Dice on the original test split is evaluated every 200 optimizer steps and at the end. `best.pt` retains the greatest score among these checkpoints; `last.pt` retains the final model. There is no additional held-out evaluation after selection.

## GPU scheduling and artifacts

Only GPUs 4, 5, 6, and 7 are used. The queue admits a job when at least 12,288 MiB is free, independently of GPU utilization or existing jobs. Existing runs have used approximately 10 GB per job. A 60-second per-GPU startup reservation prevents admitting another job before CUDA's initial allocations become visible. The short smoke run successfully shared GPU 4 with an existing training process. Free memory is checked again before each queued launch; jobs continue to share compute as long as the memory gate permits admission.

Verified assignment: A and E on GPU 5, B on GPU 6, C and F on GPU 7, D on GPU 4. E and F started when sufficient memory was available, sharing their GPUs with already-running jobs. Existing standard-protocol B-F jobs and their artifacts were retained and were not terminated. Demo manifests were checked for `selection_split=test`, `test_for_selection=true`, and 80 completed epochs.

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

Source code, configurations, compact completed metrics, selection curves, and this report are published. Data, annotations, checkpoints, full runtime logs, and credentials are excluded from GitHub. Checkpoints remain in the server output directory for platform use.
