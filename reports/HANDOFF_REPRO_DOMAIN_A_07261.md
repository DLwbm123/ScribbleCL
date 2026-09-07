# Handoff: reproduce Domain A foreground Dice 0.7261

## Single objective

Reproduce the **foreground-only test Dice `0.7261413306722405`** obtained on Domain A after stage 1 of the ZS-GPM continual run `m7v2q` (seed 42). Do not optimize background-inclusive Dice and do not substitute an independent-training result.

## Target evidence

- Public result: `results/domain_zs_gpm/summary.json`, `matrix[0][0]`.
- Server result directory: `/home/jiangsuiyang/q1d7f/runs/m7v2q`.
- Existing target checkpoint: `/home/jiangsuiyang/q1d7f/runs/m7v2q/s01.pt`.
- Best validation: `0.7049937620651452`, epoch 128, iteration 9800.
- Test Dice by patient: `0.8250721886456398`, `0.74915653620881`, `0.6041952671622719`.
- Test prediction foreground fraction: `0.0211731441437252`.

The target is the A-to-A diagonal entry immediately after learning A. It is not the final six-domain A-Dice of ZS-GPM (`0.3246786311`).

## Exact recorded protocol

| Field | Value |
|---|---|
| Scenario/method | Domain-CL, `zs-gpm` |
| Task | A only for the reproduction |
| Supervision | frozen seed-42 scribbles |
| Seed | 42 |
| Epochs | 150 |
| Batch size | 4 |
| Learning rate | 0.03, polynomial decay |
| Optimizer | SGD, momentum 0.9 |
| PCE/global/spatial weights | 1.0 / 1.0 / 0.0 |
| Validation interval | 200 iterations |
| GPM threshold | 0.97 |
| GPM threshold step | 0.001 |
| GPM examples | 16 |
| GPM max patches/layer | 4096 |
| GPM max matrix elements | 4,000,000 |
| Training samples/batches | 301 slices, 76 batches/epoch, 11,400 iterations |

Recorded server environment currently available at `/home/jiangsuiyang/anaconda3/envs/py38/bin/python`: Python 3.10.6, PyTorch 2.2.1+cu121, NumPy 1.26.4, h5py 3.16.0, SciPy 1.13.0. Reuse this environment; do not reinstall PyTorch.

## Critical protocol warning

Do **not** use current `main` for the first reproduction. Commit `6b6ff65` changed the shared evaluator to include background, which changes validation checkpoint selection and `benchmark_mean`. Use public source commit `2cdb1bec5939d8b6b2399413434b8b3aaa9ea7c2`, the last canonical source before that metric change. Its continual training path matches the recorded foreground-only protocol.

The original `m7v2q` manifest did not record a Git commit, so the exact training-source SHA is unavailable. The pinned commit above is the strongest preserved source candidate, not a claim of byte-identical provenance.

## Server paths

```text
Python:       /home/jiangsuiyang/anaconda3/envs/py38/bin/python
Data root:    /home/jiangsuiyang/medical_continual_segmentation_domain_fastlane/data
Sparse root:  /home/jiangsuiyang/medical_continual_segmentation_domain_fastlane/data/sparse_annotations/domain
Target ckpt:  /home/jiangsuiyang/q1d7f/runs/m7v2q/s01.pt
New outputs:  /data_nas/jiangsuiyang/ScribbleCL/repro_domain_A_07261_seed42_<timestamp>
```

Keep the source checkout small under `/home`; all new checkpoints and logs must be written under `/data_nas`.

## Execution order

### 1. Create an isolated pinned checkout

Do not modify `/home/jiangsuiyang/q1d7f` or overwrite `m7v2q`.

```bash
git clone https://github.com/DLwbm123/ScribbleCL.git /home/jiangsuiyang/ScribbleCL_repro_07261
cd /home/jiangsuiyang/ScribbleCL_repro_07261
git checkout 2cdb1bec5939d8b6b2399413434b8b3aaa9ea7c2
```

If that checkout already exists, verify its current commit instead of cloning again.

### 2. Gate 0: replay the preserved checkpoint

Run this before retraining. It must recover the target metric from the existing `s01.pt` using the pinned foreground-only evaluator.

```bash
cd /home/jiangsuiyang/ScribbleCL_repro_07261
CUDA_VISIBLE_DEVICES=4 /home/jiangsuiyang/anaconda3/envs/py38/bin/python - <<'PY'
from pathlib import Path
import torch
from runner_core import TASKS, H5Slices, _build_model, _loader, evaluate

device = torch.device("cuda:0")
task = TASKS["domain"][0]
data_root = Path("/home/jiangsuiyang/medical_continual_segmentation_domain_fastlane/data")
model = _build_model("domain").to(device)
model.activate_stage(0)
model.load_state_dict(torch.load(
    "/home/jiangsuiyang/q1d7f/runs/m7v2q/s01.pt",
    map_location=device,
))
test = H5Slices(data_root / task.folder / task.filename, "test")
score = evaluate(model, _loader(test, 4, False, 0, 0), test.ends, device, None, task.classes)
test.close()
print(score)
assert abs(score["benchmark_mean"] - 0.7261413306722405) < 1e-8
PY
```

If Gate 0 fails, stop: the mismatch is evaluation/data/source related, and retraining would not be interpretable.

### 3. Train only stage A through the ordinary continual path

Use one free GPU; GPU 4 is shown below only as the preferred default. Confirm it is idle first.

```bash
run_root=/data_nas/jiangsuiyang/ScribbleCL/repro_domain_A_07261_seed42_$(date +%Y%m%d_%H%M%S)
tmux new-session -d -s repro-domain-a-07261 "cd /home/jiangsuiyang/ScribbleCL_repro_07261 && \
CUDA_VISIBLE_DEVICES=4 /home/jiangsuiyang/anaconda3/envs/py38/bin/python -u main.py --setting-run \
  --data-root /home/jiangsuiyang/medical_continual_segmentation_domain_fastlane/data \
  --sparse-root /home/jiangsuiyang/medical_continual_segmentation_domain_fastlane/data/sparse_annotations/domain \
  --output $run_root --device cuda:0 --seed 42 --max-task 1 \
  --epochs-per-task 150 --batch-size 4 --lr 0.03 --workers 8 \
  --validate-every 200 --method zs-gpm --pce-loss-weight 1.0 \
  --zs-global-weight 1.0 --zs-spatial-loss-weight 0.0 \
  --gpm-threshold 0.97 --gpm-threshold-step 0.001 --gpm-examples 16 \
  --gpm-max-patches-per-layer 4096 --gpm-max-matrix-elements 4000000 \
  > ${run_root}_coordinator.log 2>&1"
```

`--max-task 1` preserves the stage-A training path while avoiding unnecessary B--F training.

## Completion and success gates

Require all of the following:

1. The tmux session and scoped Python process have exited normally; no traceback or non-finite/CUDA OOM error appears.
2. `train.jsonl` reaches epoch 149 and iteration 11,400.
3. `s01_best.pt`, `s01.pt`, `summary.json`, `stages.json`, and `matrix.csv` exist and are readable.
4. `summary.json -> matrix[0][0]` is foreground-only Domain-A test Dice.
5. Exact reproduction target: `abs(matrix[0][0] - 0.7261413306722405) < 1e-8`.
6. If GPU nondeterminism prevents exact equality, record it as a near reproduction only when Dice is within `0.7261 ± 0.02`; do not relabel it exact.

## If retraining misses the target

Do not start a coefficient sweep immediately.

- Gate 0 passes but retraining fails: compare RNG/determinism, augmentation order, worker count, environment, and the missing original source-SHA provenance.
- Gate 0 fails: resolve evaluator, data split, checkpoint compatibility, or code revision first.
- Do not use `_run_independent_references`; the two completed independent A runs (`0.2565` tuned and `0.0761` matched-loss) used a different execution path and are not reproductions of `m7v2q` stage A.

## Closeout

After completion, publish the exact command/config, compact log summary, foreground result, validation-selected epoch, and comparison with `0.7261413306722405` to the public `DLwbm123/ScribbleCL` repository. Keep checkpoints, full logs, data, sparse annotations, and credentials off GitHub.
