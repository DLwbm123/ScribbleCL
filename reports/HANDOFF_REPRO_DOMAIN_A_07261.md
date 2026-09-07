# Handoff: tune Domain-A independent training to 0.7261

## Objective

Improve the **scribble-supervised independent Domain-A experiment** until its foreground-only test Dice reproduces or approaches **`0.7261413306722405`** with seed 42. The historical ZS-GPM stage-A run `m7v2q` is an executable reference for the model, optimizer, loss, data order, validation, and evaluation code. It is not the final experiment label.

Do not optimize the test split. Tune and select checkpoints using foreground validation Dice; evaluate the test split once after configuration selection. Continue reporting both foreground and background-inclusive Dice.

## Why ZS-GPM stage A is the right reference

The target is `results/domain_zs_gpm/summary.json -> matrix[0][0]`. At stage A:

- no previous task has been learned, so there is no cross-domain warm start;
- GPM gradient projection is applied only when `stage > 0`, so it does not modify A training;
- the GPM representation is collected only after the best A checkpoint is selected;
- therefore the A result is effectively a strong independent Domain-A run through the ordinary continual training path.

The main useful difference is that the ZS-GPM path is the known-good training implementation. The current `_run_independent_references` path is a separate duplicated loop and has not reproduced its behavior.

## Target evidence

- Public result: `results/domain_zs_gpm/summary.json`, `matrix[0][0]`.
- Server run: `/home/jiangsuiyang/q1d7f/runs/m7v2q`.
- Existing A checkpoint: `/home/jiangsuiyang/q1d7f/runs/m7v2q/s01.pt`.
- Foreground test Dice: `0.7261413306722405`.
- Best validation Dice: `0.7049937620651452`, epoch 128, iteration 9800.
- Test patient Dice: `0.8250721886456398`, `0.74915653620881`, `0.6041952671622719`.
- Test prediction foreground fraction: `0.0211731441437252`.

This is the A-to-A current-task score after training A, not the final six-stage ZS-GPM A-Dice.

## Failed independent references to retain as negative controls

| Run | Configuration | A foreground Dice |
|---|---|---:|
| PCE independent | PCE only, 80 epochs | 0.0991 |
| tuned ZS independent | PCE/global/spatial = 0.5/1/0.05, 80 epochs | 0.2565 |
| matched-loss ZS independent | PCE/global/spatial = 1/1/0, 80 epochs | 0.0761 |

The last run showed that disabling spatial loss alone does not solve the problem. It still differed from the ZS-GPM A reference in training length and optimizer behavior: the independent loop used 80 epochs and weight decay `1e-4`, while ZS-GPM used 150 epochs and weight decay `0`.

## Exact reference protocol

| Field | Value |
|---|---|
| Scenario/task | Domain-CL, A |
| Method used as reference | `zs-gpm`, stage A only |
| Supervision | seed-42 scribbles |
| Seed | 42 |
| Epochs | 150 |
| Batch size | 4 |
| Learning rate | 0.03, polynomial decay |
| Optimizer | SGD, momentum 0.9, weight decay 0 |
| PCE/global/spatial | 1.0 / 1.0 / 0.0 |
| Validation interval | 200 iterations |
| GPM threshold/step/examples | 0.97 / 0.001 / 16 |
| GPM patch/matrix limits | 4096 / 4,000,000 |
| Training size | 301 slices, 76 batches/epoch, 11,400 iterations |

Reuse `/home/jiangsuiyang/anaconda3/envs/py38/bin/python`: Python 3.10.6, PyTorch 2.2.1+cu121, NumPy 1.26.4, h5py 3.16.0, and SciPy 1.13.0. Do not reinstall PyTorch or create another environment.

## Server and storage

```text
SSH host:     10.12.208.180
SSH user:     jiangsuiyang
SSH port:     22
Python:       /home/jiangsuiyang/anaconda3/envs/py38/bin/python
Data root:    /home/jiangsuiyang/medical_continual_segmentation_domain_fastlane/data
Sparse root:  /home/jiangsuiyang/medical_continual_segmentation_domain_fastlane/data/sparse_annotations/domain
Target ckpt:  /home/jiangsuiyang/q1d7f/runs/m7v2q/s01.pt
New outputs:  /data_nas/jiangsuiyang/ScribbleCL/tune_independent_A_07261_seed42_<timestamp>
```

Connect with `ssh -p 22 jiangsuiyang@10.12.208.180`. The password is supplied out of band and must not be written into code, logs, reports, or GitHub. Source checkouts may remain under `/home`; all new checkpoints and complete logs go under `/data_nas`.

## Metric and code-version warning

Commit `6b6ff65` changed the shared evaluator to include background, altering `benchmark_mean` and checkpoint selection. For the initial parity run, use public commit `2cdb1bec5939d8b6b2399413434b8b3aaa9ea7c2`, whose ordinary continual path uses foreground-only Domain Dice like `m7v2q`.

The original manifest did not record a source SHA. This commit is the strongest preserved source candidate, not proof of byte-identical provenance. After parity is established, port the minimal independent fix to current `main`, keeping explicit foreground checkpoint selection and dual-metric reporting.

## Required execution order

### 1. Create an isolated pinned checkout

Do not modify `/home/jiangsuiyang/q1d7f` or overwrite `m7v2q`.

```bash
git clone https://github.com/DLwbm123/ScribbleCL.git /home/jiangsuiyang/ScribbleCL_repro_07261
cd /home/jiangsuiyang/ScribbleCL_repro_07261
git checkout 2cdb1bec5939d8b6b2399413434b8b3aaa9ea7c2
```

If the checkout already exists, verify its current commit instead of cloning again.

### 2. Gate 0: replay the preserved checkpoint

This confirms that the pinned evaluator, data split, and stored checkpoint recover the target before any new training.

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

If Gate 0 fails, stop and resolve the evaluator, data, or source mismatch before training.

### 3. Run the ZS-GPM stage-A oracle

This is the first independent candidate because A has no earlier task and GPM does not project gradients at stage 0.

```bash
run_root=/data_nas/jiangsuiyang/ScribbleCL/tune_independent_A_07261_seed42_$(date +%Y%m%d_%H%M%S)
tmux new-session -d -s tune-independent-a-07261 "cd /home/jiangsuiyang/ScribbleCL_repro_07261 && \
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

Do not train B--F. The only target is a strong independent A result.

### 4. Make the independent implementation match the oracle

If the oracle reproduces `0.7261`, use it to correct the current independent path instead of launching a broad coefficient sweep.

Preferred minimal fix:

1. Route single-task independent ZS training through the same ordinary stage-training implementation used by ZS-GPM.
2. For an A-only run, preserve stage index 0, SGD weight decay 0, 150 epochs, loader seed 42, PCE/global/spatial `1/1/0`, and validation every 200 iterations.
3. Do not copy another training loop. The existing duplicated independent loop is the suspected source of protocol drift.
4. Add only a compact result adapter if `independent_scores.json` is required by downstream RMA calculation.

Before a full 150-epoch rerun, compare one fixed batch and one optimizer step between the independent path and the stage-A oracle: initial state, total/PCE/global loss, gradients, updated parameters, and learning rate must agree within numerical tolerance. This is a diagnostic parity check, not a new training metric.

### 5. Tune only if exact parity remains below target

Use validation only. Start from the `1/1/0`, LR 0.03, weight-decay 0 anchor. Change one small group at a time; do not reintroduce spatial loss until the oracle-matched baseline works. A broad A--F sweep is out of scope until A reaches the target range.

## Completion and success gates

1. Gate 0 reproduces the existing checkpoint score exactly.
2. The new A run reaches epoch 149 and iteration 11,400 and writes the best/final checkpoint and summary artifacts.
3. Primary metric is foreground-only Domain-A test Dice.
4. Exact target: `abs(Dice - 0.7261413306722405) < 1e-8`.
5. If nondeterministic GPU kernels prevent exact equality, label the result a near reproduction only when Dice is within `0.7261 ± 0.02`.
6. The final accepted run must use the independent/single-task interface or be explicitly documented as the stage-A single-task oracle; do not mix it with B--F training.

## Closeout

Publish the minimal code change, exact command/config, compact log summary, foreground and inclusive Dice, validation-selected epoch, and comparison with `0.7261413306722405` to the public `DLwbm123/ScribbleCL` repository. Keep credentials, data, sparse annotations, checkpoints, and full logs off GitHub.
