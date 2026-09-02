# ScribbleCL Domain-CL reproduction

This repository contains the public, data-free code and compact outputs for a
six-domain weakly supervised continual segmentation experiment. The shared
model is the ZScribbleSeg U-Net with one binary output head.

The canonical joint-training implementation is
`main.py --setting-run --method zs-joint`. A seed-42 short convergence study reached **0.5662 mean validation
Dice** and **0.5628 mean A-F test Dice** after five epochs with batch size 4 and
learning rate 0.04.

The public package also preserves earlier continual and standalone diagnostic
runs:

- Domain-CL with ZS-GPM
- Domain-CL with ZS-DER++
- two legacy standalone joint-domain runs

See [RESULTS.md](RESULTS.md), the [short convergence
report](reports/joint_short_convergence_20260902.md), and the [code-versus-data
diagnosis](reports/code_vs_data_diagnosis_20260902.md) for measured results and
interpretation.

## Data contract

Data and sparse annotations are not distributed here. The runner expects:

```text
<data-root>/Domain_Prostate/{BIDMC,HK,ISBI,UCL,ISBI_1.5,I2CVB}.h5
<sparse-root>/{A,B,C,D,E,F}_v2_s2_seed42.npz
```

Each HDF5 file must provide `train_images`, `val_images`, `test_images`, the
corresponding labels, and patient-boundary arrays used by the evaluator. Each
sparse archive contains one `annotations` array. Do not commit these inputs.

## Environment

The validated joint run used Python 3.10 and the versions pinned in
`requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Continual runs

ZS-GPM:

```bash
python -u main.py --setting-run \
  --data-root <data-root> --sparse-root <sparse-root> \
  --output runs/domain_zs_gpm --device cuda:0 --seed 42 \
  --epochs-per-task 150 --batch-size 4 --lr 0.03 --workers 8 \
  --validate-every 200 --method zs-gpm --zs-global-weight 1.0 \
  --gpm-threshold 0.97 --gpm-threshold-step 0.001 --gpm-examples 16
```

ZS-DER++:

```bash
python -u main.py --setting-run \
  --data-root <data-root> --sparse-root <sparse-root> \
  --output runs/domain_zs_derpp --device cuda:0 --seed 42 \
  --epochs-per-task 150 --batch-size 4 --lr 0.03 --workers 8 \
  --validate-every 200 --method zs-derpp --zs-global-weight 1.0 \
  --der-buffer-size 64 --der-minibatch-size 8 \
  --der-alpha 0.5 --der-beta 0.5
```

## Canonical joint-domain reference

`main.py --setting-run --method zs-joint` pools all A-F training slices into
one dataset. Model selection uses the equal mean of the six validation-domain
Dice scores, and the final report retains one `joint` performance-matrix row.
`--max-task` is intentionally rejected for this method.

```bash
python -u main.py --setting-run \
  --data-root <data-root> --sparse-root <sparse-root> \
  --output runs/zs_joint_b4_lr004 --device cuda:0 --seed 42 \
  --epochs-per-task 5 --batch-size 4 --lr 0.04 --workers 4 \
  --validate-every 567 --method zs-joint --zs-global-weight 1.0
```

The removed standalone `zs_joint_domain.py` duplicated the canonical path and
was the source of divergent experiments. Use only the command above for joint
training. The five-epoch result is an implementation/convergence check, not a
multi-seed final upper-bound estimate.

## Public-release scope

Published: source, run manifests, compact summaries, stage summaries, and
metric matrices. Excluded: HDF5/NPZ data, patient inputs, model weights,
checkpoints, caches, and raw runtime directories.
