# ScribbleCL Domain-CL reproduction

This repository contains the public, data-free code and compact outputs for a
six-domain weakly supervised continual segmentation experiment. The shared
model is the ZScribbleSeg U-Net with one binary output head.

The public package covers four completed runs:

- Domain-CL with ZS-GPM
- Domain-CL with ZS-DER++
- joint-domain ZS training with batch size 2 and learning rate 0.015
- joint-domain ZS training with batch size 4 and learning rate 0.03

See [RESULTS.md](RESULTS.md) for the measured results and interpretation.

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

The completed runs used Python 3.12 and the versions pinned in
`requirements.txt` on NVIDIA A100 40 GB GPUs.

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

## Joint-domain reference

`main.py --setting-run --method zs-joint` pools all A-F training slices into
one dataset. Model selection uses the equal mean of the six validation-domain
Dice scores, and the final report retains one `joint` performance-matrix row.
`--max-task` is intentionally rejected for this method.

```bash
# Configuration 1
python -u main.py --setting-run \
  --data-root <data-root> --sparse-root <sparse-root> \
  --output runs/zs_joint_b2_lr0015 --device cuda:0 --seed 42 \
  --epochs-per-task 150 --batch-size 2 --lr 0.015 --workers 8 \
  --validate-every 2000 --method zs-joint --zs-global-weight 1.0

# Configuration 2
python -u main.py --setting-run \
  --data-root <data-root> --sparse-root <sparse-root> \
  --output runs/zs_joint_b4_lr003 --device cuda:0 --seed 42 \
  --epochs-per-task 150 --batch-size 4 --lr 0.03 --workers 8 \
  --validate-every 1000 --method zs-joint --zs-global-weight 1.0
```

The joint protocol is an intended reference for upper-bound analysis, but the
two configurations reported here did not numerically exceed ZS-GPM. They
must not be described as an empirical upper bound without further tuning.

## Public-release scope

Published: source, run manifests, compact summaries, stage summaries, and
metric matrices. Excluded: HDF5/NPZ data, patient inputs, model weights,
checkpoints, caches, and raw runtime directories.
