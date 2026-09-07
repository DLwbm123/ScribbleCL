# Independent Domain-A: stage-loop alignment and corrected scribble protocol

Status: **both corrected runs completed 150 epochs / 11,400 iterations with exit code 0. Independent foreground test Dice is 0.6792972027; the 0.7261413307 target was not reached.**

## What was corrected

The Domain-A scribble-independent interface (`--method zs-sequential --independent-reference --independent-task 1`) now uses the ordinary shared stage-training loop. It starts from a fresh seed-42 model at stage 0, defaults to 150 epochs, selects by foreground validation Dice, and reports foreground and background-inclusive Dice separately. No second training loop, GPM projection, replay, or B–F training was added. Other independent scenarios retain their existing path.

The reference optimizer sets SGD weight decay to zero but adds `1e-4 * parameter` to gradients before the step, including stage A. The independent interface preserves this behavior. Thus optimizer weight decay alone does not explain the old performance gap.

## Critical handoff correction: different training scribbles

The original `m7v2q` launch invocation recorded on 2026-08-12 used `sparse_annotations_pattern_f5_b10/domain` from the source project's `medical_continual_segmentation_domain_gptpro/data`, together with `OMP_NUM_THREADS=4`. The handoff instead pointed at `medical_continual_segmentation_domain_fastlane/data/sparse_annotations/domain`.

The two existing Domain-A archives were opened read-only. Both have shape `(301, 256, 256)` and the same filename `A_v2_s2_seed42.npz`, but their annotation budgets differ:

| Existing protocol | Background labeled pixels | Foreground labeled pixels | Total labeled coverage |
|---|---:|---:|---:|
| fastlane `sparse_annotations/domain` | 101,136 | 29,726 | 0.663387% |
| historical `sparse_annotations_pattern_f5_b10/domain` | 1,087,492 | 142,866 | 6.237134% |

The initial two jobs using the handoff's incorrect root were stopped and retained as protocol-mismatch diagnostics. No annotations were generated, edited, or optimized. The corrected runs reuse the recovered historical protocol. Any improvement over the old sparse-root experiments must therefore not be attributed solely to the training-loop change.

Checkpoint replay cannot validate training scribbles: it only reads dense evaluation data. This explains why Gate 0 passed despite the incorrect sparse root.

## Verification

- Preserved checkpoint `m7v2q/s01.pt`: foreground test Dice **0.7261413306722405**, inclusive Dice **0.8605453027538156**. These are replayed historical scores, not a new trained result.
- One fixed, augmented batch from the corrected historical scribbles: preserved `2cdb1be` stage-A loop versus the new independent interface. Initial state, gradients, and updated state all had maximum absolute difference **0**. PCE/global/total loss were **0.7386324406 / 0.1788328290 / 0.9174652696** in both paths. Learning rate changed identically from **0.03** to **0.029997631568559224** on the 11,400-step schedule.
- The diagnostic also repeats the oracle step. Deterministic algorithm preferences are diagnostic-only, with warnings enabled; CUDA grid-sampler backward remains nondeterministic in this environment. Earlier stricter checks observed gradient differences of about `5e-6`. Full training retains historical CUDA defaults, so exact end-to-end reproducibility is not established by this check.
- A one-batch end-to-end smoke completed, wrote best/last checkpoints and a validation-only summary, and did not evaluate test data. Existing evaluator tests, extended to check the two explicit metrics, passed: **2 passed**.
- Corrected oracle first-epoch loss **0.4235124270** is close to historical **0.4206887929**; the wrong-root diagnostic was **0.2987813844**. This supports the protocol correction, but does not prove final reproduction.

Compact evidence: [checkpoint replay](../results/independent_domain_a_alignment_20260907/gate0.json), [single-step parity](../results/independent_domain_a_alignment_20260907/parity.json), [annotation counts](../results/independent_domain_a_alignment_20260907/sparse_protocols.json).

## Completed results

| Run | Best foreground validation Dice | Selected epoch (zero-based) / iteration | Foreground test Dice | Inclusive test Dice |
|---|---:|---|---:|---:|
| historical checkpoint replay | 0.7049937621 | 128 / 9,800 | 0.7261413307 | 0.8605453028 |
| new stage-A reference | 0.6693702634 | 78 / 6,000 | 0.7202847429 | not recorded by pinned evaluator |
| **new independent Domain A** | **0.7290565244** | **128 / 9,800** | **0.6792972027** | **0.8363642266** |

The reference completed at 2026-09-07 14:12:54 CST; independent training completed at 14:13:37 CST. Both exited successfully. The independent checkpoint was selected solely by foreground validation Dice, then evaluated once on test at 14:27:01. Its validation score exceeded the historical validation anchor, but the test score was **0.0468441279 below** the target (4.68 percentage points). High validation Dice therefore did not establish target test performance.

The new reference is within the handoff's numerical near-reproduction interval, but remains a reference run and is not relabeled as the final independent result. Single-step implementation parity passed; the full runs diverged under the retained nondeterministic CUDA training behavior. The source candidate also lacks a verified historical training SHA. These results do not isolate a unique cause of the generalization gap.

The independent run used no additional coefficient sweep. No further training is currently running. Any next sweep must predefine candidates and rank them by validation only; the observed test score must not be used to select checkpoints or hyperparameters. The one-seed result is not a multi-seed performance claim.

Final evidence: [independent selected checkpoint and test](../results/independent_domain_a_alignment_20260907/independent_selected_test.json), [independent summary](../results/independent_domain_a_alignment_20260907/independent_summary.json), [reference summary](../results/independent_domain_a_alignment_20260907/oracle_summary.json), [validation curves](../results/independent_domain_a_alignment_20260907/validation_curves.csv).

## Training configuration

Both jobs use seed 42, 150 epochs, batch 4, LR 0.03 with polynomial decay, SGD momentum 0.9, optimizer decay 0 and manual gradient decay 1e-4, PCE/global/spatial 1/1/0, 8 loader workers, 4 OpenMP threads, and validation every 200 iterations. Domain A has 301 slices, 76 batches per epoch, and 11,400 total steps.

The GPU-6 oracle uses the unchanged pinned stage loop at `2cdb1bec5939d8b6b2399413434b8b3aaa9ea7c2`; its task list is restricted to A. GPU 7 runs the independent interface on current source with this patch. They are aligned baseline runs, not a coefficient sweep. The independent training run used `--independent-skip-test`; its validation-selected `best.pt` was evaluated once after training finished. The initial validation-only summary is retained remotely as `validation_summary.json`; final summaries now explicitly mark `score_split=test`.

Use the existing Python 3.10.6 / Torch 2.2.1+cu121 environment. On the authorized experiment host:

```bash
cd /home/jiangsuiyang/ScribbleCL_independent_A_20260907
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 \
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -u main.py --setting-run \
  --data-root /home/jiangsuiyang/medical_continual_segmentation_domain_fastlane/data \
  --sparse-root /home/jiangsuiyang/medical_continual_segmentation_domain_gptpro/data/sparse_annotations_pattern_f5_b10/domain \
  --output /data_nas/jiangsuiyang/ScribbleCL/tune_independent_A_07261_seed42_20260907_1303/independent_pattern_f5_b10 \
  --device cuda:0 --seed 42 --independent-reference --independent-task 1 --independent-skip-test \
  --epochs-per-task 150 --batch-size 4 --lr 0.03 --workers 8 --validate-every 200 \
  --method zs-sequential --pce-loss-weight 1 --zs-global-weight 1 --zs-spatial-loss-weight 0
```

The output must be new for a rerun. Full logs, checkpoints, launch scripts, and exit-code markers are under the common `/data_nas/.../tune_independent_A_07261_seed42_20260907_1303` directory. The original `/home/jiangsuiyang/q1d7f` was not modified. No data, annotations, checkpoints, full logs, or credentials are included in this public release.

Closeout: code, configuration, compact validation curves, and final metrics are public. Original and new checkpoints, annotations, data, and full logs remain on the experiment host. This aligned baseline is complete; optimization to the independent test target remains unresolved.
