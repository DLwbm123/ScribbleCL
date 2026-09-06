# Independent per-task training: Domain-CL and Class-CL

Status: launched on 2026-09-06; final metrics are pending.

## Scope

- Domain-CL: A/B/C/D/E/F.
- Class-CL: T1/T2/T3.
- One seed (`42`), 80 epochs per run.
- Every task is initialized from scratch and trained twice: first with the complete H5 training labels, then with the frozen scribble annotations.
- Both runs use the same ZScribbleSeg U-Net backbone, PCE objective, data split, augmentation, SGD settings, and batch size (`4`). Only the training-label source changes.
- Full supervision never reads the scribble archive. Class T2/T3 dense labels are shifted to the benchmark's global class IDs; the frozen scribble archives already use those IDs.

## Checkpoint and metric contract

- Validation is performed every five epochs and once after epoch 80.
- `best.pt` is the checkpoint with the best observed validation score; `last.pt` is retained separately.
- Every run records foreground-only Dice, background-inclusive Dice, per-class Dice including background, and foreground prediction fraction.
- Domain-CL selects `best.pt` by background-inclusive Dice, matching the current Domain audit convention.
- Class-CL selects `best.pt` by foreground Dice, while still recording the background-inclusive value for comparison.
- `manifest.json`, `train.jsonl`, `independent_scores.json`, `best.pt`, and `last.pt` are written per run.

## GPU queues

| GPU | Ordered task pairs (full, then scribble) |
|---:|---|
| 4 | Class T1; Domain B |
| 5 | Class T2; Domain D |
| 6 | Class T3; Domain A |
| 7 | Domain F; Domain E; Domain C |

The queues are balanced using training-slice counts. All large outputs and logs are stored on `data_nas`; the small source checkout contains no checkpoints.

## Launch verification

- The full-supervision Domain smoke run completed end to end and wrote its best checkpoint and both Dice variants.
- The scribble-supervision Class smoke run completed end to end.
- A dense-label Class T2 forward/loss check confirmed the local labels are shifted to global IDs and produce a finite loss.
- All four formal queues created their manifests, training processes, and GPU allocations before this report was committed.

Final numbers must not be inferred from this launch report. Publish the completed `independent_scores.json` files and a consolidated comparison table after all 18 runs finish.

## Throughput update at 14:57 CST

Live sampling showed that one process used only 4.7--4.8 GB of each 24 GB GPU and averaged roughly 37%--61% utilization because compute bursts alternated with H5 input waits. The running jobs were preserved, and one additional full-then-scribble pair was added per GPU:

- GPU 4: Class T1 pair plus Domain B pair.
- GPU 5: Class T2 pair plus Domain D pair.
- GPU 6: Class T3 pair plus Domain A pair.
- GPU 7: the Domain F/E queue plus a concurrent Domain C pair.

Post-launch verification found eight training processes, about 9.4--9.5 GB allocated per GPU, 96%--100% instantaneous utilization, and a manifest for every newly started full-supervision run. No active training process was interrupted or restarted.
