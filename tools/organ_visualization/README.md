# Organ-CL qualitative visualization

Four binary tasks: left atrium (UtahI), prostate (UCL), liver (Lits), brain tumor (brain). Inference uses each task's own output head. Five baselines use checkpoints s01/s02/s03/s04 at the end of the corresponding task; Ours uses final s04 for every row. This deliberately compares different stages and must not be presented as a matched-stage quantitative ranking.

The six runs are the established 60/60/10/10-epoch cohort: Dense-Sequential, PCE-Sequential, ZS-Sequential, ZS-EWC, ZS-GPM and ZS-DER++. ER's separate 40/40/40/40 cohort is not included. No training, checkpoint tuning, postprocessing or mask editing is performed.

Edit config.example.json with private paths and run in the campaign's PyTorch/NumPy/Matplotlib environment:

```sh
python visualize.py self-check
python visualize.py infer config.json
python visualize.py render /path/to/private/output
```

Before selection, every loaded model's complete-task patient-mean foreground Dice is checked against the original summary within 1e-5. Output must have two channels; patient boundaries must cover the complete task. Figure annotations use full-slice foreground Dice, excluding background, unlike the background-inclusive main report.

Selection: require at least 32 GT pixels; keep Ours >=70% of its best eligible slice Dice; require all methods >=32 predicted foreground pixels and Dice >=0.01; Ours must strictly exceed all baselines. Rank by the weakest Dice among ZS-Sequential/EWC/GPM, then minimum pairwise baseline foreground disagreement in the displayed crop, Ours Dice and lower slice index. This prioritizes stronger ZS predictions, while retaining Ours superiority. These are post-hoc qualitative cases, not representative or aggregate performance estimates.

All methods share the slice, contrast window and crop. Display follows the Class/Domain figures: red foreground fill at alpha 0.42 and matching contours, white dashed ground truth over prediction panels, Dice and checkpoint stage beneath each prediction. Full-view and zoomed figures use identical masks and scores.

Medical images, candidate caches, raw predictions, patient/slice identifiers, private checkpoint paths and per-slice score records remain private. Public delivery contains only scripts, redacted configuration and this note.

Validation: 24 method/task patient-mean foreground scores matched original summaries within 1e-5 across 298/100/496/1223 test slices. Ours selected-slice Dice: 0.827897 / 0.793330 / 0.904557 / 0.932461. T3 has no candidate with an Ours margin >=0.02 over all baselines; strict superiority is used without that fixed gap. The stronger baselines on T3/T4 consequently have smaller visual differences.
