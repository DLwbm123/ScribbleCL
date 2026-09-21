# Class-CL cumulative visualization from final models

Corrected definition: every row uses the same final T3 checkpoint (`s03.pt`) for each method. The rows are cumulative views of classes C1–C3, C1–C5, and C1–C7. They are not snapshots of intermediate model stages, nor separate 3/2/2 new-class evaluations.

The six completed, matched-budget models are Dense-Sequential, PCE-Sequential, ZS-Sequential, ZS-EWC, ZS-GPM, and ZS-DER++ + MiB (`main_s0`) from the September 10 campaign: seed 42, 80 epochs/task. No training, tuning, or checkpoint reselection.

Use the campaign's frozen runtime and its PyTorch/NumPy/Matplotlib environment. Edit config.example.json with private paths and save as config.json:

```sh
python visualize.py self-check
python visualize.py infer config.json
python visualize.py render /path/to/private/output
```

Ground truth comes from the existing fully annotated `MMWHS/whole_heart_test.h5` (1,350 slices, nine patients). This differs from the 900-slice new-class task subsets used by the former exporter. Therefore cumulative scores must not be compared as if from the same subset metrics.

Inference always takes argmax over all eight channels (background plus seven classes). Only after argmax are labels above the row's cumulative limit mapped to background for display, with the same filtering applied to ground truth. Excluded winners are not reassigned by a restricted-logit argmax. Foreground Dice for included classes is unchanged by this display filtering. Raw complete GT and predictions are retained in the private NPZ.

Select the highest Ours macro foreground Dice separately for each cumulative view, requiring every included class to be present in the slice. Break ties by larger cumulative foreground area, then lower index. All methods use exactly the selected slice, contrast and crop. This is post-hoc best-case selection, not an estimate of average performance. Ours must be last in the configuration.

The exporter verifies all seven complete-test per-class patient-mean Dice scores against the existing `whole_class_dice` summary within 1e-5; it also checks patient-boundary coverage, cumulative label ranges, and equality of raw-versus-displayed foreground scores. The self-check specifically covers 3/5/7 filtering and prevents relabeling of excluded predictions.

Colored masks use verified H5 IDs C1–C7. White dashed contours indicate cumulative GT. Dice appears below each prediction. The display window uses GT-foreground intensity percentiles 1–99, padded by 25% of that range on each side, identically across methods. Full-field and zoomed figures preserve identical masks and scores.

The earlier 3/2/2 subset visualization is superseded for this requested cumulative view. Generated medical images, masks, per-slice tables, private paths and NPZ data remain private; this directory publishes source and a redacted configuration only.

Validation: self-check passed; final-model inference on all 1,350 whole-heart slices reproduced all seven stored per-class means within 1e-5. Each selected row passed GT class-presence, displayed prediction-range, and raw/display score-equivalence assertions.

## Requested acquisition-baseline contrastive mode

For this separate view, the five baselines use s01/s02/s03 after each row task; Ours uses the final s03 in every row. Stage labels are printed beneath every prediction. It is a different-stage qualitative comparison, not a same-stage benchmark ranking.

First preserve a completed cumulative export from the main mode above. Create a private phase configuration with the same runtime, data and methods, set `source_output` to that completed export and `output` to a new directory. Optionally set `preserve_t1_from` to a previous stage-baseline export to retain its T1 row. Run:

```sh
python select_slices.py self-check
python select_slices.py phase_config.json
```

Selection deliberately favors Ours: keep its scores at least 90% of the best eligible slice for the cumulative class set; require every acquisition baseline to have at least 32 predicted foreground pixels and Dice >=0.01. T1 maximizes the margin over the strongest baseline unless explicitly preserved. T2/T3 additionally require an Ours margin >=0.10, then maximize the minimum pairwise class-aware foreground disagreement among the five baselines in the displayed crop. Ties use mean disagreement, Ours margin, Ours Dice, then lower index. Disagreement is the fraction of unequal class labels within the pair's foreground union; background agreement does not dominate selection.

Fresh selected-Ours inference uses the same batch of four consecutive test indices as the original export, avoiding batch-shape numerical differences at argmax boundaries. Its Dice must match the cached validated score within 1e-5. Stage output channels are checked (4/6/8). Private candidate masks, selection scores and checkpoint stages are saved. Predictions and aggregate experimental results are not edited.

Validation: self-check covers visible foreground, contrastive selection, pair disagreement, and diversity ranking. T2/T3 screened 94/40 high-Ours candidates; every selected baseline met foreground/overlap requirements. The closest baseline pair disagreement increased from 8.18% to 19.84% for T2 and 18.30% to 34.24% for T3, with Ours Dice 0.831357 and 0.716666. T1 was retained. Figures were visually inspected; stage labels and Dice remain visible. Some genuine baseline similarities remain, especially the final baselines' lack of old-class predictions.

`audit_foreground.py config.json` optionally audits final-model class presence across the complete whole-heart test set; `self-check` tests its pixel counting. The September 21 audit found no C1–C3 predicted pixels for any of the five final baselines on all 1,350 slices. Four also had no C1–C5 predictions; PCE had some C4 predictions. This explains why changing slices alone could not make every final baseline visible in the earlier cumulative rows. Raw per-slice audit outputs remain private.
