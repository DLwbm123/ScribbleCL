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
