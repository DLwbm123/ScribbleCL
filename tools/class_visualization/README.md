# Class-CL best-slice visualization

Post-hoc visualization of six existing, completed 80-epoch-per-task models from the September 10 comparison campaign. Uses the campaign's frozen Class runtime, its H5Slices label mapping and its inference function. The methods are Dense-Sequential, PCE-Sequential, ZS-Sequential, ZS-EWC, ZS-GPM, and ZS-DER++ + MiB (main_s0). All use s03.pt after T3; historical v6c43 is not substituted. No new training or model selection.

Edit the private paths in config.example.json and save as config.json, then use the existing runtime environment (PyTorch, NumPy, Matplotlib):

```sh
python visualize.py self-check
python visualize.py infer config.json
python visualize.py render /path/to/private/output
```

The main method must be last. Inference performs eight-channel argmax (background plus all seven learned classes), with no task-specific output masking or postprocessing. T1 is MYO/LV/LA, T2 is RA/RV, T3 is AO/PA. Select the greatest macro foreground slice Dice for each task among test slices containing all its foreground classes; absent classes cannot inflate selection. Ties favor greater target area, then the smaller slice index. Baselines use exactly the same slices. These post-hoc best cases do not represent mean test performance.

Every figure uses the same anatomy colors, slice, contrast, and crop across methods. Predictions show all seven class labels, including later classes on earlier-task images; the available ground truth contains the task's classes only. Dashed white contours denote task GT. Dice uses the full slice and task classes, excluding background. Zoom changes only the display; full-field figures retain predictions outside the crop.

The exporter checks the main method's full-test patient/class mean for each task against its persisted summary with tolerance 1e-5. Its small runnable self-check covers label identity, absent-class exclusion, and unmasked later-class mistakes.

Generated images, masks, per-slice scores, NPZ data, and private paths stay local or on the authorized server. This directory contains source and a redacted configuration only.

Figure legends use verified H5 global IDs C1–C7. The historical protocol names task-level anatomy groups, but the original anatomy-to-H5 conversion table was not found during this export; no unverified per-channel anatomy names are assigned.

Display window: 1st–99th percentiles of intensities inside the selected task ground truth, expanded by 25% of that range on both sides, shared by all methods. This changes grayscale display only; masks and scores are unchanged.

Validation (2026-09-21): the self-check passed; all 2,700 test slices were evaluated for the main method, reproducing its three stored patient/class means within 1e-5. The overview and single-task layouts were rendered and visually checked.
