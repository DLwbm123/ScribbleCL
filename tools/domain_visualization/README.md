# Domain-CL best-slice visualization

Post-hoc qualitative comparison using existing final-stage checkpoints. No training, model selection, or aggregate benchmark changes. Edit `config.example.json` with private runtime/data/checkpoint paths, then run:

```sh
python visualize.py self-check
python visualize.py infer config.json
python visualize.py render /path/to/private/output
```

Uses the existing runtime's `H5Slices`, `DomainModel`, and `zs_forward`. Requires its PyTorch environment plus NumPy and Matplotlib. The configuration contains exactly five methods in display order, with ZS-DER++ last. All use `s06.pt` after A–F. Historical baseline runs used 150 epochs/task; ZS-DER++ used 80. ZS-Sequential and ZS-EWC use global weight 1, runs x7p2c/x7p2d; these explicit checkpoints take precedence over legacy rounded table entries.

Select the maximum ZS-DER++ foreground slice Dice independently per domain, excluding empty ground truth; break ties by greater target area then lower slice index. Score the baselines on exactly those selected slices. All scores use the full slice, even in zoomed figures. No mask postprocessing. These best-case examples are not estimates of typical performance or aggregate method superiority.

The exporter verifies the six whole-test, patient-mean foreground scores against the preserved formal report before saving. The September 21 run reproduced all six within 1e-5, scanning 1,030 slices. The figure covers PCE-Sequential, ZS-Sequential, ZS-EWC, ZS-GPM, and ZS-DER++. Dense-Sequential and incomplete ER/DER runs are not included.

Keep all generated medical images, masks, per-slice tables, selection metadata, and NPZ exports private. This directory publishes only reusable source, a redacted configuration, and this method note.
