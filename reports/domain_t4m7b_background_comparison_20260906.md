# Domain-CL background-inclusive Dice comparison (2026-09-06)

## Scope and status

- Scenario: Domain-CL A-to-F, method `zs-derpp`, seed 42.
- Evaluated checkpoint: completed formal run `t4m7b`, final `s06.pt` after all six stages.
- Run status is `complete`; `summary.json` records six completed stages.
- The same predictions were used once to calculate both metrics. Dice uses smoothing `1e-5`, first averaged per patient/class and then across domains.

## Results

| Domain | Foreground-only Dice | Background Dice | Background-inclusive Dice | Inclusive - foreground |
|---|---:|---:|---:|---:|
| A | 0.6608 | 0.9939 | 0.8273 | +0.1666 |
| B | 0.7487 | 0.9954 | 0.8721 | +0.1234 |
| C | 0.6055 | 0.9932 | 0.7994 | +0.1939 |
| D | 0.2368 | 0.9945 | 0.6156 | +0.3789 |
| E | 0.3750 | 0.9901 | 0.6826 | +0.3076 |
| F | 0.6358 | 0.9951 | 0.8154 | +0.1797 |
| **A-Dice** | **0.5437** | **0.9937** | **0.7687** | **+0.2250** |

Each Domain-CL task has one foreground class, so the background-inclusive score is the arithmetic mean of background Dice and foreground Dice. The inclusive score is higher for every domain because background Dice is close to 0.99. Domains D and E show the masking effect most strongly.

## Interpretation boundary

The background-inclusive A-Dice is the requested reporting definition, but it is dominated by the easy background class. Foreground-only Dice should remain visible in audit material so poor foreground segmentation is not hidden.
