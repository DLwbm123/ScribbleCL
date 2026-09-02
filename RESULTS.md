# Completed results

All four runs completed six-domain evaluation without a recorded traceback,
CUDA OOM, non-finite loss, or disk error. Values below are Dice fractions.

| Run | A | B | C | D | E | F | Mean / A-Dice | BWTR | E-FWT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ZS-GPM | 0.090966 | 0.094861 | 0.142258 | 0.087189 | 0.160191 | 0.058225 | 0.105615 | -0.056962 | 0.063095 |
| ZS-DER++ | 0.088818 | 0.087868 | 0.127312 | 0.074516 | 0.135325 | 0.053173 | 0.094502 | -0.000891 | 0.046705 |
| ZS-Joint, batch 2, LR 0.015 | 0.087712 | 0.086226 | 0.134911 | 0.075801 | 0.154687 | 0.052147 | 0.098581 | n/a | n/a |
| ZS-Joint, batch 4, LR 0.03 | 0.080601 | 0.079450 | 0.114604 | 0.066674 | 0.122744 | 0.047692 | 0.085294 | n/a | n/a |

For continual runs, A-F are the final row of the performance matrix and the
aggregate is the recorded A-Dice. Joint runs use the six-domain test mean.
BWTR and E-FWT are continual-learning metrics and are not defined for joint
training.

The batch-2 joint configuration selected its best checkpoint at epoch 1,
iteration 2000, with mean validation Dice 0.108281. The batch-4 configuration
selected epoch 12, iteration 7000, with mean validation Dice 0.094450.

These are completed experiment outputs, not paper claims. In this sweep the
joint configurations were below ZS-GPM, so calling them a measured upper
bound would be inaccurate.
