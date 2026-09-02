# Completed results

## Validated canonical joint training

The synchronized `ScribbleCL44` code path pools all 2,266 A-F training slices
and uses 567 shuffled batches per epoch. Values below are Dice fractions.

| LR | Epochs | Best validation | Test mean |
|---:|---:|---:|---:|
| 0.02 | 3 | 0.4261 | 0.3730 |
| 0.03 | 3 | 0.4876 | 0.4858 |
| 0.04 | 3 | 0.5228 | 0.5414 |
| 0.03 | 5 | 0.5287 | 0.4998 |
| **0.04** | **5** | **0.5662** | **0.5628** |

The selected five-epoch checkpoint produced A-F Dice
`0.6163 / 0.5365 / 0.6602 / 0.5964 / 0.6039 / 0.3635`; training loss decreased
from 0.3166 to 0.1877. All published short runs passed the output-contract
audit.

## Legacy standalone runs

The four older runs completed six-domain evaluation without a recorded traceback,
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

These are preserved experiment outputs, not paper claims. The two low joint
rows came from the now-removed standalone training entry and are superseded by
the canonical convergence study above; they must not be used as the joint
upper-bound result.
