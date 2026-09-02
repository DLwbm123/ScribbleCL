# Joint-training code-versus-data diagnosis

## Conclusion

The low Joint Dice was caused primarily by a divergent training entry and an
incomparable optimization schedule, not by missing or mismatched Domain-CL
data. The canonical entry is now `main.py --setting-run --method zs-joint`;
the duplicate standalone entry has been removed.

## Evidence

- Both servers expose the same six HDF5 file sizes, train/validation/test slice
  counts, sampled image statistics, foreground fractions, sparse-archive
  sizes, and sparse-label counts. The pooled training count is 2,266 and every
  sparse array has the expected `(slices, 256, 256)` shape with labels
  `{-100, 0, 1}`.
- The published canonical loader shuffles the pooled dataset into 567 batches
  per epoch. The divergent diagnostic loader oversampled every domain to the
  largest domain and produced 1,056 single-domain batches per epoch.
- The divergent runs also changed PuzzleMix transport/global-loss scaling and
  used Adam at `1e-4` or SGD at `0.03`; their early mean validation Dice was
  0.0426-0.0867.
- With commit `67b98ee` and the same data, a 200-batch SGD `lr=0.03` gate gave
  0.1016 on Python 3.12/Torch 2.6 and 0.1049 on Python 3.10/Torch 2.2. The small
  gap rules out the runtime version as the primary cause.
- The canonical `lr=0.04`, five-epoch schedule was stopped after its first full
  epoch and reached 0.1161 validation Dice. This closely reproduces the
  published three-epoch run's first-epoch value of 0.1147 and confirms that the
  canonical path trains normally on the current data.

## Scope

This diagnosis intentionally stopped after the first comparable epoch. The
published five-epoch result remains the convergence reference; no new
multi-seed or final upper-bound claim is made here. Raw data and checkpoints
are excluded from the public repository.
