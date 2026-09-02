import numpy as np

from zs_gco_compat import cut_grid_graph


unary = np.zeros((3, 4, 2), dtype=np.float64)
unary[..., 1] = 1.0
pairwise = np.array([[0.0, 1.0], [1.0, 0.0]])
labels = cut_grid_graph(
    unary,
    pairwise,
    np.ones((2, 4), dtype=np.float64),
    np.ones((3, 3), dtype=np.float64),
)
assert labels.shape == (3, 4)
assert set(np.unique(labels)).issubset({0, 1})
print("PASS")
