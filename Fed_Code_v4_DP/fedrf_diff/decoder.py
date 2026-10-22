"""Stage 3 client decoder: latent coordinates -> feature space.

1. RF-compatible points: for each training sample j, intersect the leaf boxes
   R_j = intersection over b of R_{j,b} of all trees, clip to the client's
   data range, and take the box centre for numeric features. Binary and
   nominal features keep the sample's own value (it lies inside R_j).
2. k-NN in diffusion coordinates (Euclidean distance in Z approximates the
   diffusion distance), weights w_ij proportional to 1 / d_diff, normalised.
3. Numeric features: weighted average of the neighbours' RF-compatible
   values. Binary/nominal features: weighted majority vote.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors


def _node_boxes(tree, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper bounds of every node's region (sklearn: x <= thr goes left)."""
    t = tree.tree_
    lower = np.full((t.node_count, n_features), -np.inf)
    upper = np.full((t.node_count, n_features), np.inf)
    stack = [0]
    while stack:
        node = stack.pop()
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:
            continue
        f, thr = t.feature[node], t.threshold[node]
        for child in (left, right):
            lower[child] = lower[node]
            upper[child] = upper[node]
        upper[left, f] = min(upper[node, f], thr)
        lower[right, f] = max(lower[node, f], thr)
        stack.extend((left, right))
    return lower, upper


def leaf_intersection_boxes(
    forest: RandomForestClassifier, X: np.ndarray, data_min: np.ndarray, data_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample box where all trees agree, clipped to [data_min, data_max]."""
    leaves = forest.apply(X)
    lo = np.tile(data_min.astype(float), (len(X), 1))
    hi = np.tile(data_max.astype(float), (len(X), 1))
    for b, est in enumerate(forest.estimators_):
        lower, upper = _node_boxes(est, X.shape[1])
        lo = np.maximum(lo, lower[leaves[:, b]])
        hi = np.minimum(hi, upper[leaves[:, b]])
    return lo, hi


def rf_compatible_points(
    forest: RandomForestClassifier, X: np.ndarray, numeric: np.ndarray,
    data_min: np.ndarray, data_max: np.ndarray,
) -> np.ndarray:
    """x~_j: centre of the leaf-intersection box (numeric), own value otherwise."""
    lo, hi = leaf_intersection_boxes(forest, X, data_min, data_max)
    X_tilde = X.astype(float).copy()
    X_tilde[:, numeric] = ((lo + hi) / 2.0)[:, numeric]
    return X_tilde


class KNNDecoder:
    def __init__(self, Z_ref: np.ndarray, X_tilde: np.ndarray, numeric: np.ndarray,
                 k: int, dist_floor: float = 1e-12):
        self.X_tilde = X_tilde
        self.numeric = numeric
        self.k = min(k, len(Z_ref))
        self.dist_floor = dist_floor
        self._nn = NearestNeighbors(n_neighbors=self.k).fit(Z_ref)

    def weights(self, Z_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dist, idx = self._nn.kneighbors(Z_query)
        # Floor the distance so exact matches (e.g. duplicate patients) get
        # large but finite, equal weights.
        w = 1.0 / np.maximum(dist, self.dist_floor)
        return w / w.sum(axis=1, keepdims=True), idx

    def decode(self, Z_query: np.ndarray) -> np.ndarray:
        w, idx = self.weights(Z_query)
        neigh = self.X_tilde[idx]  # (m, k, d_x)
        X_hat = np.einsum("mk,mkd->md", w, neigh)
        for f in np.flatnonzero(~self.numeric):
            values = neigh[:, :, f]
            candidates = np.unique(values)
            scores = np.stack([(w * (values == v)).sum(axis=1) for v in candidates], axis=1)
            X_hat[:, f] = candidates[np.argmax(scores, axis=1)]
        return X_hat
