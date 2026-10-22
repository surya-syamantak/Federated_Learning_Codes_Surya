"""Random-forest kernel and diffusion-map encoder (Stage 1).

For a forest with trees b = 1..B and reference samples x_1..x_n,

    K(x, x_r) = (1/B) * sum_b 1[x, x_r share a leaf of tree b] / N_b(x)

where N_b(x) is the number of reference samples in x's leaf of tree b. On the
reference set itself K is symmetric, positive semidefinite and doubly
stochastic. Its eigenpairs K v_j = lambda_j v_j give the diffusion map

    Z = sqrt(n) * [v_1 lambda_1^tau, ..., v_dz lambda_dz^tau]

(the constant eigenvector v_0 with lambda_0 = 1 is dropped). New points are
embedded with the Nystrom extension v_j(x) = (1/lambda_j) sum_r K(x, x_r) v_j(r).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class ForestEncoder:
    """RF kernel + diffusion map for one forest and one fixed reference set."""

    def __init__(self, forest: RandomForestClassifier, X_ref: np.ndarray, d_z: int, tau: float):
        self.forest = forest
        self.d_z = d_z
        self.tau = tau
        self.ref_leaves = forest.apply(X_ref)  # (n, B) leaf node id per tree
        self.n, self.B = self.ref_leaves.shape
        # Reference-sample count of every node of every tree.
        self._leaf_counts = [
            np.bincount(self.ref_leaves[:, b], minlength=est.tree_.node_count)
            for b, est in enumerate(forest.estimators_)
        ]
        self.eigenvalues: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.Z: np.ndarray | None = None

    def kernel(self) -> np.ndarray:
        """n x n RF kernel over the reference set."""
        K = np.zeros((self.n, self.n))
        for b in range(self.B):
            leaves = self.ref_leaves[:, b]
            same_leaf = leaves[:, None] == leaves[None, :]
            K += same_leaf / self._leaf_counts[b][leaves][:, None]
        return K / self.B

    def cross_kernel(self, X: np.ndarray) -> np.ndarray:
        """m x n kernel between new points and the reference set.

        Trees whose leaf for x holds no reference sample are skipped and the
        average is taken over the remaining trees, so every row still sums to 1.
        """
        leaves = self.forest.apply(X)
        Kx = np.zeros((len(X), self.n))
        used = np.zeros(len(X))
        for b in range(self.B):
            counts = self._leaf_counts[b][leaves[:, b]]
            ok = counts > 0
            same_leaf = leaves[ok, b][:, None] == self.ref_leaves[None, :, b]
            Kx[ok] += same_leaf / counts[ok][:, None]
            used += ok
        nonzero = used > 0
        Kx[nonzero] /= used[nonzero][:, None]
        return Kx

    def fit(self) -> "ForestEncoder":
        """Spectral decomposition and diffusion coordinates of the reference set."""
        K = self.kernel()
        evals, evecs = np.linalg.eigh(K)
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0.0, 1.0)
        evecs = evecs[:, order]
        # Drop the trivial constant eigenvector (lambda_0 = 1).
        lam = evals[1 : self.d_z + 1]
        V = evecs[:, 1 : self.d_z + 1]
        # Deterministic signs: largest-magnitude entry of each eigenvector positive.
        signs = np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])])
        V = V * np.where(signs == 0, 1.0, signs)
        self.eigenvalues, self.V = lam, V
        self.Z = np.sqrt(self.n) * V * lam**self.tau
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Nystrom out-of-sample diffusion coordinates for new points."""
        if self.V is None:
            raise RuntimeError("call fit() first")
        Kx = self.cross_kernel(X)
        lam = self.eigenvalues
        scale = np.zeros_like(lam)
        positive = lam > 1e-12
        scale[positive] = lam[positive] ** (self.tau - 1.0)
        return np.sqrt(self.n) * (Kx @ self.V) * scale


def train_forest(X: np.ndarray, y: np.ndarray, n_trees: int, min_samples_leaf: int,
                 max_features: str, seed: int) -> RandomForestClassifier:
    """Fit a local RF. A fixed seed per client keeps rounds comparable."""
    forest = RandomForestClassifier(
        n_estimators=n_trees,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=seed,
        n_jobs=1,
    )
    return forest.fit(X, y)
