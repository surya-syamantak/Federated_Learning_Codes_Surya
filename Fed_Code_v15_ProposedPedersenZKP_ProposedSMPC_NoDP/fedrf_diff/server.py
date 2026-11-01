"""The global server: Stage 2 (alignment, aggregation, global model)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .alignment import orthogonal_procrustes
from .client import LatentMessage


class GlobalForest:
    """The broadcast M_glob: a random forest with Laplace-smoothed probabilities.

    A raw forest returns probability 0 or 1 when every tree agrees, and one
    confidently wrong validation patient then adds ~34 / n_val to the log-loss
    (about 0.6 for Hungary), which makes Delta L jump between rounds.
    Smoothing (B * p + 1) / (B + 2) caps a unanimous vote at (B + 1) / (B + 2).
    """

    def __init__(self, forest: RandomForestClassifier):
        self.forest = forest
        self.classes_ = forest.classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        B, k = len(self.forest.estimators_), len(self.classes_)
        return (self.forest.predict_proba(X) * B + 1.0) / (B + k)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forest.predict(X)


class Server:
    def __init__(self, seed: int, n_trees: int = 300, min_samples_leaf: int = 50):
        self.seed = seed
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.P_ref: np.ndarray | None = None

    def set_reference(self, P_ref: np.ndarray) -> None:
        """Freeze the landmark reference matrix P_ref (m x d)."""
        self.P_ref = P_ref.copy()

    def align(self, msg: LatentMessage) -> np.ndarray:
        """R_k^*(r) = argmin ||P_k A - P_ref||_F with A^T A = I."""
        return orthogonal_procrustes(msg.P, self.P_ref)

    @staticmethod
    def aggregate(messages: dict[str, LatentMessage], A: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Z_glob^(r): aligned rows Z_k R_k of every accepted client, kept per client.

        Clients hold different patients, so their rows are stacked rather than
        summed; stacking weights each client by N_k, i.e. w_k = N_k / sum N_j.
        """
        return {name: msg.Z_tilde @ A[name] for name, msg in messages.items()}

    def train_global_model(
        self, Z_G: dict[str, np.ndarray], messages: dict[str, LatentMessage]
    ) -> tuple[GlobalForest, dict[str, np.ndarray]]:
        """M_glob^(r) = f_G(Z_glob^(r)): random forest on the aligned latents.

        Also returns each client's pseudo-labels y^ for its training rows,
        taken out-of-bag (only from trees that did not train on that row).
        The rows are part of Z_glob, so in-sample predictions would mostly hand
        back y itself and the decoded data would carry no global knowledge.
        """
        names = list(Z_G)
        X = np.vstack([Z_G[n] for n in names])
        y = np.concatenate([messages[n].y for n in names])
        model = RandomForestClassifier(
            n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf,
            oob_score=True, random_state=self.seed, n_jobs=1,
        ).fit(X, y)
        oob = model.oob_decision_function_
        labels = model.classes_[np.argmax(np.nan_to_num(oob, nan=-1.0), axis=1)]
        no_oob = np.isnan(oob).any(axis=1)  # row was in-bag for every tree
        if no_oob.any():
            labels[no_oob] = model.predict(X[no_oob])
        bounds = np.cumsum([0] + [len(Z_G[n]) for n in names])
        pseudo = {n: labels[bounds[i] : bounds[i + 1]].astype(int) for i, n in enumerate(names)}
        return GlobalForest(model), pseudo
