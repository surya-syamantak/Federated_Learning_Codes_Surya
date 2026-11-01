"""A federated client: Stage 1 (encode) and Stage 3 (loss, decode, retrain). No DP."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import log_loss

from .config import Config
from .data import ClientData, numeric_mask
from .decoder import KNNDecoder, rf_compatible_points
from .metrics import confusion_counts, metrics_from_counts
from .rf_kernel import ForestEncoder, train_forest


@dataclass
class LatentMessage:
    """What a client transmits in Stage 1 (after ZKP authentication)."""

    client: str
    Z_tilde: np.ndarray  # latent rows of the training samples (sent as computed, no DP)
    P: np.ndarray  # latent rows of the public landmarks (P_k)
    y: np.ndarray  # training labels, needed by the server to train M_glob
    n: int  # N_k, training-set size

    @property
    def full_matrix(self) -> np.ndarray:
        """Z_k including landmark rows - the matrix covered by the ZKP."""
        return np.vstack([self.Z_tilde, self.P])


class Client:
    def __init__(self, data: ClientData, X_landmarks: np.ndarray, cfg: Config, index: int):
        self.name = data.name
        self.data = data
        self.X_landmarks = X_landmarks
        self.cfg = cfg
        # Fixed per-client seed: retraining alone must not move Z between rounds.
        self.seed = cfg.seed + 1000 * (index + 1)
        self.numeric = numeric_mask()
        self.data_min = data.X_train.min(axis=0)
        self.data_max = data.X_train.max(axis=0)
        self.forest = self._train(data.X_train, data.y_train)  # RF_k^(0)
        self.encoder: ForestEncoder | None = None
        self.Z: np.ndarray | None = None
        self.Z_tilde: np.ndarray | None = None
        # (encoder, R_k) that produced the rows M_glob^(r) was trained on.
        self.snapshot: tuple[ForestEncoder, np.ndarray] | None = None
        # Seconds spent in each local step of the current round.
        self.timing: dict[str, float] = {}

    @property
    def n_train(self) -> int:
        return self.data.n_train

    def _train(self, X: np.ndarray, y: np.ndarray):
        c = self.cfg
        return train_forest(X, y, c.n_trees, c.min_samples_leaf, c.max_features, self.seed)

    # --- Stage 1 ----------------------------------------------------------
    def encode(self) -> LatentMessage:
        """RF kernel of RF_k^(r-1) -> diffusion map.

        No differential privacy: the latent vectors (patients and landmarks)
        are sent exactly as computed, without clipping or noise.
        """
        t0 = time.perf_counter()
        c = self.cfg
        n = self.n_train
        X_ref = np.vstack([self.data.X_train, self.X_landmarks])
        self.encoder = ForestEncoder(self.forest, X_ref, c.d_z, c.tau).fit()
        self.Z = self.encoder.Z
        self.Z_tilde = self.Z.copy()
        self.timing = {"encode_s": time.perf_counter() - t0}
        return LatentMessage(
            client=self.name, Z_tilde=self.Z_tilde[:n], P=self.Z_tilde[n:],
            y=self.data.y_train, n=n,
        )

    # --- Stage 3 ----------------------------------------------------------
    def evaluate(self, model, A: np.ndarray) -> dict:
        """L_k^(r) and confusion-matrix metrics on the local validation split.

        Only scalars leave the client (loss and TP/FP/TN/FN counts).
        """
        t0 = time.perf_counter()
        Z_val = self.encoder.transform(self.data.X_val)
        proba = _positive_proba(model, Z_val @ A)
        self.snapshot = (self.encoder, A)
        counts = confusion_counts(self.data.y_val, proba >= 0.5)
        self.timing["evaluate_s"] = time.perf_counter() - t0
        return {
            "loss": float(log_loss(self.data.y_val, proba, labels=[0, 1])),
            "n_val": len(self.data.y_val),
            **counts,
            **metrics_from_counts(**counts),
        }

    def decode_and_update(self, A: np.ndarray, y_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode the aligned coordinates, augment D_train, retrain RF_k^(r).

        ``y_hat`` are the global model's (out-of-bag) labels for this client's
        training rows, sent by the server with M_glob and R_k.
        """
        t0 = time.perf_counter()
        n = self.n_train
        if len(y_hat) != n:
            raise ValueError(f"expected {n} pseudo-labels, got {len(y_hat)}")
        aligned = self.Z_tilde[:n] @ A
        # Map the aligned coordinates back to this client's frame (A^-1 = A^T).
        query = aligned @ A.T
        X_tilde = rf_compatible_points(
            self.forest, self.data.X_train, self.numeric, self.data_min, self.data_max
        )
        decoder = KNNDecoder(self.Z[:n], X_tilde, self.numeric, self.cfg.k_neighbors)
        X_hat = decoder.decode(query)
        self.timing["decode_s"] = time.perf_counter() - t0  # decoding only (before retraining)
        # Rebuilt from the original D_train every round (not accumulated).
        X_aug = np.vstack([self.data.X_train, X_hat])
        y_aug = np.concatenate([self.data.y_train, y_hat])
        self.forest = self._train(X_aug, y_aug)
        self.timing["decode_update_s"] = time.perf_counter() - t0
        return X_hat, y_hat

    # --- inference after training ---------------------------------------------
    def predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        """P(heart disease) for new patients, using the encoder M_glob was trained with."""
        if self.snapshot is None:
            raise RuntimeError("no completed round yet")
        encoder, A = self.snapshot
        return _positive_proba(model, encoder.transform(X) @ A)


def _positive_proba(model, Z: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(Z)
    return proba[:, list(model.classes_).index(1)]
