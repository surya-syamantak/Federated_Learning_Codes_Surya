"""Stage 4: dual convergence criterion (Delta Z and Delta L) with patience.

    Delta Z^(t) = ||Z_G^(t) - Z_G^(t-1)||_F / (||Z_G^(t-1)||_F + eps_0)
    Delta L^(t) = |L_avg^(t) - L_avg^(t-1)|

Stop when both are below their thresholds for ``patience`` consecutive rounds.
No differential privacy: the latents carry no noise, so Delta Z is the plain
relative change and the threshold is eps_Z.
"""

from __future__ import annotations

import numpy as np


def relative_change(new: np.ndarray, old: np.ndarray, eps_0: float) -> float:
    """||new - old||_F / (||old||_F + eps_0)."""
    return float(np.linalg.norm(new - old) / (np.linalg.norm(old) + eps_0))


def weighted_loss(losses: dict[str, float], sizes: dict[str, int]) -> float:
    """L_avg = sum_c (n_c / sum_r n_r) L_c over the clients that reported."""
    total = sum(sizes[c] for c in losses)
    return float(sum(sizes[c] / total * losses[c] for c in losses))


class ConvergenceMonitor:
    def __init__(self, eps_z: float, eps_l: float, patience: int, eps_0: float):
        self.eps_z, self.eps_l, self.patience, self.eps_0 = eps_z, eps_l, patience, eps_0
        self.counter = 0
        self._prev_Z: dict[str, np.ndarray] | None = None
        self._prev_L: float | None = None

    def delta_z(self, Z_G: dict[str, np.ndarray]) -> float | None:
        """Compare with the previous round on the clients present in both."""
        prev, self._prev_Z = self._prev_Z, {k: v.copy() for k, v in Z_G.items()}
        if prev is None:
            return None
        common = [c for c in Z_G if c in prev]
        if not common:
            return None
        new = np.vstack([Z_G[c] for c in common])
        old = np.vstack([prev[c] for c in common])
        return relative_change(new, old, self.eps_0)

    def delta_l(self, L_avg: float) -> float | None:
        prev, self._prev_L = self._prev_L, L_avg
        return None if prev is None else abs(L_avg - prev)

    def step(self, dz: float | None, dl: float | None) -> bool:
        """Update the patience counter; True means converged (stop)."""
        if dz is None or dl is None:  # round 1: nothing to compare yet
            return False
        if dz < self.eps_z and dl < self.eps_l:
            self.counter += 1
        else:
            self.counter = 0
        return self.counter >= self.patience
