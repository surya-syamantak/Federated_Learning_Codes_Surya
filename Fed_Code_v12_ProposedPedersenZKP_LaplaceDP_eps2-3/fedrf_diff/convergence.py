"""Stage 4: dual convergence criterion (Delta Z and Delta L) with patience.

    Delta Z^(t) = ||Z_G^(t) - Z_G^(t-1)||_F / (||Z_G^(t-1)||_F + eps_0)
    Delta L^(t) = |L_avg^(t) - L_avg^(t-1)|

Stop when both are below their thresholds for ``patience`` consecutive rounds.

With Laplace DP every patient row of Z_G carries fresh zero-mean noise of
standard deviation sigma = sqrt(2) b each round. sigma is public, so the server removes the noise's known
expected energy before comparing rounds (N noisy rows, d coordinates):

    Delta Z = sqrt(max(0, ||dZ_G||^2 - 2 N d sigma^2)) / sqrt(max(eps_0, ||Z_G^(t-1)||^2 - N d sigma^2))

The noise also perturbs the decoded data that feeds the next round, so Delta Z
still settles higher under DP: about 0.11 + sigma / C on seed 42 (C = clip
norm). The threshold is therefore eps_Z + sigma / C, which reduces to eps_Z
without DP.
"""

from __future__ import annotations

import numpy as np


def relative_change(new: np.ndarray, old: np.ndarray, eps_0: float,
                    sigma: float = 0.0, noisy_entries: int = 0, sigma_old: float | None = None,
                    base_floor: float = 0.0) -> float:
    """Relative change of ``new`` vs ``old``, debiased for iid zero-mean noise.

    ``sigma`` is the noise std in ``new`` and ``sigma_old`` in ``old``
    (the same unless the DP budget changes between rounds). ``base_floor`` is
    a noise-free lower bound for the signal energy ||old||^2: when the noise
    is as large as the signal, the debiased denominator can drop to ~0 and
    the ratio explode, so it is never allowed below this floor.
    """
    sigma_old = sigma if sigma_old is None else sigma_old
    diff = np.linalg.norm(new - old) ** 2 - noisy_entries * (sigma ** 2 + sigma_old ** 2)
    base = np.linalg.norm(old) ** 2 - noisy_entries * sigma_old ** 2
    if sigma == 0 and sigma_old == 0:
        return float(np.linalg.norm(new - old) / (np.linalg.norm(old) + eps_0))
    return float(np.sqrt(max(0.0, diff)) / np.sqrt(max(eps_0, base, base_floor)))


def weighted_loss(losses: dict[str, float], sizes: dict[str, int]) -> float:
    """L_avg = sum_c (n_c / sum_r n_r) L_c over the clients that reported."""
    total = sum(sizes[c] for c in losses)
    return float(sum(sizes[c] / total * losses[c] for c in losses))


class ConvergenceMonitor:
    def __init__(self, eps_z: float, eps_l: float, patience: int, eps_0: float,
                 sigma: float = 0.0, clip_norm: float = 1.0):
        self.eps_l, self.patience, self.eps_0 = eps_l, patience, eps_0
        self.eps_z_base, self.clip_norm = eps_z, clip_norm
        self.sigma = self._prev_sigma = sigma
        self._prev_row_energy = 0.0
        self.counter = 0
        self._prev_Z: dict[str, np.ndarray] | None = None
        self._prev_L: float | None = None

    @property
    def eps_z(self) -> float:
        """Delta Z threshold for the current round: eps_Z + sigma / C."""
        return self.eps_z_base + (self.sigma / self.clip_norm if self.sigma > 0 else 0.0)

    def set_sigma(self, sigma: float) -> None:
        """This round's noise std (when the DP budget changes from round to round)."""
        self.sigma = sigma

    def delta_z(self, Z_G: dict[str, np.ndarray], row_energy: float = 0.0) -> float | None:
        """Compare with the previous round on the clients present in both.

        ``row_energy`` is a noise-free estimate of the mean squared norm of a
        latent row this round (from the public landmark rows); the previous
        round's value floors the denominator.
        """
        prev, self._prev_Z = self._prev_Z, {k: v.copy() for k, v in Z_G.items()}
        sigma_old, self._prev_sigma = self._prev_sigma, self.sigma
        energy_old, self._prev_row_energy = self._prev_row_energy, row_energy
        if prev is None:
            return None
        common = [c for c in Z_G if c in prev]
        if not common:
            return None
        new = np.vstack([Z_G[c] for c in common])
        old = np.vstack([prev[c] for c in common])
        return relative_change(new, old, self.eps_0, self.sigma, new.size, sigma_old,
                               base_floor=len(old) * energy_old)

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
