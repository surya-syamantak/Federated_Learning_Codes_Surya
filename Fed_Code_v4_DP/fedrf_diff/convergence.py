"""Stage 4: dual convergence criterion (Delta Z and Delta L) with patience.

    Delta Z^(r) = ||Z_glob^(r) - Z_glob^(r-1)||_F / (||Z_glob^(r-1)||_F + eps_0)
    Delta L^(r) = |L_glob^(r) - L_glob^(r-1)|

Stop when both are below their thresholds for ``patience`` consecutive rounds.

With Gaussian DP every patient row of Z_glob carries fresh N(0, sigma^2) noise
each round. sigma is public, so the server removes the noise's known
expected energy before comparing rounds (N noisy rows, d coordinates):

    Delta Z = sqrt(max(0, ||dZ_G||^2 - 2 N d sigma^2)) / sqrt(max(eps_0, ||Z_glob^(r-1)||^2 - N d sigma^2))

The noise also perturbs the decoded data that feeds the next round, so Delta Z
still settles higher under DP: about 0.11 + sigma / C on seed 42 (C = clip
norm). The threshold is therefore eps_z + sigma / C, which reduces to eps_z
without DP.
"""

from __future__ import annotations

import numpy as np


def relative_change(new: np.ndarray, old: np.ndarray, eps_0: float,
                    sigma: float = 0.0, noisy_entries: int = 0,
                    noise_new: float | None = None, noise_old: float | None = None) -> float:
    """Relative change of ``new`` vs ``old``, debiased for independent Gaussian noise.

    ``noise_new`` / ``noise_old`` are the expected noise energies (sum of the
    entries' sigma^2); by default both are noisy_entries * sigma^2.
    """
    if noise_new is None or noise_old is None:
        noise_new = noise_old = noisy_entries * sigma ** 2
    diff = np.linalg.norm(new - old) ** 2 - noise_new - noise_old
    base = np.linalg.norm(old) ** 2 - noise_old
    if noise_new == 0 and noise_old == 0:
        return float(np.linalg.norm(new - old) / (np.linalg.norm(old) + eps_0))
    return float(np.sqrt(max(0.0, diff)) / np.sqrt(max(eps_0, base)))


def weighted_loss(losses: dict[str, float], sizes: dict[str, int]) -> float:
    """L_glob = sum_k (N_k / Σ_j N_j) L_k over the clients that reported."""
    total = sum(sizes[c] for c in losses)
    return float(sum(sizes[c] / total * losses[c] for c in losses))


class ConvergenceMonitor:
    def __init__(self, eps_z: float, eps_l: float, patience: int, eps_0: float,
                 sigma: float = 0.0, clip_norm: float = 1.0):
        self.eps_l, self.patience, self.eps_0, self.sigma = eps_l, patience, eps_0, sigma
        self.eps_z = eps_z + (sigma / clip_norm if sigma > 0 else 0.0)
        self.counter = 0
        self._prev_Z: dict[str, np.ndarray] | None = None
        self._prev_sigma: dict[str, float] = {}
        self._prev_L: float | None = None

    def delta_z(self, Z_G: dict[str, np.ndarray],
                sigmas: dict[str, float] | None = None) -> float | None:
        """Compare with the previous round on the clients present in both.

        ``sigmas`` is each client's noise std this round (it differs by client
        and round with adaptive clipping); by default the monitor's sigma.
        """
        sigmas = sigmas if sigmas is not None else {k: self.sigma for k in Z_G}
        prev, self._prev_Z = self._prev_Z, {k: v.copy() for k, v in Z_G.items()}
        prev_sigma, self._prev_sigma = self._prev_sigma, dict(sigmas)
        if prev is None:
            return None
        common = [c for c in Z_G if c in prev]
        if not common:
            return None
        new = np.vstack([Z_G[c] for c in common])
        old = np.vstack([prev[c] for c in common])
        noise_new = sum(Z_G[c].size * sigmas[c] ** 2 for c in common)
        noise_old = sum(prev[c].size * prev_sigma[c] ** 2 for c in common)
        return relative_change(new, old, self.eps_0, noise_new=noise_new, noise_old=noise_old)

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
