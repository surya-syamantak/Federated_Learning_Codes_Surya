"""Stage 1: Gaussian differential privacy for the transmitted latent vectors.

Every round each client privatises every latent vector z_i (row of Ψ_k) it
sends to the server with the Gaussian mechanism:

    1. clip:   z_i <- z_i * min(1, C / ||z_i||_2)          (L2 norm <= C)
    2. noise:  z~_i = z_i + N(0, sigma^2 I)

After clipping, any two vectors differ by at most Delta = 2C in L2 norm (the
sensitivity), so one release is rho-zCDP with rho = Delta^2 / (2 sigma^2)
(Bun & Steinke 2016). sigma is calibrated so that one round meets the
per-round budget (EPSILON, DELTA):

    epsilon = rho + 2 sqrt(rho ln(1/delta))
    =>  sqrt(rho) = sqrt(EPSILON + ln(1/DELTA)) - sqrt(ln(1/DELTA)),
        sigma = Delta / sqrt(2 rho).

Releases compose additively in rho, so after t rounds the privacy spent on
each vector is (rho_r + 2 sqrt(rho_r ln(1/DELTA)), DELTA)-DP with
rho_r = t rho. The guarantee is per transmitted vector (local DP on what
leaves the client); it does not cover the training labels, which the server
needs to train the global model, nor the influence one patient has on other
patients' vectors through the shared kernel.

Adaptive clipping (ADAPTIVE_CLIP; Andrew et al. 2021, "Differentially
Private Learning with Adaptive Clipping"): each client tracks the
CLIP_QUANTILE-quantile of its own latent norms instead of using a fixed C.
Each round it also releases the noisy fraction of its vectors inside C_r,

    b~ = (sum_i 1[||z_i|| <= C_r] + N(0, sigma_b^2)) / n

(sensitivity 1, so rho_b = 1 / (2 sigma_b^2)), and moves the clip norm:
C_(t+1) = C_r * exp(-CLIP_LR * (b~ - CLIP_QUANTILE)). The round's rho is
split: CLIP_BUDGET_FRACTION * rho for b~, the rest for the vectors, with
sigma_r = 2 C_r / sqrt(2 rho_vec). The total per round is still rho, so the
epsilon spent is the same as with a fixed C; sigma / C is fixed by the budget
alone, so the stopping threshold eps_z + sigma / C does not change either.

All DP settings live here; there is no command-line option for them.
"""

from __future__ import annotations

import math

import numpy as np

# --- Gaussian DP settings --------------------------------------------------
EPSILON = 20.0  # per-round privacy budget of every transmitted latent vector
DELTA = 1e-5  # failure probability of the (epsilon, delta) guarantee
CLIP_NORM = 1.0  # C: L2 bound on each latent vector (sensitivity = 2C); C_0 if adaptive
# Adaptive clipping: each client's C tracks a quantile of its latent norms.
ADAPTIVE_CLIP = True
CLIP_QUANTILE = 0.5  # target fraction of vectors left unclipped
CLIP_LR = 0.5  # geometric learning rate of the C update
CLIP_BUDGET_FRACTION = 0.02  # share of each round's rho spent on the norm count
# Validation embeddings never leave the client, so they stay clean.
NOISE_ON_VALIDATION = False
# The 100 landmark vectors embed the public landmark cohort, not private
# patients, so they are clipped but sent without noise. Noisy landmarks make
# the Procrustes alignment jump every round (seed 42, epsilon = 20: accuracy
# 0.79 instead of 0.82, and Delta Z does not settle).
NOISE_ON_LANDMARKS = False
# Per-round budgets for the accuracy-vs-DP-budget figure (each is one extra
# learning run with the same seed; the ZKP does not affect accuracy, so these
# runs skip it). None = no DP.
EPSILON_SWEEP = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, None)


def rho_for_epsilon(epsilon: float, delta: float) -> float:
    """zCDP parameter rho whose (epsilon, delta) conversion equals ``epsilon``."""
    log_term = math.log(1.0 / delta)
    return (math.sqrt(epsilon + log_term) - math.sqrt(log_term)) ** 2


def epsilon_for_rho(rho: float, delta: float) -> float:
    """(epsilon, delta)-DP implied by rho-zCDP."""
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def gaussian_sigma(epsilon: float, delta: float, clip_norm: float) -> float:
    """Noise std for one (epsilon, delta) release of a vector clipped to clip_norm."""
    if epsilon <= 0 or not 0 < delta < 1 or clip_norm <= 0:
        raise ValueError("need epsilon > 0, 0 < delta < 1 and clip_norm > 0")
    return 2.0 * clip_norm / math.sqrt(2.0 * rho_for_epsilon(epsilon, delta))


def clip_rows(Z: np.ndarray, clip_norm: float) -> np.ndarray:
    """Scale every row to L2 norm at most clip_norm."""
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z * np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))


def gaussian_perturb(Z: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if sigma == 0:
        return Z.copy()
    return Z + rng.normal(0.0, sigma, size=Z.shape)


class GaussianMechanism:
    """Per-client Gaussian mechanism with a running privacy account.

    ``epsilon=None`` switches DP off (no clipping, no noise, nothing spent).
    With ``adaptive`` (default ADAPTIVE_CLIP) the clip norm follows the
    CLIP_QUANTILE of the client's latent norms; ``clip_norm`` is then C_0.
    """

    def __init__(self, epsilon: float | None = EPSILON, delta: float = DELTA,
                 clip_norm: float = CLIP_NORM, adaptive: bool | None = None):
        self.epsilon, self.delta, self.clip_norm = epsilon, delta, clip_norm
        self.adaptive = (ADAPTIVE_CLIP if adaptive is None else adaptive) and epsilon is not None
        self.rho = 0.0 if epsilon is None else rho_for_epsilon(epsilon, delta)
        share = CLIP_BUDGET_FRACTION if self.adaptive else 0.0
        self.rho_vec, self.rho_count = (1 - share) * self.rho, share * self.rho
        # sigma / C is fixed by the budget, whatever C is.
        self.noise_ratio = 0.0 if epsilon is None else 2.0 / math.sqrt(2.0 * self.rho_vec)
        self.releases = 0
        # Clip norm and sigma of the latest release (one entry per round).
        self.clip_history: list[float] = []
        self.sigma_history: list[float] = []

    @property
    def enabled(self) -> bool:
        return self.epsilon is not None

    @property
    def sigma(self) -> float:
        """Noise std for the current clip norm."""
        return self.noise_ratio * self.clip_norm

    @property
    def last_clip(self) -> float:
        """Clip norm of the latest release (C_0 before the first)."""
        return self.clip_history[-1] if self.clip_history else self.clip_norm

    def privatize(self, Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One round's release of Z: clip, add noise, charge the budget.

        With adaptive clipping it also releases the noisy fraction of rows
        inside the clip norm and moves the clip norm for the next round.
        """
        if not self.enabled:
            return Z.copy()
        self.releases += 1
        self.clip_history.append(self.clip_norm)
        self.sigma_history.append(self.sigma)
        out = gaussian_perturb(clip_rows(Z, self.clip_norm), self.sigma, rng)
        if self.adaptive:
            inside = np.sum(np.linalg.norm(Z, axis=1) <= self.clip_norm)
            sigma_b = 1.0 / math.sqrt(2.0 * self.rho_count)
            frac = (inside + rng.normal(0.0, sigma_b)) / len(Z)
            self.clip_norm *= math.exp(-CLIP_LR * (frac - CLIP_QUANTILE))
        return out

    def spent_epsilon(self) -> float:
        """Total epsilon spent so far (zCDP composition over the releases)."""
        if not self.enabled or self.releases == 0:
            return 0.0
        return epsilon_for_rho(self.releases * self.rho, self.delta)


def settings(epsilon: float | None = EPSILON) -> dict:
    """The DP settings, recorded in each run's config.json."""
    if epsilon is None:
        return {"mechanism": "none"}
    mech = GaussianMechanism(epsilon)
    adaptive = ({"clip_quantile": CLIP_QUANTILE, "clip_lr": CLIP_LR,
                 "clip_budget_fraction": CLIP_BUDGET_FRACTION} if ADAPTIVE_CLIP else {})
    return {"mechanism": "gaussian (clip + zCDP calibration)", "epsilon_per_round": epsilon,
            "delta": DELTA, "clip_norm": CLIP_NORM, "sensitivity": 2 * CLIP_NORM,
            "sigma": mech.sigma, "sigma_over_clip": mech.noise_ratio,
            "adaptive_clip": ADAPTIVE_CLIP, **adaptive,
            "noise_on_validation": NOISE_ON_VALIDATION, "noise_on_landmarks": NOISE_ON_LANDMARKS}
