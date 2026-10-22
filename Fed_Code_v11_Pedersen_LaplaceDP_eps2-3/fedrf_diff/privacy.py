"""Stage 1: Laplace differential privacy for the transmitted latent vectors.

Every round each client privatises every latent vector z_i (row of Z_c) it
sends to the server with the Laplace mechanism:

    1. clip:   z_i <- z_i * min(1, C / ||z_i||_1)          (L1 norm <= C)
    2. noise:  z~_i = z_i + Lap(0, b) on every coordinate, b = Delta_1 / EPSILON

After clipping, any two vectors differ by at most Delta_1 = 2C in L1 norm
(the sensitivity), so one release is (EPSILON, 0)-DP: pure differential
privacy, no failure probability delta (Dwork et al. 2006). Releases compose
additively: after t rounds each vector has spent t * EPSILON.

The guarantee is per transmitted vector (local DP on what leaves the
client); it does not cover the training labels, which the server needs to
train the global model, nor the influence one patient has on other patients'
vectors through the shared kernel.

Per-round budget: EPSILON = "random" (the default) draws a new epsilon_t
uniformly from RANDOM_RANGE = [2, 3] every round of the main (ZKP) run
(reproducible from the run seed; all clients use the same epsilon_t that round;
spent = epsilon_1 + ... + epsilon_t). A number instead, e.g. EPSILON = 5, makes
each round (5, 0)-DP, so after t rounds each vector has spent 5t. Every budget
used (EPSILON, RANDOM_RANGE and the EPSILON_SWEEP budgets) must lie in
EPSILON_RANGE = [1, 5]; check_epsilon() enforces it.

All DP settings live here; there is no command-line option for them.
"""

from __future__ import annotations

import math

import numpy as np

# --- Laplace DP settings ---------------------------------------------------
EPSILON_RANGE = (1.0, 5.0)  # allowed per-round budgets
RANDOM = "random"
RANDOM_RANGE = (2.0, 3.0)  # epsilon_t ~ U(RANDOM_RANGE) each round when EPSILON = RANDOM
EPSILON = RANDOM  # per-round budget of every transmitted vector (a number, or RANDOM: new epsilon_t each round)
CLIP_NORM = 1.0  # C: L1 bound on each latent vector (sensitivity = 2C)
# Validation embeddings never leave the client, so they stay clean.
NOISE_ON_VALIDATION = False
# The 100 landmark vectors embed the public landmark cohort, not private
# patients, so they are clipped but sent without noise. Noisy landmarks make
# the Procrustes alignment jump every round, and Delta Z does not settle.
NOISE_ON_LANDMARKS = False
# Per-round budgets for the accuracy-vs-DP-budget figure (each is one extra
# learning run with the same seed; the ZKP does not affect accuracy, so these
# runs skip it). None = the no-DP reference point.
EPSILON_SWEEP = (1.0, 2.0, 3.0, 4.0, 5.0, None)


def check_epsilon(epsilon: float | str) -> None:
    """Raise unless epsilon is "random" or lies in EPSILON_RANGE."""
    low, high = EPSILON_RANGE
    if epsilon == RANDOM:
        if not low <= RANDOM_RANGE[0] <= RANDOM_RANGE[1] <= high:
            raise ValueError(f"RANDOM_RANGE = {list(RANDOM_RANGE)} must lie inside the allowed range "
                             f"[{low:g}, {high:g}] (privacy.EPSILON_RANGE)")
        return
    if not low <= epsilon <= high:
        raise ValueError(f"epsilon = {epsilon:g} per round is outside the allowed range "
                         f"[{low:g}, {high:g}] (privacy.EPSILON_RANGE)")


def round_epsilons(rounds: int, seed: int) -> np.ndarray:
    """epsilon_1..epsilon_rounds, each drawn uniformly from RANDOM_RANGE (reproducible)."""
    rng = np.random.default_rng([seed, 2024])  # own stream: does not disturb any other randomness
    return rng.uniform(*RANDOM_RANGE, size=rounds)


def laplace_scale(epsilon: float, clip_norm: float) -> float:
    """Laplace scale b for one epsilon-DP release of a vector clipped to L1 norm clip_norm."""
    if epsilon <= 0 or clip_norm <= 0:
        raise ValueError("need epsilon > 0 and clip_norm > 0")
    return 2.0 * clip_norm / epsilon


def clip_rows(Z: np.ndarray, clip_norm: float) -> np.ndarray:
    """Scale every row to L1 norm at most clip_norm."""
    norms = np.abs(Z).sum(axis=1, keepdims=True)
    return Z * np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))


def laplace_perturb(Z: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    if scale < 0:
        raise ValueError("scale must be >= 0")
    if scale == 0:
        return Z.copy()
    return Z + rng.laplace(0.0, scale, size=Z.shape)


class LaplaceMechanism:
    """Per-client Laplace mechanism with a running privacy account.

    ``epsilon=None`` switches DP off (no clipping, no noise, nothing spent);
    ``epsilon="random"`` expects set_epsilon() with each round's epsilon_t.
    ``sigma`` is the noise standard deviation per coordinate (sqrt(2) b),
    which the convergence monitor uses to discount the noise.
    """

    delta = 0.0  # pure DP

    def __init__(self, epsilon: float | str | None = EPSILON, clip_norm: float = CLIP_NORM):
        self.clip_norm = clip_norm
        self.random = epsilon == RANDOM
        self.releases, self.spent = 0, 0.0
        self.set_epsilon(RANDOM_RANGE[1] if self.random else epsilon)

    def set_epsilon(self, epsilon: float | None) -> None:
        """This round's budget (and hence the Laplace scale and noise std)."""
        self.epsilon = epsilon
        self.scale = 0.0 if epsilon is None else laplace_scale(epsilon, self.clip_norm)
        self.sigma = math.sqrt(2.0) * self.scale

    @property
    def enabled(self) -> bool:
        return self.epsilon is not None

    def privatize(self, Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One round's release of Z: clip, add noise, charge the budget."""
        if not self.enabled:
            return Z.copy()
        self.releases += 1
        self.spent += self.epsilon
        return laplace_perturb(clip_rows(Z, self.clip_norm), self.scale, rng)

    def spent_epsilon(self) -> float:
        """Total epsilon spent so far (basic composition: sum of the per-round epsilons)."""
        return self.spent


def settings(epsilon: float | str | None = EPSILON) -> dict:
    """The DP settings, recorded in each run's config.json."""
    if epsilon is None:
        return {"mechanism": "none"}
    if epsilon == RANDOM:
        return {"mechanism": "laplace (L1 clip, pure epsilon-DP)", "epsilon_per_round": RANDOM,
                "epsilon_range": list(RANDOM_RANGE), "delta": 0.0, "clip_norm": CLIP_NORM,
                "sensitivity": 2 * CLIP_NORM,
                "noise_on_validation": NOISE_ON_VALIDATION, "noise_on_landmarks": NOISE_ON_LANDMARKS}
    scale = laplace_scale(epsilon, CLIP_NORM)
    return {"mechanism": "laplace (L1 clip, pure epsilon-DP)", "epsilon_per_round": epsilon,
            "delta": 0.0, "clip_norm": CLIP_NORM, "sensitivity": 2 * CLIP_NORM,
            "laplace_scale": scale, "sigma": math.sqrt(2.0) * scale,
            "noise_on_validation": NOISE_ON_VALIDATION, "noise_on_landmarks": NOISE_ON_LANDMARKS}
