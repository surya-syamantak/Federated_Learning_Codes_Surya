"""Stage 1.6: bounded-update enforcement.

Before the server aligns and aggregates a client's update (its latent matrix
Z_k = [patient rows; landmark rows] and its training labels), it checks that
the update lies inside a public bound that every honest update satisfies. An
update outside the bound is rejected for that round, like a failed ZKP, so it
never reaches the global model. The bound depends on how the latents are
released:

* No DP (latents sent as computed). The diffusion map is Z = sqrt(N_tot) U diag(mu)^gamma
  with orthonormal eigenvector columns U and eigenvalues 0 <= mu_j <= 1, so
  every honest update satisfies, exactly,
      |z_rj| <= sqrt(N_tot)              (entry bound; |U_rj| <= 1)
      ||Z||_F^2 <= N_tot·d           (energy bound; sum_r Z_rj^2 = N_tot mu_j^(2 gamma))
  with n = patient rows + landmark rows.
* Laplace DP (rows clipped to L1 norm C, then Lap(0, b) noise on the patient rows).
      landmark rows: ||p||_1 <= C   (clipped, not noised: exact)
      patient rows:  mean ||z~||_2^2 <= C^2 + 2 b^2 d + TAIL * sd,
  sd = sqrt((20 b^4 d + 8 C^2 b^2) / n_patients) (variance of ||z + e||^2 for
  ||z||_2 <= C and e ~ Lap(0, b)^d), so an honest update is rejected with
  negligible probability.
* Gaussian DP (rows clipped to L2 norm C_r, then N(0, sigma_r^2) noise).
      landmark rows: ||p||_2 <= C_r
      patient rows:  mean ||z~||_2^2 <= C_r^2 + d sigma_r^2 + TAIL * sd,
  sd = sqrt((2 d sigma_r^4 + 4 C_r^2 sigma_r^2) / n_patients), with the declared
  C_r itself capped by the largest honest row norm sqrt(N_tot d).

Every mode also checks the shape (patient rows = number of labels, landmark
rows = the public cohort size, d columns), that every value is finite and
that the labels are 0 or 1.

All bounded-update settings live here; there is no command-line option for them.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

# --- bounded-update settings ---------------------------------------------------
ENABLED = True  # check every authenticated update before aggregation
TAIL = 8.0  # standard deviations of tail allowance for the noisy (statistical) bounds
RTOL = 1e-9  # relative tolerance of the exact bounds (floating point)


def settings() -> dict:
    return {"enabled": ENABLED, "tail_sd": TAIL, "rtol": RTOL}


@dataclass(frozen=True)
class UpdateBound:
    """The public bound one client's update must satisfy this round."""

    mode: str  # "none", "laplace" or "gaussian"
    n_landmarks: int
    d_z: int
    entry_max: float = math.inf  # |z| for every value
    frob_sq_max: float = math.inf  # ||Z||_F^2 over all rows
    landmark_norm_ord: int | None = None  # 1 or 2: norm of every landmark row
    landmark_norm_max: float = math.inf
    patient_mean_sq_max: float = math.inf  # mean ||z~||_2^2 over the patient rows
    landmark_mean_sq_max: float = math.inf  # the same for noised landmark rows (if ever noised)
    invalid: str = ""  # a declared release parameter that no honest client can have
    description: str = ""


def bound_none(n_rows: int, n_landmarks: int, d_z: int) -> UpdateBound:
    """No DP: the exact diffusion-map bounds (n_rows = patient + landmark rows)."""
    return UpdateBound("none", n_landmarks, d_z, entry_max=math.sqrt(n_rows) * (1 + RTOL),
                       frob_sq_max=n_rows * d_z * (1 + RTOL),
                       description=f"|z| <= sqrt(N_tot) = {math.sqrt(n_rows):.3f}, ||Z||_F^2 <= N_tot d = {n_rows * d_z}")


def _noisy_mean_sq_max(mean: float, var_per_row: float, rows: int) -> float:
    return mean + TAIL * math.sqrt(var_per_row / max(rows, 1))


def bound_laplace(n_patients: int, n_landmarks: int, d_z: int, clip_l1: float, scale: float,
                  landmarks_noised: bool = False) -> UpdateBound:
    """Laplace DP: L1 clip C on every row, Lap(0, b) noise on the patient rows."""
    mean = clip_l1**2 + 2 * scale**2 * d_z
    var = 20 * scale**4 * d_z + 8 * clip_l1**2 * scale**2
    patient = _noisy_mean_sq_max(mean, var, n_patients)
    if landmarks_noised:
        return UpdateBound("laplace", n_landmarks, d_z, patient_mean_sq_max=patient,
                           landmark_mean_sq_max=_noisy_mean_sq_max(mean, var, n_landmarks),
                           description=f"mean ||z||^2 <= {patient:.4f} (b = {scale:.4f}, noised landmarks)")
    return UpdateBound("laplace", n_landmarks, d_z, landmark_norm_ord=1, landmark_norm_max=clip_l1 * (1 + RTOL),
                       patient_mean_sq_max=patient,
                       description=f"landmark ||p||_1 <= C = {clip_l1:g}, patient mean ||z||^2 <= "
                                   f"{patient:.4f} (b = {scale:.4f})")


def bound_gaussian(n_patients: int, n_landmarks: int, d_z: int, clip_l2: float, sigma: float,
                   noise_ratio: float | None = None, landmarks_noised: bool = False) -> UpdateBound:
    """Gaussian DP: L2 clip C_r on every row, N(0, sigma_r^2) noise on the patient rows.

    C_r and sigma_r are the client's declared release parameters (adaptive
    clipping). An honest sigma_r equals ``noise_ratio`` * C_r (fixed by the
    budget), and an honest C_r never needs to exceed the largest possible
    latent row norm sqrt(N_tot d); otherwise the update is out of bound.
    """
    n_rows = n_patients + n_landmarks
    invalid = ""
    if noise_ratio is not None and abs(sigma - noise_ratio * clip_l2) > 1e-9 * max(1.0, sigma):
        invalid = f"declared sigma_r = {sigma:.6g} != (sigma / C) * C_r = {noise_ratio * clip_l2:.6g}"
    elif clip_l2 > math.sqrt(n_rows * d_z) * (1 + RTOL):
        invalid = f"declared clip norm C_r = {clip_l2:.4g} > sqrt(N_tot d) = {math.sqrt(n_rows * d_z):.4g}"
    mean = clip_l2**2 + d_z * sigma**2
    var = 2 * d_z * sigma**4 + 4 * clip_l2**2 * sigma**2
    patient = _noisy_mean_sq_max(mean, var, n_patients)
    landmark = dict(landmark_mean_sq_max=_noisy_mean_sq_max(mean, var, n_landmarks)) if landmarks_noised else \
        dict(landmark_norm_ord=2, landmark_norm_max=clip_l2 * (1 + RTOL))
    return UpdateBound("gaussian", n_landmarks, d_z, patient_mean_sq_max=patient, invalid=invalid, **landmark,
                       description=f"landmark ||p||_2 <= C_r = {clip_l2:.4f}, patient mean ||z||^2 <= "
                                   f"{patient:.4f} (sigma_r = {sigma:.4f})")


@dataclass
class BoundCheck:
    """Outcome of one bounded-update check."""

    within: bool
    reasons: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    check_s: float = 0.0

    @property
    def outcome(self) -> str:
        return "Within bound" if self.within else "Out of bound"


def usage(check: BoundCheck, bound: UpdateBound) -> float:
    """Largest share (%) of any applicable bound that the update uses (> 100 = out of bound)."""
    pairs = [("entry_abs_max", bound.entry_max), ("frob_sq", bound.frob_sq_max),
             ("landmark_norm_max", bound.landmark_norm_max), ("patient_mean_sq", bound.patient_mean_sq_max),
             ("landmark_mean_sq", bound.landmark_mean_sq_max)]
    shares = [100 * check.stats[k] / b for k, b in pairs if k in check.stats and math.isfinite(b) and b > 0]
    return max(shares) if shares else float("nan")


def check_update(Z_tilde: np.ndarray, P: np.ndarray, y: np.ndarray, bound: UpdateBound) -> BoundCheck:
    """Check one update (patient rows Z_tilde, landmark rows P, labels y) against ``bound``."""
    t0 = time.perf_counter()
    reasons: list[str] = []
    Z_tilde, P, y = np.asarray(Z_tilde), np.asarray(P), np.asarray(y)
    if Z_tilde.ndim != 2 or P.ndim != 2 or Z_tilde.shape[1] != bound.d_z or P.shape[1] != bound.d_z:
        reasons.append(f"latent rows must have d = {bound.d_z} columns")
    if len(P) != bound.n_landmarks:
        reasons.append(f"expected {bound.n_landmarks} landmark rows, got {len(P)}")
    if len(y) != len(Z_tilde):
        reasons.append(f"{len(Z_tilde)} patient rows but {len(y)} labels")
    if len(y) and not np.isin(y, (0, 1)).all():
        reasons.append("labels must be 0 or 1")
    if bound.invalid:
        reasons.append(bound.invalid)
    stats: dict = {}
    if not reasons:
        full = np.vstack([Z_tilde, P])
        if not np.isfinite(full).all():
            reasons.append("non-finite values")
        else:
            stats["entry_abs_max"] = float(np.abs(full).max()) if full.size else 0.0
            stats["frob_sq"] = float(np.sum(full**2))
            if stats["entry_abs_max"] > bound.entry_max:
                reasons.append(f"entry |z| = {stats['entry_abs_max']:.4g} > {bound.entry_max:.4g}")
            if stats["frob_sq"] > bound.frob_sq_max:
                reasons.append(f"||Z||_F^2 = {stats['frob_sq']:.4g} > {bound.frob_sq_max:.4g}")
            if bound.landmark_norm_ord is not None and len(P):
                norms = np.linalg.norm(P, ord=bound.landmark_norm_ord, axis=1)
                stats["landmark_norm_max"] = float(norms.max())
                if stats["landmark_norm_max"] > bound.landmark_norm_max:
                    reasons.append(f"landmark L{bound.landmark_norm_ord} norm {stats['landmark_norm_max']:.4g} "
                                   f"> {bound.landmark_norm_max:.4g}")
            if math.isfinite(bound.patient_mean_sq_max) and len(Z_tilde):
                stats["patient_mean_sq"] = float(np.mean(np.sum(Z_tilde**2, axis=1)))
                if stats["patient_mean_sq"] > bound.patient_mean_sq_max:
                    reasons.append(f"patient mean ||z||^2 = {stats['patient_mean_sq']:.4g} "
                                   f"> {bound.patient_mean_sq_max:.4g}")
            if math.isfinite(bound.landmark_mean_sq_max) and len(P):
                stats["landmark_mean_sq"] = float(np.mean(np.sum(P**2, axis=1)))
                if stats["landmark_mean_sq"] > bound.landmark_mean_sq_max:
                    reasons.append(f"landmark mean ||p||^2 = {stats['landmark_mean_sq']:.4g} "
                                   f"> {bound.landmark_mean_sq_max:.4g}")
    return BoundCheck(within=not reasons, reasons=reasons, stats=stats, check_s=time.perf_counter() - t0)
