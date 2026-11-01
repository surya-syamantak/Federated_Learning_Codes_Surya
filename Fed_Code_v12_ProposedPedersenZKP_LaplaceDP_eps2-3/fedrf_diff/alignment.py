"""Stage 2 alignment: orthogonal Procrustes onto the reference landmarks.

    A* = argmin_A || P_k A - P_ref ||_F^2   s.t.  A^T A = I

Solution: with P_k^T P_ref = U S V^T, A* = U V^T.
"""

from __future__ import annotations

import numpy as np


def orthogonal_procrustes(P_c: np.ndarray, P_ref: np.ndarray) -> np.ndarray:
    if P_c.shape != P_ref.shape:
        raise ValueError(f"landmark shapes differ: {P_c.shape} vs {P_ref.shape}")
    U, _, Vt = np.linalg.svd(P_c.T @ P_ref)
    return U @ Vt
