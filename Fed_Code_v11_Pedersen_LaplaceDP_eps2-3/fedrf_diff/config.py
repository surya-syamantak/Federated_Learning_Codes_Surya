"""Run configuration for the FedRF-Diff pipeline (seed 42 by default).

Every learning setting is a field here and a command-line option. The
Laplace DP settings live in privacy.py and the ZKP settings in zkp.py;
neither needs a command-line option.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from . import privacy, zkp
from . import bounds

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "heart_statlog_cleveland_hungary_final.csv"


@dataclass
class Config:
    # --- data -----------------------------------------------------------
    data_path: str = str(DEFAULT_DATA_PATH)
    # Drop the 2 VA rows with impossible values (resting bp = 0, ST slope = 0).
    drop_invalid_rows: bool = True
    # Keep every copy of a duplicated patient on the same side of the
    # train/validation split, so the validation loss is not inflated.
    group_aware_val_split: bool = True
    val_fraction: float = 0.2
    # Public landmark cohort (unique patients, removed from all clients) that
    # every client embeds; its rows give P_k and P_ref for Procrustes.
    # Fewer than ~50 makes the alignment flip between rounds (Delta Z jumps).
    n_landmarks: int = 100
    # Client whose round-1 landmark embedding becomes the frozen P_ref.
    reference_client: str = "Client 4(Cleveland)"

    # --- local random forest + diffusion map ------------------------------
    n_trees: int = 500  # B
    min_samples_leaf: int = 20
    max_features: str = "sqrt"
    d_z: int = 4  # latent dimension, d << d_x = 11
    tau: float = 1.0  # diffusion time gamma

    # --- global model M_glob (random forest on the aligned latents, server side)
    global_n_trees: int = 300
    # Large leaves keep M_glob smooth; with 5 the closed loop flip-flops (Delta Z ~0.25).
    global_min_samples_leaf: int = 50

    # --- privacy --------------------------------------------------------
    # Laplace DP is always applied; its settings live in privacy.py.

    # --- decoder --------------------------------------------------------
    k_neighbors: int = 10  # K

    # --- rounds and stopping rule -----------------------------------------
    max_rounds: int = 30  # T
    # Threshold on the noise-corrected Delta Z without DP noise (it plateaus
    # near 0.11). With DP the run stops on eps_z + sigma / C (see
    # convergence.py), which tracks the plateau at every privacy budget.
    eps_z: float = 0.15
    eps_l: float = 0.02  # threshold on Delta L
    eps_0: float = 1e-12  # stabiliser in the Delta Z denominator
    patience: int = 3  # k consecutive rounds

    # --- ZKP vector authentication: settings live in zkp.py ----------------

    # --- misc -----------------------------------------------------------
    seed: int = 42
    output_dir: str = "outputs/run"

    def to_json(self, path: str | Path) -> None:
        record = {**asdict(self), "privacy": privacy.settings(), "zkp": zkp.settings(), "bounds": bounds.settings()}
        Path(path).write_text(json.dumps(record, indent=2))
