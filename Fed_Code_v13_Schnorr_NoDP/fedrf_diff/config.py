"""Run configuration for the FedRF-Diff pipeline (seed 42 by default).

Every learning setting is a field here and a command-line option. The ZKP
settings live in zkp.py and need no command-line option. This version uses
no differential privacy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from . import zkp
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
    # every client embeds; its rows give P_c and P_ref for Procrustes.
    # Fewer than ~50 makes the alignment flip between rounds (Delta Z jumps).
    n_landmarks: int = 100
    # Client whose round-1 landmark embedding becomes the frozen P_ref.
    reference_client: str = "Client 4(Cleveland)"

    # --- local random forest + diffusion map ------------------------------
    n_trees: int = 500  # B
    min_samples_leaf: int = 20
    max_features: str = "sqrt"
    d_z: int = 4  # latent dimension, d_z << d_x = 11
    tau: float = 1.0  # diffusion time

    # --- global model M_G (random forest on the aligned latents, server side)
    global_n_trees: int = 300
    # Large leaves keep M_G smooth; with 5 the closed loop flip-flops (Delta Z ~0.25).
    global_min_samples_leaf: int = 50

    # --- privacy --------------------------------------------------------
    # No differential privacy: latents are sent without clipping or noise.

    # --- decoder --------------------------------------------------------
    k_neighbors: int = 10  # K

    # --- rounds and stopping rule -----------------------------------------
    max_rounds: int = 30  # T
    # Threshold on Delta Z (without noise it plateaus near 0.11 on seed 42).
    eps_z: float = 0.15
    eps_l: float = 0.02  # threshold on Delta L
    eps_0: float = 1e-12  # stabiliser in the Delta Z denominator
    patience: int = 3  # k consecutive rounds

    # --- ZKP vector authentication: settings live in zkp.py ----------------

    # --- misc -----------------------------------------------------------
    seed: int = 42
    output_dir: str = "outputs/run"

    def to_json(self, path: str | Path) -> None:
        record = {**asdict(self), "privacy": {"mechanism": "none"}, "zkp": zkp.settings(), "bounds": bounds.settings()}
        Path(path).write_text(json.dumps(record, indent=2))
