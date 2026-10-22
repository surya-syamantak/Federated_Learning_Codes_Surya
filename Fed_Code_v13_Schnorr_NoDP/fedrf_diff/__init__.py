"""FedRF-Diff: federated random-forest diffusion maps with Procrustes alignment,
a Schnorr ZKP and no differential privacy (Algorithm 1)."""

from .config import Config
from .pipeline import RunResult, run

__all__ = ["Config", "RunResult", "run"]
