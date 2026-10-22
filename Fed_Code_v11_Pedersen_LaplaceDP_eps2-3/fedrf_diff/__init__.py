"""FedRF-Diff: federated random-forest diffusion maps with Procrustes alignment,
Laplace DP noise and a Pedersen ZKP for client verification (Algorithm 1)."""

from .config import Config
from .pipeline import RunResult, run

__all__ = ["Config", "RunResult", "run"]
