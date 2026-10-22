"""FedRF-Diff: federated random-forest diffusion maps with Procrustes alignment,
Laplace DP noise and ZKP client authentication (Algorithm 1)."""

from .config import Config
from .pipeline import RunResult, run

__all__ = ["Config", "RunResult", "run"]
