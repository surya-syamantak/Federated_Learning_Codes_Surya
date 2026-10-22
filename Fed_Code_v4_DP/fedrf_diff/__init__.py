"""FedRF-Diff: federated random-forest diffusion maps with Procrustes alignment,
Gaussian latent noise and ZKP vector authentication (Algorithm 1)."""

from .config import Config
from .pipeline import RunResult, run

__all__ = ["Config", "RunResult", "run"]
