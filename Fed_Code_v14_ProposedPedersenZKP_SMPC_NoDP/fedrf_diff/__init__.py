"""FedRF-Diff: federated random-forest diffusion maps with Procrustes alignment,
ZKP client authentication and SMPC secure aggregation, without DP (Algorithm 1)."""

from .config import Config
from .pipeline import RunResult, run

__all__ = ["Config", "RunResult", "run"]
