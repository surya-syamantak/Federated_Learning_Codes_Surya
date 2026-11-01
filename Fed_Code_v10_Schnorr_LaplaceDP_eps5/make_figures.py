"""Generate all journal-quality figures from a finished run's CSV files.

Run the pipeline once (``python -m fedrf_diff``), then, from this folder:

    python make_figures.py                         # reads outputs/run
    python make_figures.py --run-dir outputs/run2  # another run's folder

No training is repeated: the figures are drawn from the CSV files the run
wrote (history.csv, zkp_times.csv, costs.csv, dp_budget_sweep.csv,
config.json). Each figure is saved as a 600-dpi PNG and a vector PDF in
<run-dir>/figures/:

    convergence.png            Delta Z, |Delta L| and L_glob per round
    zkp_times.png              ZKP authentication, proof generation and
                               proof verification time per client per round
    accuracy_vs_dp_budget.png  accuracy per client vs the DP budget
    computation_cost.png       computation cost per round
    communication_cost.png     communication cost per round
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from fedrf_diff.data import CLIENT_NAMES
from fedrf_diff.plotting import (plot_accuracy_vs_dp_budget, plot_communication_cost, plot_computation_cost,
                                 plot_convergence, plot_zkp_times)


def _main_label(privacy: dict, dp: bool) -> str | None:
    """x label of the main run in the accuracy-vs-DP-budget figure."""
    if not dp:
        return "No DP"
    eps = privacy.get("epsilon_per_round")
    if eps == "random":
        low, high = privacy["epsilon_range"]
        return f"U[{low:g}, {high:g}]"
    return f"{eps:g}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="outputs/run", help="folder of a finished run (default: outputs/run)")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    if not (run / "history.csv").exists():
        sys.exit(f"No run found in {run.resolve()}: run 'python -m fedrf_diff' first "
                 f"(or pass --run-dir).")

    cfg = json.loads((run / "config.json").read_text())
    privacy = cfg.get("privacy") or {}
    dp = isinstance(privacy, dict) and privacy.get("mechanism", "none") != "none"
    history = pd.read_csv(run / "history.csv")
    costs = pd.read_csv(run / "costs.csv")
    converged = bool(history["patience"].iloc[-1] >= cfg["patience"])
    # A per-round threshold column (random DP budget) takes precedence inside plot_convergence.
    eps_z = cfg["eps_z"] + (privacy["sigma"] / privacy["clip_norm"] if dp and "sigma" in privacy else 0.0)
    figs = run / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    jobs = [("convergence.png", plot_convergence,
             (history, eps_z, cfg["eps_l"], len(history) if converged else None)),
            ("computation_cost.png", plot_computation_cost, (costs,)),
            ("communication_cost.png", plot_communication_cost, (costs,))]
    if (run / "zkp_times.csv").exists():
        jobs.insert(1, ("zkp_times.png", plot_zkp_times, (pd.read_csv(run / "zkp_times.csv"),)))
    if (run / "dp_budget_sweep.csv").exists():
        jobs.insert(2, ("accuracy_vs_dp_budget.png", plot_accuracy_vs_dp_budget,
                        (pd.read_csv(run / "dp_budget_sweep.csv"), history, list(CLIENT_NAMES),
                         _main_label(privacy, dp))))
    else:
        print("dp_budget_sweep.csv not found (the run stopped before the DP-budget sweep): "
              "accuracy_vs_dp_budget.png skipped")

    failed = 0
    for name, draw, data in jobs:
        try:
            draw(*data, figs / name)
            print(f"saved {(figs / name).resolve()} (+ .pdf)")
        except Exception as err:  # noqa: BLE001 - report and continue with the others
            failed += 1
            print(f"FAILED {name}: {type(err).__name__}: {err}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
