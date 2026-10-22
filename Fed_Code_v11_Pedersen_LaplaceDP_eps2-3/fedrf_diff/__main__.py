"""Command line: ``python -m fedrf_diff`` (seed 42; no options needed).

Every Config field is an option (e.g. ``--max-rounds 40``). The Laplace DP
settings are in fedrf_diff/privacy.py and the ZKP settings in
fedrf_diff/zkp.py; neither takes a command-line option.

One run prints every round's ZKP times and metrics, then the round averages
and the ZKP time analysis, runs the accuracy-vs-DP-budget sweep (in parallel
processes, without ZKP, for the same number of rounds) and writes the CSV
files and figures to --output-dir.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

import pandas as pd

from . import privacy
from .config import Config
from .pipeline import run, sweep_point, sweep_row
from .plotting import (plot_accuracy_vs_dp_budget, plot_communication_cost, plot_computation_cost,
                       plot_convergence, plot_zkp_times)
from .report import format_end_of_run


def _parse_bool(text: str) -> bool:
    if text.lower() in ("1", "true", "yes", "on"):
        return True
    if text.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")


def parse_config(argv: list[str] | None) -> Config:
    parser = argparse.ArgumentParser(prog="fedrf_diff", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = Config()
    for f in fields(Config):
        default = getattr(defaults, f.name)
        kind = _parse_bool if isinstance(default, bool) else type(default)
        parser.add_argument("--" + f.name.replace("_", "-"), type=kind, default=default,
                            help=f"(default: {default})")
    args, unknown = parser.parse_known_args(argv)
    if any(a.startswith(("--sigma", "--epsilon", "--delta", "--noise")) for a in unknown):
        parser.error("the DP settings are in fedrf_diff/privacy.py, not on the command line")
    if any(a.startswith("--zkp") for a in unknown):
        parser.error("the ZKP settings are in fedrf_diff/zkp.py, not on the command line")
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return Config(**vars(args))


def dp_budget_sweep(cfg: Config, main_result, rounds: int) -> pd.DataFrame:
    """Accuracy at every budget in privacy.EPSILON_SWEEP, same seed and rounds.

    Runs the budgets in parallel processes; if that is not possible (for
    example a worker process fails on Windows) it runs them one at a time.
    """
    todo = [e for e in privacy.EPSILON_SWEEP if e != main_result.epsilon]
    order_key = lambda e: (2, 0) if e is None else (1, 0) if e == privacy.RANDOM else (0, e)  # noqa: E731
    workers = max(1, min(len(todo), (os.cpu_count() or 2) - 1, 4))
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            points = dict(zip(todo, pool.map(sweep_point, [cfg] * len(todo), todo, [rounds] * len(todo))))
    except Exception as err:  # noqa: BLE001 - fall back to the sequential sweep
        print(f"  parallel sweep unavailable ({type(err).__name__}: {err}); running the budgets one at a time")
        points = {}
        for e in todo:
            print(f"  epsilon = {'No DP' if e is None else f'{e:g}'} ...", flush=True)
            points[e] = sweep_point(cfg, e, rounds)
    points[main_result.epsilon] = sweep_row(main_result)
    order = sorted(points, key=order_key)
    return pd.DataFrame([points[e] for e in order])


def _figure(path: Path, draw, *args) -> bool:
    """Draw one figure (PNG + PDF); report the path, or the error without stopping the run."""
    try:
        draw(*args, path)
    except Exception as err:  # noqa: BLE001 - one figure must not stop the others
        print(f"  FAILED {path.name}: {type(err).__name__}: {err}")
        return False
    print(f"  saved {path.resolve()} (+ .pdf)")
    return True


def main(argv: list[str] | None = None) -> None:
    cfg = parse_config(argv)
    out = Path(cfg.output_dir)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out / "config.json")

    result = run(cfg)
    result.history.to_csv(out / "history.csv", index=False)
    result.final_metrics.to_csv(out / "final_metrics.csv", index=False)
    result.global_history.to_csv(out / "global_server_metrics.csv", index=False)
    result.costs.to_csv(out / "costs.csv", index=False)
    if result.zkp_log is not None:
        result.zkp_log.to_csv(out / "zkp_times.csv", index=False)
    print(format_end_of_run(result))
    status = "Converged" if result.converged else "Stopped at max_rounds without converging"
    print(f"\n{status} after {result.rounds_run} rounds (seed {cfg.seed}).")

    # Figures that need only the main run are saved now, before the sweep.
    mech = result.clients[0].dp
    eps_z = cfg.eps_z + (mech.sigma / mech.clip_norm if mech.enabled else 0.0)
    clients = [c.name for c in result.clients]
    print("\nFigures (600-dpi PNG + vector PDF):")
    _figure(figs / "convergence.png", plot_convergence, result.history, eps_z, cfg.eps_l,
            result.rounds_run if result.converged else None)
    if result.zkp_log is not None:
        _figure(figs / "zkp_times.png", plot_zkp_times, result.zkp_log)
    _figure(figs / "computation_cost.png", plot_computation_cost, result.costs)
    _figure(figs / "communication_cost.png", plot_communication_cost, result.costs)

    print(f"\nAccuracy vs DP budget: {len(privacy.EPSILON_SWEEP)} budgets, "
          f"{result.rounds_run} rounds each, seed {cfg.seed} (no ZKP; takes a few minutes) ...", flush=True)
    sweep = dp_budget_sweep(cfg, result, result.rounds_run)
    sweep.to_csv(out / "dp_budget_sweep.csv", index=False)
    budget = sweep["budget"]
    for prefix, title in (("mean_accuracy[", f"Accuracy averaged over the {result.rounds_run} rounds"),
                          ("accuracy[", f"Accuracy after round {result.rounds_run}")):
        cols = [c for c in sweep.columns if c.startswith(prefix)]
        shown = sweep[["mean_epsilon_per_round", "laplace_scale", "epsilon_spent", *cols]].rename(columns=lambda c: (
            c.replace(prefix, "").replace("]", "").replace("All clients (pooled)", "Global Server")))
        shown.insert(0, "Epsilon per round", budget)
        print(f"{title}:")
        print(shown.rename(columns={"mean_epsilon_per_round": "Mean epsilon per round",
                             "laplace_scale": "Mean Laplace scale b", "epsilon_spent": "Epsilon spent"})
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    _figure(figs / "accuracy_vs_dp_budget.png", plot_accuracy_vs_dp_budget, sweep, result.history,
            clients, sweep_row(result)["label"])
    print(f"\nCSV files in {out.resolve()} and figures in {figs.resolve()}")


if __name__ == "__main__":
    main()
