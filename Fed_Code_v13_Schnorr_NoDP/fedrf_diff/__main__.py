"""Command line: ``python -m fedrf_diff`` (seed 42; no options needed).

Every Config field is an option (e.g. ``--max-rounds 40``). The ZKP settings
are in fedrf_diff/zkp.py and take no command-line option. This version uses
no differential privacy, so there are no DP options either.

One run prints every round's ZKP times and metrics, then the round averages
and the ZKP time analysis, and writes the CSV files and figures to
--output-dir.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from .config import Config
from .pipeline import run
from .plotting import plot_communication_cost, plot_computation_cost, plot_convergence, plot_zkp_times
from .report import format_end_of_run
from .report import bounds_summary


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
    if any(a.startswith(("--sigma", "--epsilon", "--delta", "--noise", "--dp")) for a in unknown):
        parser.error("this version uses no differential privacy, so there are no DP options")
    if any(a.startswith("--zkp") for a in unknown):
        parser.error("the ZKP settings are in fedrf_diff/zkp.py, not on the command line")
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return Config(**vars(args))


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
    if result.bounds_log is not None:
        result.bounds_log.to_csv(out / "bounds_log.csv", index=False)
    print(format_end_of_run(result))
    print(bounds_summary(result))
    status = "Converged" if result.converged else "Stopped at max_rounds without converging"
    print(f"\n{status} after {result.rounds_run} rounds (seed {cfg.seed}).")

    print("\nFigures (600-dpi PNG + vector PDF):")
    _figure(figs / "convergence.png", plot_convergence, result.history, cfg.eps_z, cfg.eps_l,
            result.rounds_run if result.converged else None)
    if result.zkp_log is not None:
        _figure(figs / "zkp_times.png", plot_zkp_times, result.zkp_log)
    _figure(figs / "computation_cost.png", plot_computation_cost, result.costs)
    _figure(figs / "communication_cost.png", plot_communication_cost, result.costs)
    print(f"\nCSV files in {out.resolve()} and figures in {figs.resolve()}")


if __name__ == "__main__":
    main()
