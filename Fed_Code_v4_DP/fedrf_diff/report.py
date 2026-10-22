"""Terminal reports: per-round ZKP times and metrics, averages and ZKP time analysis.

"Global Server" is the global model M_G scored on all clients' validation
patients together, from the TP/FP/TN/FN counts the clients report (the
"All clients (pooled)" row of final_metrics.csv).
"""

from __future__ import annotations

import pandas as pd

POOLED = "All clients (pooled)"
GLOBAL = "Global Server"
# Terminal headings: full metric names with capitalised initials.
METRIC_NAMES = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1 Score",
    "mcc": "Matthews Correlation Coefficient (MCC)",
    "kappa": "Cohen's Kappa",
}
ZKP_TIMES = {
    "authentication_s": "ZKP Authentication Time (s)",
    "proof_generation_s": "ZKP Proof Generation Time (s)",
    "proof_verification_s": "ZKP Proof Verification Time (s)",
}


def clients_and_global(table: pd.DataFrame) -> pd.DataFrame:
    """Client rows plus the pooled row relabelled "Global Server" (no macro row)."""
    rows = table[table["client"] != "Macro average"].copy()
    rows["client"] = rows["client"].replace({POOLED: GLOBAL})
    return rows


def metrics_text(table: pd.DataFrame) -> str:
    """Client / Global Server rows with the seven metrics under their full names."""
    shown = table[["client", *METRIC_NAMES]].rename(columns={"client": "Client", **METRIC_NAMES})
    return shown.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def zkp_round_text(sessions: dict) -> str:
    rows = [{"Client": name, "Status": "Successful" if s.authenticated else "Failed",
             "Vectors": s.vectors,
             ZKP_TIMES["authentication_s"]: s.authentication_s,
             ZKP_TIMES["proof_generation_s"]: s.proof_generation_s,
             ZKP_TIMES["proof_verification_s"]: s.proof_verification_s}
            for name, s in sessions.items()]
    return pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}")


def _fmt(x: float | None, spec: str = ".4f") -> str:
    return "n/a" if x is None else format(x, spec)


def format_round(t: int, sessions: dict, final_metrics: pd.DataFrame | None, row: dict | None,
                 zkp_enabled: bool, eps_z: float = 0.0, patience: int = 0,
                 epsilon: float | None = None, delta: float = 0.0) -> str:
    out = ["", "=" * 30 + f" Round {t} " + "=" * 30]
    if zkp_enabled:
        out += ["ZKP authentication (per client):", zkp_round_text(sessions)]
    else:
        out.append("ZKP authentication: disabled")
    if final_metrics is None:
        return "\n".join(out)
    out += ["", f"Validation metrics of the global model after round {t}:",
            metrics_text(clients_and_global(final_metrics))]
    dp = (f" | DP budget spent per vector: epsilon = {row['epsilon_spent']:.2f} "
          f"(delta = {delta:g}, {epsilon:g} per round)" if epsilon is not None else "")
    out.append(f"Server: Delta Z = {_fmt(row['delta_z'])} (threshold {eps_z:.3f}) | "
               f"L_avg = {row['L_avg']:.4f} | Delta L = {_fmt(row['delta_l'])} | "
               f"patience {row['patience']}/{patience} | round time {row['round_seconds']:.1f} s" + dp)
    return "\n".join(out)


def averaged_over_rounds(round_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean of every metric over the rounds, per client and for the Global Server."""
    rows = clients_and_global(round_metrics)
    return rows.groupby("client", sort=False)[["log_loss", *METRIC_NAMES]].mean().reset_index()


def zkp_time_analysis(zkp_log: pd.DataFrame) -> pd.DataFrame:
    """Min, max and mean of the three ZKP times per client over the rounds."""
    stats = zkp_log.groupby("client", sort=False)[list(ZKP_TIMES)].agg(["min", "max", "mean"])
    stats.columns = [f"{ZKP_TIMES[k].replace(' (s)', '')} {s.capitalize()} (s)" for k, s in stats.columns]
    return stats.reset_index().rename(columns={"client": "Client"})


def format_end_of_run(result) -> str:
    out = [""]
    if result.round_metrics is not None and len(result.round_metrics):
        out += [f"=== Metrics averaged over all {result.rounds_run} rounds ===",
                metrics_text(averaged_over_rounds(result.round_metrics))]
    if result.zkp_log is not None and len(result.zkp_log):
        log = result.zkp_log
        stats = zkp_time_analysis(log)
        out += ["", f"=== ZKP time analysis over {log['round'].nunique()} rounds "
                f"(min, max and mean per client) ==="]
        for key, label in ZKP_TIMES.items():
            name = label.replace(" (s)", "")
            cols = ["Client"] + [f"{name} {s} (s)" for s in ("Min", "Max", "Mean")]
            out += [f"{label}:", stats[cols].rename(columns=lambda c: c.replace(name + " ", ""))
                    .to_string(index=False, float_format=lambda v: f"{v:.4f}")]
        n_ok = int(log["authenticated"].sum())
        out.append(f"Authentications: {n_ok}/{len(log)} successful, {len(log) - n_ok} failed")
    else:
        out += ["", "=== ZKP time analysis ===", "ZKP disabled; no timing."]
    return "\n".join(out)
