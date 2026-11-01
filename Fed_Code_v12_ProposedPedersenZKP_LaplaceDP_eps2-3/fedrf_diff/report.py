"""Terminal reports: per-round ZKP, encoding and decoding times and metrics,
averages and the ZKP / encoding-decoding time analyses.

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
# ZKP authentications take milliseconds: shown in ms here (the CSV files keep seconds).
ZKP_TIMES = {
    "authentication_s": "ZKP Authentication Time (ms)",
    "proof_generation_s": "ZKP Proof Generation Time (ms)",
    "proof_verification_s": "ZKP Proof Verification Time (ms)",
}
# Client-side encoding (RF kernel, diffusion map, Laplace DP) and decoding
# (k-NN reconstruction of the aligned latents, before RF retraining), in ms.
CODEC_TIMES = {
    "encode_s": "Encoding Time (ms)",
    "decode_s": "Decoding Time (ms)",
}
# The same times as columns of costs.csv.
CODEC_COSTS = {"encode_dp_s": "encode_s", "decode_s": "decode_s"}
MS = 1000.0


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
    rows = [{"Client": name, "Proof": s.outcome, "Vectors": s.vectors,
             **{label: getattr(s, key) * MS for key, label in ZKP_TIMES.items()}}
            for name, s in sessions.items()]
    return pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}")


def codec_round_text(timings: dict) -> str:
    """Per client: this round's encoding and decoding times (n/a if the client did not decode)."""
    rows = [{"Client": name, **{label: t[key] * MS if key in t else None
                                for key, label in CODEC_TIMES.items()}}
            for name, t in timings.items()]
    return pd.DataFrame(rows).to_string(index=False, na_rep="n/a",
                                        float_format=lambda v: f"{v:.4f}")


def _status(ok: bool) -> str:
    return ("Authentication Successful (proof valid, update accepted)" if ok
            else "Authentication Failed (proof invalid, update rejected)")


def zkp_status_text(sessions: dict) -> str:
    width = max(len(n) for n in sessions)
    return "\n".join(f"  {name:<{width}} : {_status(s.authenticated)}" for name, s in sessions.items())


def zkp_authentication_summary(zkp_log: pd.DataFrame) -> str:
    """Per client: rounds, successful and failed authentications, and the failed rounds."""
    rows = []
    for name, g in zkp_log.groupby("client", sort=False):
        failed = g.loc[~g["authenticated"].astype(bool), "round"].tolist()
        rows.append({"Client": name, "Rounds": len(g), "Successful": len(g) - len(failed),
                     "Failed": len(failed),
                     "ZKP Authentication": ("Successful in every round" if not failed
                                            else "Failed in round(s) " + ", ".join(map(str, failed)))})
    return pd.DataFrame(rows).to_string(index=False)


def _fmt(x: float | None, spec: str = ".4f") -> str:
    return "n/a" if x is None else format(x, spec)


def format_round(t: int, sessions: dict, final_metrics: pd.DataFrame | None, row: dict | None,
                 zkp_enabled: bool, eps_z: float = 0.0, patience: int = 0,
                 epsilon: float | None = None, delta: float = 0.0, codec: dict | None = None, bound_checks: dict | None = None) -> str:
    out = ["", "=" * 30 + f" Round {t} " + "=" * 30]
    if zkp_enabled:
        out += ["ZKP client authentication (per client):", zkp_round_text(sessions), "",
                "ZKP authentication status:", zkp_status_text(sessions)]
    else:
        out.append("ZKP authentication: disabled")
    if codec:
        out += ["", "Encoding and decoding times (per client):", codec_round_text(codec)]
    if bound_checks:
        out += ["", "Bounded-update enforcement (per client):", bounds_round_text(bound_checks)]
    if final_metrics is None:
        return "\n".join(out)
    out += ["", f"Validation metrics of the global model after round {t}:",
            metrics_text(clients_and_global(final_metrics))]
    dp = (f" | DP budget spent per vector: epsilon = {row['epsilon_spent']:.2f} "
          f"(pure DP; epsilon = {epsilon:.3f} this round)" if epsilon is not None else "")
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
    stats = zkp_log.groupby("client", sort=False)[list(ZKP_TIMES)].agg(["min", "max", "mean"]) * MS
    stats.columns = [f"{ZKP_TIMES[k].replace(' (ms)', '')} {s.capitalize()} (ms)" for k, s in stats.columns]
    return stats.reset_index().rename(columns={"client": "Client"})


def codec_time_analysis(costs: pd.DataFrame) -> pd.DataFrame:
    """Min, max and mean of the encoding and decoding times per client over the rounds."""
    clients = costs[costs["client"] != GLOBAL].rename(columns=CODEC_COSTS)
    stats = clients.groupby("client", sort=False)[list(CODEC_TIMES)].agg(["min", "max", "mean"]) * MS
    stats.columns = [f"{CODEC_TIMES[k].replace(' (ms)', '')} {s.capitalize()} (ms)" for k, s in stats.columns]
    return stats.reset_index().rename(columns={"client": "Client"})


def _time_tables(stats: pd.DataFrame, labels: dict) -> list[str]:
    """One min / max / mean table per time in ``labels``."""
    out = []
    for label in labels.values():
        name = label.replace(" (ms)", "")
        cols = ["Client"] + [f"{name} {s} (ms)" for s in ("Min", "Max", "Mean")]
        out += [f"{label}:", stats[cols].rename(columns=lambda c: c.replace(name + " ", ""))
                .to_string(index=False, float_format=lambda v: f"{v:.4f}")]
    return out


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
        out += _time_tables(stats, ZKP_TIMES)
        out += ["", "=== ZKP authentication summary ===", zkp_authentication_summary(log)]
        n_ok = int(log["authenticated"].sum())
        out.append(f"ZKP authentications: {n_ok}/{len(log)} successful (updates accepted), "
                   f"{len(log) - n_ok} failed (updates rejected)")
    else:
        out += ["", "=== ZKP time analysis ===", "ZKP disabled; no timing."]
    costs = getattr(result, "costs", None)
    if costs is not None and len(costs) and "decode_s" in costs:
        out += ["", f"=== Encoding and decoding time analysis over {costs['round'].nunique()} rounds "
                f"(min, max and mean per client) ==="]
        out += _time_tables(codec_time_analysis(costs), CODEC_TIMES)
    return "\n".join(out)


# --- bounded-update enforcement (bounds.py) ------------------------------------------
_BOUND_MS = 1000.0


def bounds_round_text(bound_checks: dict) -> str:
    """Per client: within / out of bound, the share of the bound used, check time, reason."""
    from . import bounds

    rows = [{"Client": name, "Update": chk.outcome, "Bound Used (%)": bounds.usage(chk, b),
             "Check Time (ms)": chk.check_s * _BOUND_MS, "Reason": "; ".join(chk.reasons) or "-"}
            for name, (chk, b) in bound_checks.items()]
    return pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}")


def bounds_summary(result) -> str:
    """End-of-run summary of the bounded-update enforcement."""
    log = getattr(result, "bounds_log", None)
    out = ["", "=== Bounded-update enforcement ==="]
    if log is None or not len(log):
        return "\n".join(out + ["Bounded-update enforcement disabled."])
    rows = []
    for name, g in log.groupby("client", sort=False):
        out_rounds = g.loc[~g["within"].astype(bool), "round"].tolist()
        rows.append({"Client": name, "Rounds": len(g), "Within Bound": len(g) - len(out_rounds),
                     "Out Of Bound": len(out_rounds), "Max Bound Used (%)": g["usage_pct"].max(),
                     "Mean Check Time (ms)": g["check_s"].mean() * _BOUND_MS,
                     "Bounded Update": ("Within bound in every round" if not out_rounds
                                        else "Rejected in round(s) " + ", ".join(map(str, out_rounds)))})
    out.append(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ok = int(log["within"].astype(bool).sum())
    out.append(f"Bounded updates: {ok}/{len(log)} within bound (accepted), {len(log) - ok} out of bound (rejected); "
               f"bound mode: {log['mode'].iloc[0]}")
    return "\n".join(out)
