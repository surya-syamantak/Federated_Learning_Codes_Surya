"""Terminal reports: per-round ZKP, SMPC, encoding and decoding times and metrics,
averages and the ZKP / SMPC / encoding-decoding time analyses.

"Global Server" is the global model M_G scored on all clients' validation
patients together, from the TP/FP/TN/FN totals the server obtains by the
proposed SMPC (adaptive prime secure aggregation with pairwise masking) (the "All clients (pooled)" row of final_metrics.csv). The
per-client metrics are the evaluation harness: computed on each client and
never sent to the server.
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
# Client-side encoding (RF kernel, diffusion map; no DP) and decoding
# (k-NN reconstruction of the aligned latents, before RF retraining), in ms.
CODEC_TIMES = {
    "encode_s": "Encoding Time (ms)",
    "decode_s": "Decoding Time (ms)",
}
# The same times as columns of costs.csv.
CODEC_COSTS = {"encode_s": "encode_s", "decode_s": "decode_s"}
# Proposed SMPC, in ms: per client (quantize + pairwise streams + self-mask + masked model,
# plus the one-time key setup in the first round) and for the server (aggregate + dequantize).
SMPC_TIMES = {
    "mask_s": "SMPC Masking Time (ms)",
    "smpc_s": "SMPC Total Time (ms)",
}
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


def smpc_round_text(agg) -> str:
    """Per client: masking time and bytes; then the prime, the aggregation and what the server learned."""
    rows = [{"Client": name, SMPC_TIMES["mask_s"]: p.mask_s * MS, "Masked Values": len(agg.masked.get(name, [])),
             "Bytes Up": p.bytes_up, "Bytes Down": p.bytes_down}
            for name, p in agg.clients.items()]
    table = pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}")
    t = agg.totals
    return (table + f"\n  Server: p_t = Pool[{agg.prime_index}] = {agg.prime}; aggregation + dequantization "
            f"{agg.aggregate_s * MS:.4f} ms; learned only the totals sum n_c L_c = {t['loss_x_n_train']:.4f}, "
            f"TP = {round(t['tp'])}, FP = {round(t['fp'])}, TN = {round(t['tn'])}, FN = {round(t['fn'])} "
            f"(no client's own loss or counts)")


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
                 codec: dict | None = None, smpc=None) -> str:
    out = ["", "=" * 30 + f" Round {t} " + "=" * 30]
    if zkp_enabled:
        out += ["ZKP client authentication (per client):", zkp_round_text(sessions), "",
                "ZKP authentication status:", zkp_status_text(sessions)]
    else:
        out.append("ZKP authentication: disabled")
    if codec:
        out += ["", "Encoding and decoding times (per client):", codec_round_text(codec)]
    if final_metrics is None:
        return "\n".join(out)
    if smpc is not None:
        out += ["", f"SMPC: adaptive prime secure aggregation with self-randomized pairwise masking "
                f"({len(smpc.participants)} participants):", smpc_round_text(smpc)]
    out += ["", f"Validation metrics of the global model after round {t} (clients: evaluation harness; "
            "Global Server: SMPC totals):", metrics_text(clients_and_global(final_metrics))]
    out.append(f"Server: Delta Z = {_fmt(row['delta_z'])} (threshold {eps_z:.3f}) | "
               f"L_avg = {row['L_avg']:.4f} | Delta L = {_fmt(row['delta_l'])} | "
               f"patience {row['patience']}/{patience} | round time {row['round_seconds']:.1f} s")
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


def smpc_time_analysis(smpc_log: pd.DataFrame) -> pd.DataFrame:
    """Min, max and mean of the SMPC times per client and for the server over the rounds."""
    stats = smpc_log.groupby("client", sort=False)[list(SMPC_TIMES)].agg(["min", "max", "mean"]) * MS
    stats.columns = [f"{SMPC_TIMES[k].replace(' (ms)', '')} {s.capitalize()} (ms)" for k, s in stats.columns]
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
    smpc_log = getattr(result, "smpc_log", None)
    if smpc_log is not None and len(smpc_log):
        clients = smpc_log[smpc_log["client"] != GLOBAL]
        server = smpc_log[smpc_log["client"] == GLOBAL]
        out += ["", f"=== SMPC time analysis over {smpc_log['round'].nunique()} rounds "
                f"(min, max and mean per client) ==="]
        out += _time_tables(smpc_time_analysis(clients), SMPC_TIMES)
        r = server["aggregate_s"] * MS
        out.append(f"Global Server aggregation + dequantization (ms): min {r.min():.4f}, max {r.max():.4f}, "
                   f"mean {r.mean():.4f}")
        first = clients[clients["round"] == clients["round"].min()]
        out.append("One-time pairwise key setup (ms, charged to the first round): " + ", ".join(
            f"{n} {v * MS:.2f}" for n, v in zip(first["client"], first["key_setup_s"])))
        used = server.groupby("prime_index", sort=True)
        out.append("Adaptive prime p_t: " + "; ".join(
            f"Pool[{i}] = {g['prime'].iloc[0]} in round(s) {', '.join(map(str, g['round']))}" for i, g in used))
        later = clients[clients["round"] > clients["round"].min()]
        ref = later if len(later) else clients
        out.append(f"SMPC: {len(server)} secure aggregations of {int(server['values'].iloc[0])} values; "
                   f"per client and round {int(ref['bytes_up'].iloc[0])} bytes up, "
                   f"{int(ref['bytes_down'].iloc[0])} bytes down (after the one-time key setup)")
    else:
        out += ["", "=== SMPC time analysis ===", "SMPC disabled; no timing."]
    costs = getattr(result, "costs", None)
    if costs is not None and len(costs) and "decode_s" in costs:
        out += ["", f"=== Encoding and decoding time analysis over {costs['round'].nunique()} rounds "
                f"(min, max and mean per client) ==="]
        out += _time_tables(codec_time_analysis(costs), CODEC_TIMES)
    return "\n".join(out)
