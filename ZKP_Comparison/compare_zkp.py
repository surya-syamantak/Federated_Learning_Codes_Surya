"""Compare the ZKP of three FedRF-Diff versions: times, communication and computation.

    v10  Fed_Code_v10_Schnorr_LaplaceDP_eps5               Schnorr ZKP
    v11  Fed_Code_v11_Pedersen_LaplaceDP_eps2-3             Pedersen commitment + Sigma/range proof
    v12  Fed_Code_v12_ProposedPedersenZKP_LaplaceDP_eps2-3  proposed (chained) Pedersen authentication

Inputs are each version's saved run (outputs/run/zkp_times.csv, costs.csv,
config.json; produce them with ``python -m fedrf_diff`` in each folder). The
three runs share the seed, data split, DP budgets (epsilon_t ~ U[2, 3]) and
number of rounds, so they do the same learning work and differ only in the ZKP.

Outputs (``--out``, default ZKP_Comparison/outputs):

  zkp_time_summary.csv          min, max, mean, median, std (ms) of the three ZKP
                                times per version and client (+ all clients)
  communication_summary.csv     ZKP message sizes and communication times
  computation_summary.csv       computation time per round, ZKP share
  comparison_summary.csv / .tex one row per version (journal table, booktabs)
  benchmark_zkp_times.csv       same-machine benchmark (unless --no-benchmark)
  figures/*.png (600 dpi) + *.pdf

Communication time is not measured by the simulation (all parties run in one
process); it is modelled from the bytes each version sends:

    time = round_trips * RTT + 8 * bytes_up / uplink + 8 * bytes_down / downlink

with --rtt-ms, --uplink-mbps and --downlink-mbps. The ZKP needs one round
trip in v10 and v11 (non-interactive: proof up, verdict down) and two in v12
(commitment and T up, challenge down, responses up, verdict down). The
learning messages (latents up; M_G, A_c and labels down) take LEARNING_ROUND_TRIPS.

Timing in the saved runs comes from three separate runs, so it includes some
run-to-run noise. The benchmark (on by default) therefore also times the three
ZKP implementations in one process on the same machine, interleaved round by
round, with updates of the same sizes as in the runs.

Usage:
    python compare_zkp.py                      # saved runs + benchmark (30 rounds)
    python compare_zkp.py --no-benchmark
    python compare_zkp.py --rtt-ms 20 --uplink-mbps 50 --downlink-mbps 100
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GLOBAL = "Global Server"
ALL = "All clients"
MS = 1000.0
LEARNING_ROUND_TRIPS = 2  # latents + labels up / M_G, A_c, pseudo-labels down; metrics up / ack


@dataclass(frozen=True)
class Version:
    key: str
    label: str  # legend text
    tick: str  # axis tick text
    folder: str
    zkp_round_trips: int  # message exchanges of one ZKP session
    color: str  # Okabe-Ito
    hatch: str  # so the versions do not rely on colour alone
    marker: str


VERSIONS = (
    Version("v10", "Schnorr ZKP (v10)", "Schnorr\n(v10)", "Fed_Code_v10_Schnorr_LaplaceDP_eps5", 1, "#0072B2", "", "o"),
    Version("v11", "Pedersen ZKP (v11)", "Pedersen\n(v11)", "Fed_Code_v11_Pedersen_LaplaceDP_eps2-3", 1, "#D55E00", "////", "s"),
    Version("v12", "Proposed Pedersen ZKP (v12)", "Proposed\nPedersen (v12)", "Fed_Code_v12_ProposedPedersenZKP_LaplaceDP_eps2-3",
            2, "#009E73", "....", "^"),
)
ZKP_TIMES = {  # column -> panel title
    "authentication_s": "Authentication time",
    "proof_generation_s": "Proof generation time",
    "proof_verification_s": "Proof verification time",
}

DOUBLE_COLUMN = 7.16  # inches
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#d9d9d9",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "hatch.linewidth": 0.5,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}


# --- loading ---------------------------------------------------------------
@dataclass
class Run:
    version: Version
    zkp: pd.DataFrame  # one row per round and client
    costs: pd.DataFrame  # one row per round and client + Global Server
    config: dict

    @property
    def clients(self) -> list[str]:
        return list(dict.fromkeys(self.zkp["client"]))


def load_run(root: Path, v: Version) -> Run:
    run_dir = root / v.folder / "outputs" / "run"
    missing = [f for f in ("zkp_times.csv", "costs.csv", "config.json") if not (run_dir / f).exists()]
    if missing:
        sys.exit(f"{run_dir}: missing {', '.join(missing)}; run `python -m fedrf_diff` in {v.folder} first")
    return Run(v, pd.read_csv(run_dir / "zkp_times.csv"), pd.read_csv(run_dir / "costs.csv"),
               json.loads((run_dir / "config.json").read_text()))


def check_comparable(runs: list[Run]) -> list[str]:
    """Warnings when the runs did not do the same learning work."""
    notes, ref = [], runs[0]
    for r in runs[1:]:
        for key in ("seed", "max_rounds", "n_trees", "n_landmarks", "d_z"):
            if r.config.get(key) != ref.config.get(key):
                notes.append(f"{r.version.key}: {key} = {r.config.get(key)} vs {ref.config.get(key)} in {ref.version.key}")
        if r.zkp["round"].max() != ref.zkp["round"].max():
            notes.append(f"{r.version.key} ran {r.zkp['round'].max()} rounds, {ref.version.key} {ref.zkp['round'].max()}")
        if r.config.get("privacy") != ref.config.get("privacy"):
            notes.append(f"{r.version.key}: DP settings differ from {ref.version.key}")
        if r.clients != ref.clients:
            notes.append(f"{r.version.key}: different clients")
    return notes


def short(client: str) -> str:
    """'Client 3(VA Long Beach)' -> 'VA Long Beach'."""
    return client.split("(", 1)[1].rstrip(")") if "(" in client else client


# --- tables ------------------------------------------------------------------
def zkp_time_summary(runs: list[Run], source: str = "saved run") -> pd.DataFrame:
    rows = []
    for r in runs:
        groups = [(c, r.zkp[r.zkp["client"] == c]) for c in r.clients] + [(ALL, r.zkp)]
        for client, g in groups:
            for col, name in ZKP_TIMES.items():
                ms = g[col] * MS
                rows.append({"source": source, "version": r.version.key, "zkp": r.version.label,
                             "client": client, "time": name, "n": len(ms),
                             "min_ms": ms.min(), "max_ms": ms.max(), "mean_ms": ms.mean(),
                             "median_ms": ms.median(), "std_ms": ms.std(ddof=1)})
    return pd.DataFrame(rows)


def link_time(n_up: float, n_down: float, round_trips: float, a: argparse.Namespace) -> float:
    """Seconds to exchange the bytes over the modelled link."""
    return (round_trips * a.rtt_ms / MS + 8 * n_up / (a.uplink_mbps * 1e6)
            + 8 * n_down / (a.downlink_mbps * 1e6))


def communication_table(r: Run, a: argparse.Namespace) -> pd.DataFrame:
    """Per round and client: ZKP and learning bytes and modelled communication times."""
    clients = r.costs[r.costs["client"] != GLOBAL][["round", "client", "bytes_up", "bytes_down"]]
    t = clients.merge(r.zkp[["round", "client", "bytes_up", "bytes_down"]], on=["round", "client"],
                      suffixes=("_total", "_zkp"))
    t["bytes_up_learning"] = t["bytes_up_total"] - t["bytes_up_zkp"]
    t["bytes_down_learning"] = t["bytes_down_total"] - t["bytes_down_zkp"]
    v = r.version
    t["zkp_transmission_s"] = [link_time(u, d, 0, a) for u, d in zip(t["bytes_up_zkp"], t["bytes_down_zkp"])]
    t["zkp_latency_s"] = v.zkp_round_trips * a.rtt_ms / MS
    t["zkp_comm_s"] = t["zkp_transmission_s"] + t["zkp_latency_s"]
    t["learning_comm_s"] = [link_time(u, d, LEARNING_ROUND_TRIPS, a)
                            for u, d in zip(t["bytes_up_learning"], t["bytes_down_learning"])]
    t["total_comm_s"] = t["zkp_comm_s"] + t["learning_comm_s"]
    return t.assign(version=v.key)


def communication_summary(runs: list[Run], a: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    rows, tables = [], {}
    for r in runs:
        t = tables[r.version.key] = communication_table(r, a)
        per_round = t.groupby("round")[["zkp_comm_s", "total_comm_s"]].max()  # clients in parallel
        rows.append({
            "version": r.version.key, "zkp": r.version.label, "zkp_round_trips": r.version.zkp_round_trips,
            "zkp_bytes_up_per_session": t["bytes_up_zkp"].mean(),
            "zkp_bytes_down_per_session": t["bytes_down_zkp"].mean(),
            "zkp_comm_ms_min": t["zkp_comm_s"].min() * MS, "zkp_comm_ms_max": t["zkp_comm_s"].max() * MS,
            "zkp_comm_ms_mean": t["zkp_comm_s"].mean() * MS,
            "zkp_transmission_ms_mean": t["zkp_transmission_s"].mean() * MS,
            "client_comm_s_per_round_min": t["total_comm_s"].min(),
            "client_comm_s_per_round_max": t["total_comm_s"].max(),
            "client_comm_s_per_round_mean": t["total_comm_s"].mean(),
            "round_comm_s_mean (slowest client)": per_round["total_comm_s"].mean(),
            "zkp_share_of_comm_time_%": 100 * t["zkp_comm_s"].sum() / t["total_comm_s"].sum(),
            "zkp_bytes_total_run": int(t["bytes_up_zkp"].sum() + t["bytes_down_zkp"].sum()),
            "zkp_share_of_bytes_%": 100 * (t["bytes_up_zkp"].sum() + t["bytes_down_zkp"].sum())
            / (t["bytes_up_total"].sum() + t["bytes_down_total"].sum()),
            "comm_s_total_run (all clients)": t["total_comm_s"].sum(),
        })
    return pd.DataFrame(rows), tables


def computation_table(r: Run) -> pd.DataFrame:
    """Per round: client and server computation, ZKP and learning parts (seconds)."""
    c = r.costs
    clients, server = c[c["client"] != GLOBAL], c[c["client"] == GLOBAL].set_index("round")
    t = clients.groupby("round").agg(
        client_total_s=("computation_s", "sum"), client_mean_s=("computation_s", "mean"),
        encode_dp_s=("encode_dp_s", "sum"), zkp_generation_s=("proof_generation_s", "sum"),
        evaluate_s=("evaluate_s", "sum"), decode_update_s=("decode_update_s", "sum"))
    t["server_learning_s"] = server["learning_s"]
    t["zkp_verification_s"] = server["proof_verification_s"]
    t["server_total_s"] = server["computation_s"]
    t["zkp_s"] = t["zkp_generation_s"] + t["zkp_verification_s"]
    t["total_s"] = t["client_total_s"] + t["server_total_s"]
    t["learning_s"] = t["total_s"] - t["zkp_s"]
    return t.reset_index().assign(version=r.version.key)


def computation_summary(runs: list[Run]) -> tuple[pd.DataFrame, dict]:
    rows, tables = [], {}
    for r in runs:
        t = tables[r.version.key] = computation_table(r)
        row = {"version": r.version.key, "zkp": r.version.label, "rounds": len(t)}
        for col, name in (("zkp_s", "zkp_per_round"), ("zkp_generation_s", "zkp_generation_per_round"),
                          ("zkp_verification_s", "zkp_verification_per_round"),
                          ("client_mean_s", "client_per_round"), ("server_total_s", "server_per_round"),
                          ("total_s", "total_per_round")):
            scale, unit = (MS, "ms") if col.startswith("zkp") else (1.0, "s")
            row.update({f"{name}_{unit}_min": t[col].min() * scale, f"{name}_{unit}_max": t[col].max() * scale,
                        f"{name}_{unit}_mean": t[col].mean() * scale})
        row["zkp_share_of_computation_%"] = 100 * t["zkp_s"].sum() / t["total_s"].sum()
        row["total_computation_s_run"] = t["total_s"].sum()
        rows.append(row)
    return pd.DataFrame(rows), tables


def overall_summary(times: pd.DataFrame, comm: pd.DataFrame, comp: pd.DataFrame,
                    bench: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for v in VERSIONS:
        t = times[(times["version"] == v.key) & (times["client"] == ALL)].set_index("time")
        cm = comm.set_index("version").loc[v.key]
        cp = comp.set_index("version").loc[v.key]
        row = {"ZKP": v.label}
        for name, short_name in (("Authentication time", "Auth"), ("Proof generation time", "Gen"),
                                 ("Proof verification time", "Verif")):
            for s in ("min", "max", "mean"):
                row[f"{short_name} {s} (ms)"] = t.loc[name, f"{s}_ms"]
        if bench is not None:
            b = bench[(bench["version"] == v.key) & (bench["client"] == ALL)].set_index("time")
            row["Bench auth mean (ms)"] = b.loc["Authentication time", "mean_ms"]
        row.update({
            "ZKP up (B/session)": cm["zkp_bytes_up_per_session"],
            "ZKP down (B/session)": cm["zkp_bytes_down_per_session"],
            "ZKP comm (ms/session)": cm["zkp_comm_ms_mean"],
            "Comm per client-round (s)": cm["client_comm_s_per_round_mean"],
            "ZKP share of comm (%)": cm["zkp_share_of_comm_time_%"],
            "ZKP comp per round (ms)": cp["zkp_per_round_ms_mean"],
            "Total comp per round (s)": cp["total_per_round_s_mean"],
            "ZKP share of comp (%)": cp["zkp_share_of_computation_%"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def to_latex(table: pd.DataFrame, path: Path, caption: str) -> None:
    """Booktabs LaTeX table (no jinja2 needed): versions as columns, quantities as rows."""
    t = table.set_index("ZKP").T
    cols = list(t.columns)
    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", r"\label{tab:zkp_comparison}",
             r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             "Quantity & " + " & ".join(cols) + r" \\", r"\midrule"]
    for name, values in t.iterrows():
        cells = []
        for v in values:
            v = float(v)
            cells.append(f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}" if abs(v) < 0.1 else f"{v:.2f}")
        lines.append(name.replace("%", r"\%") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --- same-machine benchmark -------------------------------------------------------
def _load_zkp_module(root: Path, v: Version):
    """Import <version>/fedrf_diff/zkp.py under a unique name (the packages share a name)."""
    path = root / v.folder / "fedrf_diff" / "zkp.py"
    spec = importlib.util.spec_from_file_location(f"zkp_{v.key}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def benchmark(root: Path, runs: list[Run], rounds: int, seed: int = 42) -> pd.DataFrame:
    """Time the three ZKP implementations in one process, interleaved round by round.

    Each client's update has the same shape as in the saved runs (latent
    matrix of ``vectors`` x d_z float64 rows and n_train int labels).
    """
    modules = {r.version.key: _load_zkp_module(root, r.version) for r in runs}
    ref = runs[0]
    d_z, n_landmarks = ref.config.get("d_z", 4), ref.config.get("n_landmarks", 100)
    vectors = ref.zkp.drop_duplicates("client").set_index("client")["vectors"]
    rng = np.random.default_rng(seed)
    params = {k: m.setup() for k, m in modules.items()}
    state = {}
    if "v12" in modules:
        m = modules["v12"]
        state["provers"] = {c: m.ClientProver(params["v12"], c) for c in vectors.index}
        state["verifier"] = m.ServerVerifier(params["v12"])

    def session(key: str, client: str, t: int, update: tuple, n_vec: int):
        m, pp = modules[key], params[key]
        if key == "v12":
            return m.authenticate(pp, state["provers"][client], state["verifier"], t, vectors=n_vec)
        return m.authenticate(pp, client, t, update)

    rows = []
    order = [r.version.key for r in runs]
    for t in range(0, rounds + 1):  # round 0 = warm-up, not recorded
        updates = {c: (rng.normal(size=(int(n), d_z)), rng.integers(0, 2, size=int(n) - n_landmarks))
                   for c, n in vectors.items()}
        for key in order[t % len(order):] + order[: t % len(order)]:  # rotate the order every round
            for client, update in updates.items():
                s = session(key, client, max(t, 1) if key != "v12" else t + 1, update, len(update[0]))
                if t > 0:
                    rows.append({"version": key, "round": t, "client": client, "authenticated": s.authenticated,
                                 "authentication_s": s.authentication_s,
                                 "proof_generation_s": s.proof_generation_s,
                                 "proof_verification_s": s.proof_verification_s,
                                 "bytes_up": s.bytes_up, "bytes_down": s.bytes_down})
        if t:
            print(f"  benchmark round {t}/{rounds}", end="\r", flush=True)
    print()
    return pd.DataFrame(rows)


# --- figures ----------------------------------------------------------------------
def _panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold")


def _save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {path.with_suffix('.png')} (+ .pdf)")


def _version_legend(fig, ncol: int = 3, y: float = 1.0) -> None:
    handles = [Patch(facecolor=v.color, edgecolor="black", hatch=v.hatch, linewidth=0.4, label=v.label)
               for v in VERSIONS]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y))


def _bars(ax, groups: list[str], per_version: dict[str, dict[str, tuple[float, float, float]]],
          width: float = 0.8) -> None:
    """Grouped bars: height = mean, whiskers = min..max."""
    x = np.arange(len(groups))
    w = width / len(VERSIONS)
    for j, v in enumerate(VERSIONS):
        stats = per_version[v.key]
        mean = np.array([stats[g][0] for g in groups])
        lo = mean - np.array([stats[g][1] for g in groups])
        hi = np.array([stats[g][2] for g in groups]) - mean
        ax.bar(x + (j - (len(VERSIONS) - 1) / 2) * w, mean, w, color=v.color, hatch=v.hatch,
               edgecolor="black", linewidth=0.4, yerr=[lo, hi], capsize=1.8,
               error_kw={"elinewidth": 0.6, "capthick": 0.6, "ecolor": "#333333"}, zorder=2)
    ax.set_xticks(x, groups)


def fig_zkp_times(times: pd.DataFrame, path: Path, source: str) -> None:
    clients = list(dict.fromkeys(times.loc[times["client"] != ALL, "client"]))
    groups = [short(c) for c in clients] + [ALL]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.55), sharey=True)
        for ax, letter, name in zip(axes, "abc", ZKP_TIMES.values()):
            sub = times[times["time"] == name]
            per_version = {v.key: {(short(c) if c != ALL else ALL): (row.mean_ms, row.min_ms, row.max_ms)
                                   for c, row in sub[sub["version"] == v.key].set_index("client").iterrows()}
                           for v in VERSIONS}
            _bars(ax, groups, per_version)
            ax.set_yscale("log")
            ax.tick_params(axis="x", rotation=35)
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")
                lbl.set_rotation_mode("anchor")
            _panel(ax, letter, name)
        axes[0].set_ylabel("Time per client per round (ms, log scale)")
        _version_legend(fig, y=0.99)
        fig.text(0.5, -0.04, f"Bars: mean; whiskers: min to max over all rounds ({source}).",
                 ha="center", fontsize=7.5, style="italic")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        _save(fig, path)


def fig_zkp_rounds(runs: list[Run], path: Path) -> None:
    """Per-round ZKP times (mean over clients, band = min..max over clients)."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.3), sharey=True)
        for ax, letter, (col, name) in zip(axes, "abc", ZKP_TIMES.items()):
            for r in runs:
                v = r.version
                g = r.zkp.groupby("round")[col].agg(["mean", "min", "max"]) * MS
                ax.fill_between(g.index, g["min"], g["max"], color=v.color, alpha=0.18, linewidth=0)
                ax.plot(g.index, g["mean"], color=v.color, marker=v.marker, markevery=3, label=v.label)
            ax.set_yscale("log")
            ax.set_xlabel("Round")
            _panel(ax, letter, name)
        axes[0].set_ylabel("Time (ms, log scale)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.99))
        fig.text(0.5, -0.05, "Line: mean over the 4 clients; band: min to max over the clients.",
                 ha="center", fontsize=7.5, style="italic")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        _save(fig, path)


def fig_communication(comm: pd.DataFrame, comm_tables: dict, path: Path, a: argparse.Namespace) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.5))
        c = comm.set_index("version")
        x = np.arange(len(VERSIONS))
        names = [v.tick for v in VERSIONS]

        # (a) ZKP message size per session, up and down.
        ax = axes[0]
        w = 0.38
        for j, (col, label, fill) in enumerate((("zkp_bytes_up_per_session", "Uplink", True),
                                                ("zkp_bytes_down_per_session", "Downlink", False))):
            vals = np.array([c.loc[v.key, col] for v in VERSIONS])
            bars = ax.bar(x + (j - 0.5) * w, vals, w, zorder=2, linewidth=0.6, label=label,
                          color=[v.color if fill else "white" for v in VERSIONS],
                          edgecolor=[v.color if not fill else "black" for v in VERSIONS],
                          hatch=None if fill else "xxxx")
            for b, val in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, val * 1.25,
                        f"{val / 1000:.1f}k" if val >= 1000 else f"{val:.0f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 2e7)
        ax.set_xticks(x, names)
        ax.set_ylabel("Bytes per session (log scale)")
        ax.legend(loc="upper center", ncol=2, handlelength=1.2, columnspacing=0.8,
                  handles=[Patch(facecolor="#888888", edgecolor="black", label="Uplink"),
                           Patch(facecolor="white", edgecolor="#555555", hatch="xxxx", label="Downlink")])
        _panel(ax, "a", "ZKP message size")

        # (b) ZKP communication time per session: latency + transmission.
        ax = axes[1]
        lat = np.array([v.zkp_round_trips * a.rtt_ms for v in VERSIONS])
        trans = np.array([c.loc[v.key, "zkp_transmission_ms_mean"] for v in VERSIONS])
        ax.bar(x, lat, 0.6, color=[v.color for v in VERSIONS], edgecolor="black", linewidth=0.4,
               alpha=0.45, zorder=2)
        ax.bar(x, trans, 0.6, bottom=lat, color=[v.color for v in VERSIONS], hatch="////",
               edgecolor="black", linewidth=0.4, zorder=2)
        for xi, l_, t_ in zip(x, lat, trans):
            ax.text(xi, l_ + t_ + 2, f"{l_ + t_:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, names)
        ax.set_ylim(0, (lat + trans).max() * 1.45)
        ax.set_ylabel("Time per session (ms)")
        ax.legend(handles=[Patch(facecolor="#bbbbbb", edgecolor="black", label="Latency (round trips × RTT)"),
                           Patch(facecolor="white", edgecolor="black", hatch="////", label="Transmission")],
                  loc="upper left", handlelength=1.2, fontsize=6.8)
        _panel(ax, "b", "ZKP communication time")

        # (c) Total communication time per client per round, ZKP part vs learning part.
        ax = axes[2]
        for i, v in enumerate(VERSIONS):
            t = comm_tables[v.key]
            learn, zkp_ = t["learning_comm_s"].mean() * MS, t["zkp_comm_s"].mean() * MS
            tot = t["total_comm_s"] * MS
            ax.bar(i, learn, 0.6, color="#cccccc", edgecolor="black", linewidth=0.4, zorder=2)
            ax.bar(i, zkp_, 0.6, bottom=learn, color=v.color, hatch=v.hatch, edgecolor="black",
                   linewidth=0.4, zorder=2, yerr=[[learn + zkp_ - tot.min()], [tot.max() - learn - zkp_]],
                   capsize=2, error_kw={"elinewidth": 0.6, "capthick": 0.6})
            ax.text(i, tot.max() + 8, f"{learn + zkp_:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, names)
        ax.set_ylim(0, max(comm_tables[v.key]["total_comm_s"].max() for v in VERSIONS) * MS * 1.45)
        ax.set_ylabel("Time per client per round (ms)")
        ax.legend(handles=[Patch(facecolor="#cccccc", edgecolor="black", label="Learning messages"),
                           Patch(facecolor="white", edgecolor="black", label="ZKP (version colour)")],
                  loc="upper left", ncol=2, handlelength=1.2, fontsize=6.6, columnspacing=0.8)
        _panel(ax, "c", "Total communication time")
        fig.text(0.5, -0.03, f"Link model: RTT {a.rtt_ms:g} ms, uplink {a.uplink_mbps:g} Mbit/s, "
                 f"downlink {a.downlink_mbps:g} Mbit/s; whiskers in (c): min to max over clients and rounds.",
                 ha="center", fontsize=7.5, style="italic")
        fig.tight_layout()
        _save(fig, path)


# Computation parts per round, partitioned into learning, ZKP and totals:
# (partition, column in computation_table, tick label).
COMPUTATION_PARTS = (
    ("Learning computation", "encode_dp_s", "Encoding\n+ DP"),
    ("Learning computation", "evaluate_s", "Validation"),
    ("Learning computation", "decode_update_s", "Decoding +\nRF retraining"),
    ("Learning computation", "server_learning_s", "Server\nlearning"),
    ("ZKP computation", "zkp_generation_s", "Proof\ngeneration"),
    ("ZKP computation", "zkp_verification_s", "Proof\nverification"),
    ("Total per round", "learning_s", "Learning\ntotal"),
    ("Total per round", "zkp_s", "ZKP\ntotal"),
    ("Total per round", "total_s", "All\ncomputation"),
)
PARTITION_SHADE = {"Learning computation": "#f2f2f2", "ZKP computation": "#fff4e0", "Total per round": "#e8f1f8"}


def _ms_label(v: float) -> str:
    return f"{v:,.0f}" if v >= 10 else f"{v:.2f}"


def fig_computation(comp_tables: dict, path: Path) -> None:
    """(a) ZKP computation per round, (b) ZKP share, (c) every computation part, partitioned.

    In (c) each part has its own group of three version bars (mean over the
    rounds, whiskers min..max) on one log axis, so parts that differ by four
    orders of magnitude (ms of ZKP vs s of learning) are all visible; shaded
    blocks separate the learning parts, the ZKP parts and the totals.
    """
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(DOUBLE_COLUMN, 5.0))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], width_ratios=[1.45, 1.0],
                                hspace=0.5, wspace=0.28)
        ax_a, ax_b, ax_c = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
        x3 = np.arange(len(VERSIONS))

        # (a) ZKP computation per round (all clients' proof generation + server verification).
        for v in VERSIONS:
            t = comp_tables[v.key]
            ax_a.plot(t["round"], t["zkp_s"] * MS, color=v.color, marker=v.marker, markevery=3, label=v.label)
        ax_a.set_yscale("log")
        ax_a.set_xlabel("Round")
        ax_a.set_ylabel("ZKP computation\nper round (ms, log)")
        _panel(ax_a, "a", "ZKP computation per round")

        # (b) ZKP share of the total computation.
        share = [100 * comp_tables[v.key]["zkp_s"].sum() / comp_tables[v.key]["total_s"].sum() for v in VERSIONS]
        for i, (h, v) in enumerate(zip(share, VERSIONS)):
            ax_b.bar(i, h, 0.6, color=v.color, hatch=v.hatch, edgecolor="black", linewidth=0.4, zorder=2)
            ax_b.text(i, h * 1.25, f"{h:.2g}%", ha="center", va="bottom", fontsize=7)
        ax_b.set_yscale("log")
        ax_b.set_ylim(min(share) / 3, max(share) * 8)
        ax_b.set_xticks(x3, [v.tick for v in VERSIONS])
        ax_b.set_ylabel("ZKP share of all\ncomputation (%, log)")
        _panel(ax_b, "b", "ZKP overhead")

        # (c) Every computation part per round, grouped and partitioned.
        ax = ax_c
        n_parts, w = len(COMPUTATION_PARTS), 0.8 / len(VERSIONS)
        xs = np.arange(n_parts, dtype=float)
        # A gap between the partitions.
        partitions = list(dict.fromkeys(p for p, _, _ in COMPUTATION_PARTS))
        for k, (part, _, _) in enumerate(COMPUTATION_PARTS):
            xs[k] += 0.6 * partitions.index(part)
        lo_all, hi_all = [], []
        for j, v in enumerate(VERSIONS):
            t = comp_tables[v.key]
            mean = np.array([t[col].mean() for _, col, _ in COMPUTATION_PARTS]) * MS
            lo = np.array([t[col].min() for _, col, _ in COMPUTATION_PARTS]) * MS
            hi = np.array([t[col].max() for _, col, _ in COMPUTATION_PARTS]) * MS
            lo_all.append(lo)
            hi_all.append(hi)
            pos = xs + (j - (len(VERSIONS) - 1) / 2) * w
            ax.bar(pos, mean, w, color=v.color, hatch=v.hatch, edgecolor="black", linewidth=0.4, zorder=2,
                   label=v.label, yerr=[mean - lo, hi - mean], capsize=1.5,
                   error_kw={"elinewidth": 0.6, "capthick": 0.6, "ecolor": "#333333"})
            for p, m, h in zip(pos, mean, hi):
                ax.text(p, h * 1.18, _ms_label(m), ha="center", va="bottom", fontsize=5.6, rotation=90)
        ymin, ymax = min(map(np.min, lo_all)) / 2.5, max(map(np.max, hi_all)) * 40
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        # Shaded block and title per partition.
        for part in partitions:
            idx = [k for k, (p, _, _) in enumerate(COMPUTATION_PARTS) if p == part]
            left, right = xs[idx[0]] - 0.5, xs[idx[-1]] + 0.5
            ax.axvspan(left, right, color=PARTITION_SHADE[part], zorder=0, linewidth=0)
            ax.text((left + right) / 2, ymax / 1.6, part, ha="center", va="top", fontsize=8, fontweight="bold")
        for part in partitions[1:]:
            k = next(k for k, (p, _, _) in enumerate(COMPUTATION_PARTS) if p == part)
            ax.axvline(xs[k] - 0.8, color="#555555", linewidth=0.8, linestyle="-", zorder=1)
        ax.set_xlim(xs[0] - 0.55, xs[-1] + 0.55)
        ax.set_xticks(xs, [label for _, _, label in COMPUTATION_PARTS], fontsize=7.5)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Computation per round (ms, log)")
        _panel(ax, "c", "Computation breakdown per round (all clients + server)")

        handles, labels = ax_a.get_legend_handles_labels()
        fig.legend([Patch(facecolor=v.color, edgecolor="black", hatch=v.hatch, linewidth=0.4) for v in VERSIONS],
                   [v.label for v in VERSIONS], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.965))
        fig.text(0.5, 0.0, "(c) Bars: mean per round; whiskers: min to max over the rounds; numbers: mean (ms). "
                 "The learning code is identical in the three versions,\nso differences in the learning parts "
                 "come from the machine load during each run, not from the ZKP.",
                 ha="center", va="top", fontsize=7.2, style="italic")
        _save(fig, path)


def fig_benchmark(bench: pd.DataFrame, path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.3), sharey=True)
        names = [v.tick for v in VERSIONS]
        for ax, letter, (col, name) in zip(axes, "abc", ZKP_TIMES.items()):
            data = [bench.loc[bench["version"] == v.key, col].to_numpy() * MS for v in VERSIONS]
            box = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                             medianprops={"color": "black", "linewidth": 1.0},
                             flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.6},
                             whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7})
            for patch, v in zip(box["boxes"], VERSIONS):
                patch.set(facecolor=v.color, alpha=0.75, hatch=v.hatch, edgecolor="black", linewidth=0.5)
            for i, d in enumerate(data, start=1):
                ax.scatter([i], [d.mean()], marker="D", s=10, color="white", edgecolor="black", zorder=3,
                           linewidth=0.6)
            ax.set_yscale("log")
            ax.set_xticks(range(1, len(VERSIONS) + 1), names)
            _panel(ax, letter, name)
        axes[0].set_ylabel("Time per session (ms, log scale)")
        n = bench.groupby("version").size().iloc[0]
        fig.text(0.5, -0.07, f"Same-machine benchmark, {n} sessions per ZKP (interleaved); "
                 "box: quartiles, whiskers: 1.5 IQR, diamond: mean.", ha="center", fontsize=7.5, style="italic")
        fig.tight_layout()
        _save(fig, path)


# --- main ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="folder holding the three version folders")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--rtt-ms", type=float, default=50.0, help="network round-trip time (ms)")
    ap.add_argument("--uplink-mbps", type=float, default=20.0, help="client -> server bandwidth (Mbit/s)")
    ap.add_argument("--downlink-mbps", type=float, default=50.0, help="server -> client bandwidth (Mbit/s)")
    ap.add_argument("--benchmark-rounds", type=int, default=30)
    ap.add_argument("--no-benchmark", action="store_true", help="only compare the saved runs")
    a = ap.parse_args(argv)
    figs = a.out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    runs = [load_run(a.root, v) for v in VERSIONS]
    print("Versions:")
    for r in runs:
        z = r.config.get("zkp", {})
        print(f"  {r.version.key}: {r.version.folder} | protocol {z.get('protocol')} | group {z.get('group')} | "
              f"{r.zkp['round'].max()} rounds x {len(r.clients)} clients")
    notes = check_comparable(runs)
    print("Comparable runs (same seed, rounds, learning settings and DP)." if not notes
          else "WARNING, the runs differ:\n  " + "\n  ".join(notes))

    times = zkp_time_summary(runs)
    comm, comm_tables = communication_summary(runs, a)
    comp, comp_tables = computation_summary(runs)
    times.to_csv(a.out / "zkp_time_summary.csv", index=False)
    comm.to_csv(a.out / "communication_summary.csv", index=False)
    comp.to_csv(a.out / "computation_summary.csv", index=False)
    pd.concat(comm_tables.values()).to_csv(a.out / "communication_per_round.csv", index=False)
    pd.concat(comp_tables.values()).to_csv(a.out / "computation_per_round.csv", index=False)

    bench_times = None
    if not a.no_benchmark:
        print(f"\nSame-machine benchmark: {a.benchmark_rounds} rounds x {len(runs[0].clients)} clients per ZKP ...")
        t0 = time.perf_counter()
        bench = benchmark(a.root, runs, a.benchmark_rounds)
        print(f"  done in {time.perf_counter() - t0:.1f} s; all sessions valid: {bool(bench['authenticated'].all())}")
        bench.to_csv(a.out / "benchmark_zkp_times.csv", index=False)
        bench_runs = [Run(r.version, bench[bench["version"] == r.version.key], r.costs, r.config) for r in runs]
        bench_times = zkp_time_summary(bench_runs, source="benchmark")
        bench_times.to_csv(a.out / "benchmark_zkp_time_summary.csv", index=False)

    summary = overall_summary(times, comm, comp, bench_times)
    summary.to_csv(a.out / "comparison_summary.csv", index=False)
    to_latex(summary, a.out / "comparison_summary.tex",
             "ZKP comparison of the three FedRF-Diff versions (seed 42, 30 rounds, 4 clients, "
             r"$\varepsilon_t \sim U[2,3]$). Times in ms per client and round.")

    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    print("\n=== ZKP times (ms), all clients over all rounds (saved runs) ===")
    shown = times[times["client"] == ALL].pivot(index="zkp", columns="time", values=["min_ms", "max_ms", "mean_ms"])
    shown.columns = [f"{t} {s.replace('_ms', '')}" for s, t in shown.columns]
    print(shown[[f"{t} {s}" for t in ZKP_TIMES.values() for s in ("min", "max", "mean")]]
          .reindex([v.label for v in VERSIONS]).T.to_string(float_format=fmt))
    print("\n=== ZKP times (ms) per client: min / max / mean ===")
    for name in ZKP_TIMES.values():
        sub = times[(times["time"] == name)]
        tab = sub.pivot(index="client", columns="zkp", values="mean_ms")
        lo = sub.pivot(index="client", columns="zkp", values="min_ms")
        hi = sub.pivot(index="client", columns="zkp", values="max_ms")
        cells = pd.DataFrame({v.label: [f"{lo.loc[c, v.label]:.3f} / {hi.loc[c, v.label]:.3f} / "
                                        f"{tab.loc[c, v.label]:.3f}" for c in tab.index] for v in VERSIONS},
                             index=tab.index).reindex(runs[0].clients + [ALL])
        print(f"{name}:")
        print(cells.to_string())
    if bench_times is not None:
        print("\n=== Same-machine benchmark: ZKP times (ms), all clients ===")
        b = bench_times[bench_times["client"] == ALL].pivot(index="time", columns="zkp",
                                                            values=["min_ms", "max_ms", "mean_ms"])
        print(b.reindex(list(ZKP_TIMES.values())).to_string(float_format=fmt))
    print(f"\n=== Communication (RTT {a.rtt_ms:g} ms, up {a.uplink_mbps:g} Mbit/s, down {a.downlink_mbps:g} Mbit/s) ===")
    print(comm.drop(columns="zkp").set_index("version").T.to_string(float_format=fmt))
    print("\n=== Computation ===")
    print(comp.drop(columns="zkp").set_index("version").T.to_string(float_format=fmt))

    print("\nFigures (600-dpi PNG + vector PDF):")
    fig_zkp_times(times, figs / "fig1_zkp_times_comparison", "saved runs")
    fig_zkp_rounds(runs, figs / "fig2_zkp_times_per_round")
    fig_communication(comm, comm_tables, figs / "fig3_communication_comparison", a)
    fig_computation(comp_tables, figs / "fig4_computation_comparison")
    if bench_times is not None:
        fig_benchmark(bench, figs / "fig5_benchmark_zkp_times")
        fig_zkp_times(bench_times, figs / "fig6_benchmark_zkp_times_per_client", "same-machine benchmark")
    print(f"\nTables in {a.out.resolve()}")


if __name__ == "__main__":
    main()
