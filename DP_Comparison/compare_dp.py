"""Compare FedRF-Diff with Laplace DP (v10) and without DP (v13): communication and computation.

    v10  Fed_Code_v10_Schnorr_LaplaceDP_eps5   Schnorr ZKP + Laplace DP (epsilon_r ~ U[2, 3] per round)
    v13  Fed_Code_v13_Schnorr_NoDP             Schnorr ZKP, no differential privacy

Both versions use the same seed, data split, learning settings and Schnorr ZKP;
they differ only in the DP. Inputs are each version's saved run
(outputs/run/costs.csv, history.csv, zkp_times.csv, config.json; produce them
with ``python -m fedrf_diff`` in each folder).

Compared:

* Communication: bytes per client and round, split into their parts
  (latent matrix, labels, metrics, ZKP up; M_glob, R_k, pseudo-labels, ZKP down),
  and the modelled communication time
      time = round trips x RTT + 8 x bytes_up / uplink + 8 x bytes_down / downlink
  (--rtt-ms, --uplink-mbps, --downlink-mbps). One round needs 3 round trips in
  both versions: latents + labels + ZKP proof up / verdict down, M_glob, R_k and
  labels down, metrics up. Reported per round and summed until each run
  stopped.
* Computation: per round, every client's encoding (+ DP in v10), ZKP proof
  generation, validation and decoding + retraining, and the server's learning
  and ZKP verification; per-round statistics and the total until each run
  stopped.
* DP overhead (benchmark): the Laplace mechanism of v10 (L1 clipping + noise
  on the patient rows, clipping of the landmark rows) is timed on this machine
  on latent matrices of the clients' sizes, so its cost is measured directly
  instead of being read off two separate runs.
* End to end: rounds until the run stopped, total computation and
  communication time, and the Global Server's accuracy for context.
* Averaged metrics: accuracy, precision, recall, specificity, F1 score, MCC
  and Cohen's kappa averaged over each run's rounds (mean, min, max, std),
  per client and for the Global Server, and the difference No DP - DP.

Outputs (``--out``, default DP_Comparison/outputs): CSV tables, a LaTeX table
(booktabs) and four figures (600-dpi PNG + vector PDF).

Usage:
    python compare_dp.py
    python compare_dp.py --rtt-ms 20 --uplink-mbps 50 --downlink-mbps 100
    python compare_dp.py --no-benchmark
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
MS = 1000.0
FLOAT_BYTES, LABEL_BYTES, METRIC_BYTES = 8, 1, 8 + 4 * 4  # as in fedrf_diff/pipeline.py
ROUND_TRIPS = 3  # (latents, labels, ZKP proof) / verdict; M_glob, R_k, labels; metrics


@dataclass(frozen=True)
class Version:
    key: str
    label: str
    tick: str
    folder: str
    color: str  # Okabe-Ito
    hatch: str
    marker: str


VERSIONS = (
    Version("v10", r"Laplace DP, $\varepsilon_r \sim U[2, 3]$ (v10)", "Laplace DP\n(v10)", "Fed_Code_v10_Schnorr_LaplaceDP_eps5",
            "#D55E00", "////", "s"),
    Version("v13", "No DP (v13)", "No DP\n(v13)", "Fed_Code_v13_Schnorr_NoDP", "#0072B2", "", "o"),
)

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


def plain(label: str) -> str:
    """Legend label without mathtext, for tables and the terminal."""
    return label.replace(r"$\varepsilon_r \sim U[2, 3]$", "eps_r ~ U[2, 3]")


# --- loading ---------------------------------------------------------------
@dataclass
class Run:
    version: Version
    costs: pd.DataFrame
    history: pd.DataFrame
    zkp: pd.DataFrame
    config: dict

    @property
    def rounds(self) -> int:
        return int(self.history["round"].max())

    @property
    def converged(self) -> bool:
        return bool(self.history["patience"].iloc[-1] >= self.config["patience"])

    @property
    def clients(self) -> list[str]:
        return [c for c in dict.fromkeys(self.costs["client"]) if c != GLOBAL]


def load_run(root: Path, v: Version) -> Run:
    d = root / v.folder / "outputs" / "run"
    missing = [f for f in ("costs.csv", "history.csv", "zkp_times.csv", "config.json") if not (d / f).exists()]
    if missing:
        sys.exit(f"{d}: missing {', '.join(missing)}; run `python -m fedrf_diff` in {v.folder} first")
    costs = pd.read_csv(d / "costs.csv").rename(columns={"encode_dp_s": "encode_s"})
    return Run(v, costs, pd.read_csv(d / "history.csv"), pd.read_csv(d / "zkp_times.csv"),
               json.loads((d / "config.json").read_text()))


def check_comparable(runs: list[Run]) -> list[str]:
    a, b = runs
    notes = [f"{key}: {a.config.get(key)} (v10) vs {b.config.get(key)} (v13)"
             for key in ("seed", "n_trees", "n_landmarks", "d_z", "global_n_trees", "eps_z", "eps_l", "patience")
             if a.config.get(key) != b.config.get(key)]
    if a.config.get("zkp", {}).get("group") != b.config.get("zkp", {}).get("group"):
        notes.append("different ZKP groups")
    if a.clients != b.clients:
        notes.append("different clients")
    return notes


# --- communication -------------------------------------------------------------
def communication_table(r: Run, a: argparse.Namespace) -> pd.DataFrame:
    """Per round and client: bytes by part and the modelled communication time."""
    d_z, n_land = r.config["d_z"], r.config["n_landmarks"]
    c = r.costs[r.costs["client"] != GLOBAL][["round", "client", "bytes_up", "bytes_down"]]
    t = c.merge(r.zkp[["round", "client", "vectors", "bytes_up", "bytes_down"]], on=["round", "client"],
                suffixes=("", "_zkp"))
    n_train = t["vectors"] - n_land
    t["up_latents"] = t["vectors"] * d_z * FLOAT_BYTES
    t["up_labels"] = n_train * LABEL_BYTES
    t["up_metrics"] = METRIC_BYTES
    t["up_zkp"] = t["bytes_up_zkp"]
    t["down_zkp"] = t["bytes_down_zkp"]
    t["down_alignment"] = d_z * d_z * FLOAT_BYTES
    t["down_labels"] = n_train * LABEL_BYTES
    t["down_model"] = t["bytes_down"] - t["down_zkp"] - t["down_alignment"] - t["down_labels"]
    check = t["up_latents"] + t["up_labels"] + t["up_metrics"] + t["up_zkp"]
    if not np.allclose(check, t["bytes_up"]):
        print(f"  note: {r.version.key} uplink parts do not add up exactly (rejected updates?)")
    t["transmission_s"] = 8 * t["bytes_up"] / (a.uplink_mbps * 1e6) + 8 * t["bytes_down"] / (a.downlink_mbps * 1e6)
    t["latency_s"] = ROUND_TRIPS * a.rtt_ms / MS
    t["comm_s"] = t["transmission_s"] + t["latency_s"]
    return t.drop(columns=["bytes_up_zkp", "bytes_down_zkp"]).assign(version=r.version.key)


def communication_summary(runs: list[Run], tables: dict) -> pd.DataFrame:
    rows = []
    for r in runs:
        t = tables[r.version.key]
        per_round = t.groupby("round").agg(comm_all_clients_s=("comm_s", "sum"), comm_slowest_s=("comm_s", "max"),
                                           bytes=("bytes_up", "sum"))
        per_round["bytes"] += t.groupby("round")["bytes_down"].sum()
        rows.append({
            "version": r.version.key, "dp": plain(r.version.label), "rounds": r.rounds,
            "bytes_up_per_client_round_mean": t["bytes_up"].mean(),
            "bytes_down_per_client_round_mean": t["bytes_down"].mean(),
            "comm_ms_per_client_round_min": t["comm_s"].min() * MS,
            "comm_ms_per_client_round_max": t["comm_s"].max() * MS,
            "comm_ms_per_client_round_mean": t["comm_s"].mean() * MS,
            "comm_s_per_round_slowest_client_mean": per_round["comm_slowest_s"].mean(),
            "comm_s_per_round_all_clients_mean": per_round["comm_all_clients_s"].mean(),
            "comm_s_total_until_stop (clients in parallel)": per_round["comm_slowest_s"].sum(),
            "comm_s_total_until_stop (all clients summed)": t["comm_s"].sum(),
            "MB_total_until_stop": per_round["bytes"].sum() / 1e6,
        })
    return pd.DataFrame(rows)


# --- computation ---------------------------------------------------------------
PARTS = (  # (column, label, side)
    ("encode_s", "Encoding (+ DP in v10)", "client"),
    ("proof_generation_s", "ZKP proof generation", "client"),
    ("evaluate_s", "Validation", "client"),
    ("decode_update_s", "Decoding + RF retraining", "client"),
    ("learning_s", "Server learning", "server"),
    ("proof_verification_s", "ZKP proof verification", "server"),
)


def computation_table(r: Run) -> pd.DataFrame:
    """Per round: every computation part summed over the clients (or the server), seconds."""
    c = r.costs
    clients, server = c[c["client"] != GLOBAL], c[c["client"] == GLOBAL].set_index("round")
    t = clients.groupby("round")[[p for p, _, side in PARTS if side == "client"]].sum()
    t["client_mean_s"] = clients.groupby("round")["computation_s"].mean()
    t["client_max_s"] = clients.groupby("round")["computation_s"].max()
    t["learning_s"] = server["learning_s"]
    t["proof_verification_s"] = server["proof_verification_s"]
    t["server_s"] = server["computation_s"]
    t["total_s"] = clients.groupby("round")["computation_s"].sum() + t["server_s"]
    return t.reset_index().assign(version=r.version.key)


def computation_summary(runs: list[Run], tables: dict) -> pd.DataFrame:
    rows = []
    for r in runs:
        t = tables[r.version.key]
        row = {"version": r.version.key, "dp": plain(r.version.label), "rounds": r.rounds}
        for col, label, _ in PARTS:
            row[f"{label} per round (ms, mean)"] = t[col].mean() * MS
        for col, name in (("client_mean_s", "per client per round"), ("server_s", "server per round"),
                          ("total_s", "total per round")):
            row.update({f"{name} (s, min)": t[col].min(), f"{name} (s, max)": t[col].max(),
                        f"{name} (s, mean)": t[col].mean()})
        row["total computation until stop (s)"] = t["total_s"].sum()
        rows.append(row)
    return pd.DataFrame(rows)


# --- DP overhead benchmark -----------------------------------------------------------
def _load_privacy(root: Path):
    path = root / VERSIONS[0].folder / "fedrf_diff" / "privacy.py"
    spec = importlib.util.spec_from_file_location("privacy_v10", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dp_benchmark(root: Path, run: Run, rounds: int, repeats: int = 50, seed: int = 42) -> pd.DataFrame:
    """Time v10's DP step (as in Client.encode) on latent matrices of the clients' sizes."""
    privacy = _load_privacy(root)
    d_z, n_land = run.config["d_z"], run.config["n_landmarks"]
    vectors = run.zkp.drop_duplicates("client").set_index("client")["vectors"]
    eps = privacy.round_epsilons(rounds, run.config.get("seed", seed))
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(rounds):
        for client, n_vec in vectors.items():
            mech = privacy.LaplaceMechanism(privacy.RANDOM)
            mech.set_epsilon(float(eps[t]))
            Z = rng.normal(scale=0.3, size=(int(n_vec), d_z))
            n = int(n_vec) - n_land
            t0 = time.perf_counter()
            for _ in range(repeats):
                np.vstack([mech.privatize(Z[:n], rng), privacy.clip_rows(Z[n:], mech.clip_norm)])
            rows.append({"round": t + 1, "client": client, "vectors": int(n_vec), "epsilon": float(eps[t]),
                         "dp_s": (time.perf_counter() - t0) / repeats})
    return pd.DataFrame(rows)


# --- figures -------------------------------------------------------------------------
def _panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold")


def _save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {path.with_suffix('.png')} (+ .pdf)")


def _version_legend(fig, y: float = 0.99, lines: bool = False) -> None:
    if lines:
        handles = [plt.Line2D([], [], color=v.color, marker=v.marker, label=v.label) for v in VERSIONS]
    else:
        handles = [Patch(facecolor=v.color, edgecolor="black", hatch=v.hatch, linewidth=0.4, label=v.label)
                   for v in VERSIONS]
    fig.legend(handles=handles, loc="lower center", ncol=len(VERSIONS), bbox_to_anchor=(0.5, y))


MESSAGE_PARTS = (  # (partition, column, tick label)
    ("Uplink (client → server)", "up_latents", "Latent\nmatrix"),
    ("Uplink (client → server)", "up_labels", "Training\nlabels"),
    ("Uplink (client → server)", "up_metrics", "Loss +\nTP/FP/TN/FN"),
    ("Uplink (client → server)", "up_zkp", "ZKP\nproof"),
    ("Uplink (client → server)", "bytes_up", "Uplink\ntotal"),
    ("Downlink (server → client)", "down_model", "Global\nmodel M_glob"),
    ("Downlink (server → client)", "down_alignment", "Alignment\nR_k"),
    ("Downlink (server → client)", "down_labels", "Pseudo-\nlabels"),
    ("Downlink (server → client)", "down_zkp", "ZKP\nverdict"),
    ("Downlink (server → client)", "bytes_down", "Downlink\ntotal"),
)
LINK_SHADE = {"Uplink (client → server)": "#f2f2f2", "Downlink (server → client)": "#e8f1f8"}


def _bytes_label(v: float) -> str:
    return f"{v / 1e3:.1f}k" if v >= 1000 else f"{v:.0f}"


def _cumulative(ax, runs: list[Run], series: dict, unit: str, fmt: str) -> None:
    """Cumulative curves with the final value and the stop reason at each end point."""
    top = max(float(s.iloc[-1]) for s in series.values())
    for r in runs:
        v = r.version
        cum = series[v.key]
        ax.plot(cum.index, cum.values, color=v.color, marker=v.marker, markevery=max(1, len(cum) // 8))
        x_end, y_end = cum.index[-1], float(cum.values[-1])
        ax.scatter([x_end], [y_end], s=40, color=v.color, edgecolor="black", zorder=3)
        stop = f"round {r.rounds}, converged" if r.converged else f"round {r.rounds}, not converged"
        ax.annotate(f"{y_end:{fmt}} {unit}\n({stop})", (x_end, y_end), xytext=(7, 0), textcoords="offset points",
                    fontsize=6.6, color=v.color, va="center", ha="left")
    ax.set_xlim(0, max(r.rounds for r in runs) * 1.6)
    ax.set_ylim(0, top * 1.18)
    ax.set_xlabel("Round")


def fig_communication(runs: list[Run], tables: dict, path: Path, a: argparse.Namespace) -> None:
    """(a) communication time per round, (b) cumulative until stop, (c) message parts (partitioned)."""
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(DOUBLE_COLUMN, 4.9))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.28)
        ax_a, ax_b, ax_c = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])

        # (a) Communication time per round (slowest client; clients communicate in parallel).
        per_round = {v.key: tables[v.key].groupby("round")["comm_s"].max() * MS for v in VERSIONS}
        for r in runs:
            v = r.version
            s = per_round[v.key]
            ax_a.plot(s.index, s.values, color=v.color, marker=v.marker, markevery=max(1, len(s) // 8))
        lo = min(s.min() for s in per_round.values())
        hi = max(s.max() for s in per_round.values())
        pad = max(hi - lo, 1.0)
        ax_a.set_ylim(lo - 1.5 * pad, hi + 1.5 * pad)
        ax_a.set_xlabel("Round")
        ax_a.set_ylabel("Communication per round (ms)")
        _panel(ax_a, "a", "Communication time per round")

        # (b) Cumulative communication time until each run stopped.
        _cumulative(ax_b, runs, {k: s.cumsum() / MS for k, s in per_round.items()}, "s", ".2f")
        ax_b.set_ylabel("Cumulative communication (s)")
        _panel(ax_b, "b", "Communication until stop")

        # (c) Bytes per client and round by message part, partitioned uplink / downlink.
        ax = ax_c
        xs = np.arange(len(MESSAGE_PARTS), dtype=float)
        links = list(dict.fromkeys(p for p, _, _ in MESSAGE_PARTS))
        for k, (link, _, _) in enumerate(MESSAGE_PARTS):
            xs[k] += 0.6 * links.index(link)
        w = 0.8 / len(VERSIONS)
        values = []
        for j, r in enumerate(runs):
            v = r.version
            t = tables[v.key]
            mean = np.array([t[col].mean() for _, col, _ in MESSAGE_PARTS], dtype=float)
            values.append(mean)
            pos = xs + (j - (len(runs) - 1) / 2) * w
            ax.bar(pos, mean, w, color=v.color, hatch=v.hatch, edgecolor="black", linewidth=0.4, zorder=2)
            for p, m in zip(pos, mean):
                ax.text(p, m * 1.25, _bytes_label(m), ha="center", va="bottom", fontsize=5.8, rotation=90)
        ymin, ymax = min(map(np.min, values)) / 2.5, max(map(np.max, values)) * 60
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        for link in links:
            idx = [k for k, (p, _, _) in enumerate(MESSAGE_PARTS) if p == link]
            left, right = xs[idx[0]] - 0.5, xs[idx[-1]] + 0.5
            ax.axvspan(left, right, color=LINK_SHADE[link], zorder=0, linewidth=0)
            ax.text((left + right) / 2, ymax / 1.6, link, ha="center", va="top", fontsize=8, fontweight="bold")
        ax.set_xlim(xs[0] - 0.55, xs[-1] + 0.55)
        ax.set_xticks(xs, [label for _, _, label in MESSAGE_PARTS], fontsize=7.2)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Bytes per client per round (log)")
        _panel(ax, "c", "Message size per client and round")

        _version_legend(fig, y=0.965)
        fig.text(0.5, 0.0, f"Link model: RTT {a.rtt_ms:g} ms, uplink {a.uplink_mbps:g} Mbit/s, downlink "
                 f"{a.downlink_mbps:g} Mbit/s, {ROUND_TRIPS} round trips per round; (a), (b): slowest client "
                 "(clients communicate in parallel). (c) Mean over clients and rounds.",
                 ha="center", va="top", fontsize=7.2, style="italic")
        _save(fig, path)


def fig_computation(runs: list[Run], tables: dict, path: Path) -> None:
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(DOUBLE_COLUMN, 4.9))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.28)
        ax_a, ax_b, ax_c = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])

        # (a) Total computation per round.
        for r in runs:
            v = r.version
            t = tables[v.key]
            ax_a.plot(t["round"], t["total_s"], color=v.color, marker=v.marker, markevery=max(1, len(t) // 8))
        ax_a.set_xlabel("Round")
        ax_a.set_ylabel("Computation per round (s)")
        ax_a.set_ylim(bottom=0)
        _panel(ax_a, "a", "Computation per round")

        # (b) Cumulative computation until each run stopped.
        _cumulative(ax_b, runs, {v.key: tables[v.key].set_index("round")["total_s"].cumsum() for v in VERSIONS},
                    "s", ".1f")
        ax_b.set_ylabel("Cumulative computation (s)")
        _panel(ax_b, "b", "Computation until stop")

        # (c) Breakdown per round: one group per part, partitioned client / server / total.
        groups = [(col, label.replace(" (+ DP in v10)", "\n(+ DP in v10)").replace(" + RF", " +\nRF")
                   .replace("ZKP proof ", "ZKP proof\n").replace("Server learning", "Server\nlearning"), side)
                  for col, label, side in PARTS] + [("total_s", "All\ncomputation", "total")]
        sides = {"client": "Client side (4 clients)", "server": "Server side", "total": "Total"}
        shade = {"client": "#f2f2f2", "server": "#fff4e0", "total": "#e8f1f8"}
        xs = np.arange(len(groups), dtype=float)
        order = list(dict.fromkeys(s for _, _, s in groups))
        for k, (_, _, side) in enumerate(groups):
            xs[k] += 0.6 * order.index(side)
        w = 0.8 / len(VERSIONS)
        allv = []
        for j, r in enumerate(runs):
            v = r.version
            t = tables[v.key]
            mean = np.array([t[col].mean() for col, _, _ in groups]) * MS
            lo = np.array([t[col].min() for col, _, _ in groups]) * MS
            hi = np.array([t[col].max() for col, _, _ in groups]) * MS
            allv += [lo, hi]
            pos = xs + (j - (len(runs) - 1) / 2) * w
            ax_c.bar(pos, mean, w, color=v.color, hatch=v.hatch, edgecolor="black", linewidth=0.4, zorder=2,
                     yerr=[mean - lo, hi - mean], capsize=1.5,
                     error_kw={"elinewidth": 0.6, "capthick": 0.6, "ecolor": "#333333"})
            for p, m, h in zip(pos, mean, hi):
                ax_c.text(p, h * 1.2, f"{m:,.0f}" if m >= 10 else f"{m:.2f}", ha="center", va="bottom",
                          fontsize=5.8, rotation=90)
        ymin, ymax = min(map(np.min, allv)) / 2.5, max(map(np.max, allv)) * 40
        ax_c.set_yscale("log")
        ax_c.set_ylim(ymin, ymax)
        for side in order:
            idx = [k for k, (_, _, s) in enumerate(groups) if s == side]
            left, right = xs[idx[0]] - 0.5, xs[idx[-1]] + 0.5
            ax_c.axvspan(left, right, color=shade[side], zorder=0, linewidth=0)
            ax_c.text((left + right) / 2, ymax / 1.6, sides[side], ha="center", va="top", fontsize=8,
                      fontweight="bold")
        ax_c.set_xlim(xs[0] - 0.55, xs[-1] + 0.55)
        ax_c.set_xticks(xs, [label for _, label, _ in groups], fontsize=7.3)
        ax_c.grid(axis="x", visible=False)
        ax_c.set_ylabel("Computation per round (ms, log)")
        _panel(ax_c, "c", "Computation breakdown per round")

        _version_legend(fig, y=0.965)
        fig.text(0.5, 0.0, "(c) Bars: mean per round; whiskers: min to max over the rounds; numbers: mean (ms). "
                 "Client parts are summed over the 4 clients.", ha="center", va="top", fontsize=7.2, style="italic")
        _save(fig, path)


def fig_dp_overhead(bench: pd.DataFrame, comp_v10: pd.DataFrame, run_v10: Run, path: Path) -> None:
    """(a) DP time per client per round (benchmark); (b) DP share of encoding and of all computation."""
    v = VERSIONS[0]
    clients = list(dict.fromkeys(bench["client"]))
    short = [c.split("(", 1)[1].rstrip(")") if "(" in c else c for c in clients]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.5), gridspec_kw={"width_ratios": [1.2, 1]})
        ax = axes[0]
        g = bench.groupby("client", sort=False)["dp_s"].agg(["mean", "min", "max"]).loc[clients] * MS
        x = np.arange(len(clients))
        ax.bar(x, g["mean"], 0.6, color=v.color, hatch=v.hatch, edgecolor="black", linewidth=0.4, zorder=2,
               yerr=[g["mean"] - g["min"], g["max"] - g["mean"]], capsize=2,
               error_kw={"elinewidth": 0.6, "capthick": 0.6})
        n_vec = bench.drop_duplicates("client").set_index("client").loc[clients, "vectors"]
        for xi, m, hi, n in zip(x, g["mean"], g["max"], n_vec):
            ax.text(xi, hi * 1.04, f"{m:.3f} ms\n({n} vectors)", ha="center", va="bottom", fontsize=6.6)
        ax.set_xticks(x, short)
        ax.set_ylabel("DP time per client per round (ms)")
        ax.set_ylim(0, g["max"].max() * 1.45)
        _panel(ax, "a", "Laplace DP cost (clip + noise)")

        ax = axes[1]
        dp_round = bench.groupby("round")["dp_s"].sum().mean()  # all clients, one round
        enc = comp_v10["encode_s"].mean()
        total = comp_v10["total_s"].mean()
        shares = [100 * dp_round / enc, 100 * dp_round / total]
        labels = ["of encoding\n(RF kernel + diffusion\nmap + DP)", "of all computation\nper round"]
        bars = ax.bar([0, 1], shares, 0.55, color=v.color, hatch=v.hatch, edgecolor="black", linewidth=0.4, zorder=2)
        for b, s in zip(bars, shares):
            ax.text(b.get_x() + b.get_width() / 2, s * 1.25, f"{s:.3g}%", ha="center", va="bottom", fontsize=7.5)
        ax.set_yscale("log")
        ax.set_ylim(min(shares) / 4, max(shares) * 8)
        ax.set_xticks([0, 1], labels, fontsize=7)
        ax.set_ylabel("DP share (%, log)")
        _panel(ax, "b", "DP overhead in v10")
        fig.text(0.5, -0.09, f"Benchmark of v10's Laplace mechanism on this machine: {bench['round'].nunique()} "
                 f"rounds × {len(clients)} clients, " r"$\varepsilon_r \sim U[2, 3]$"
                 "; whiskers: min to max. Shares use v10's "
                 "saved run.", ha="center", fontsize=7.2, style="italic")
        fig.tight_layout()
        _save(fig, path)


def fig_end_to_end(runs: list[Run], comp_tables: dict, comm_tables: dict, path: Path) -> None:
    """(a) rounds until stop, (b) total time until stop (computation + communication), (c) accuracy."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.45))
        x = np.arange(len(runs))
        ticks = [r.version.tick for r in runs]

        ax = axes[0]
        for i, r in enumerate(runs):
            v = r.version
            ax.bar(i, r.rounds, 0.55, color=v.color, hatch=v.hatch, edgecolor="black", linewidth=0.4, zorder=2)
            ax.text(i, r.rounds + 0.6, f"{r.rounds}\n({'converged' if r.converged else 'not converged'})",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, ticks)
        ax.set_ylabel("Rounds")
        ax.set_ylim(0, max(r.rounds for r in runs) * 1.35)
        _panel(ax, "a", "Rounds until stop")

        ax = axes[1]
        for i, r in enumerate(runs):
            v = r.version
            comp = comp_tables[v.key]["total_s"].sum()
            comm = comm_tables[v.key].groupby("round")["comm_s"].max().sum()
            ax.bar(i, comp, 0.55, color=v.color, hatch=v.hatch, edgecolor="black", linewidth=0.4, zorder=2)
            ax.bar(i, comm, 0.55, bottom=comp, color="white", hatch="xxxx", edgecolor=v.color, linewidth=0.6,
                   zorder=2)
            ax.text(i, (comp + comm) * 1.02, f"{comp + comm:.1f} s", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, ticks)
        ax.set_ylabel("Time until stop (s)")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.legend(handles=[Patch(facecolor="#888888", edgecolor="black", label="Computation"),
                           Patch(facecolor="white", edgecolor="#555555", hatch="xxxx", label="Communication")],
                  loc="upper right", fontsize=6.8, handlelength=1.2)
        _panel(ax, "b", "Total time until stop")

        ax = axes[2]
        for i, r in enumerate(runs):
            v = r.version
            acc = r.history["accuracy[all]"]
            ax.bar(i, acc.mean(), 0.55, color=v.color, hatch=v.hatch, edgecolor="black", linewidth=0.4, zorder=2,
                   yerr=[[acc.mean() - acc.min()], [acc.max() - acc.mean()]], capsize=2,
                   error_kw={"elinewidth": 0.6, "capthick": 0.6})
            ax.text(i, acc.max() + 0.01, f"{acc.mean():.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, ticks)
        ax.set_ylabel("Global Server accuracy")
        ax.set_ylim(0.5, 1.0)
        _panel(ax, "c", "Accuracy (context)")
        fig.text(0.5, -0.08, "(b) Communication with clients in parallel (slowest client per round). "
                 "(c) Bar: mean over the rounds; whiskers: min to max.", ha="center", fontsize=7.2, style="italic")
        fig.tight_layout()
        _save(fig, path)


# --- averaged classification metrics ----------------------------------------------------
METRICS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
           "specificity": "Specificity", "f1": "F1 score", "mcc": "MCC", "kappa": "Cohen's κ"}


def metrics_summary(runs: list[Run]) -> pd.DataFrame:
    """Mean, min and max over the rounds of the seven metrics, per client and Global Server."""
    rows = []
    for r in runs:
        for client in [*r.clients, GLOBAL]:
            key = "all" if client == GLOBAL else client
            for metric, name in METRICS.items():
                s = r.history[f"{metric}[{key}]"]
                rows.append({"version": r.version.key, "dp": plain(r.version.label), "rounds": r.rounds,
                             "client": client, "metric": name.replace("κ", "kappa"),
                             "mean": s.mean(), "min": s.min(), "max": s.max(), "std": s.std(ddof=1)})
    return pd.DataFrame(rows)


def metrics_latex(table: pd.DataFrame, path: Path) -> None:
    """Booktabs table: mean metric per client (rows) and version (column pairs)."""
    names = [n.replace("κ", "kappa") for n in METRICS.values()]
    clients = list(dict.fromkeys(table["client"]))
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{n.replace('kappa', r'$\kappa$')}}}" for n in names)
    sub = " & ".join(["DP & No DP"] * len(names))
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{Validation metrics averaged over the rounds, with Laplace DP (v10, "
             r"$\varepsilon_r\sim U[2,3]$, 30 rounds) and without DP (v13, 7 rounds), seed 42.}",
             r"\label{tab:dp_metrics}", r"\begin{tabular}{l" + "rr" * len(names) + "}", r"\toprule",
             "Client & " + head + r" \\", " & " + sub + r" \\", r"\midrule"]
    for c in clients:
        cells = []
        for n in names:
            for key in ("v10", "v13"):
                v = table[(table["client"] == c) & (table["metric"] == n) & (table["version"] == key)]["mean"]
                cells.append(f"{float(v.iloc[0]):.3f}")
        lines.append(c.replace("&", r"\&") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def fig_metrics(runs: list[Run], path: Path) -> None:
    """(a) Global Server, (b)-(e) each client: the seven metrics averaged over the rounds;
    (f) difference No DP - DP of every mean metric."""
    clients = [*runs[0].clients, GLOBAL]
    order = [GLOBAL, *runs[0].clients]
    short = {c: (c.split("(", 1)[1].rstrip(")") if "(" in c else c) for c in clients}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 2, figsize=(DOUBLE_COLUMN, 7.4))
        axes = axes.ravel()
        x = np.arange(len(METRICS))
        w = 0.8 / len(runs)
        lows = []
        for r in runs:
            for c in clients:
                key = "all" if c == GLOBAL else c
                lows += [r.history[f"{m}[{key}]"].min() for m in METRICS]
        y_low = min(0.0, np.floor(min(lows) * 10) / 10)
        for ax, letter, client in zip(axes, "abcde", order):
            key = "all" if client == GLOBAL else client
            for j, r in enumerate(runs):
                v = r.version
                s = [r.history[f"{m}[{key}]"] for m in METRICS]
                mean = np.array([q.mean() for q in s])
                lo = np.array([q.min() for q in s])
                hi = np.array([q.max() for q in s])
                pos = x + (j - (len(runs) - 1) / 2) * w
                # Clip at 0: a metric equal in every round gives mean - min of about -1e-17.
                ax.bar(pos, mean, w, color=v.color, hatch=v.hatch, edgecolor="black", linewidth=0.4, zorder=2,
                       yerr=[np.maximum(mean - lo, 0), np.maximum(hi - mean, 0)], capsize=1.5,
                       error_kw={"elinewidth": 0.6, "capthick": 0.6, "ecolor": "#333333"})
                for p, m, h in zip(pos, mean, hi):
                    ax.text(p, h + 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=5.4, rotation=90)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xticks(x, list(METRICS.values()), fontsize=7, rotation=30, ha="right", rotation_mode="anchor")
            ax.set_ylim(y_low, 1.32)
            ax.set_yticks(np.arange(max(y_low, -1.0), 1.01, 0.2).round(1))
            if letter in "ace":
                ax.set_ylabel("Mean over the rounds")
            title = "Global Server (all clients pooled)" if client == GLOBAL else client
            _panel(ax, letter, title)

        # (f) Difference of the means, No DP - DP, per client and metric.
        ax = axes[5]
        diff = np.array([[runs[1].history[f"{m}[{'all' if c == GLOBAL else c}]"].mean()
                          - runs[0].history[f"{m}[{'all' if c == GLOBAL else c}]"].mean() for m in METRICS]
                         for c in order])
        lim = max(0.05, np.abs(diff).max())
        im = ax.imshow(diff, cmap="PuOr", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(diff.shape[0]):
            for k in range(diff.shape[1]):
                val = diff[i, k]
                ax.text(k, i, f"{val:+.3f}", ha="center", va="center", fontsize=6.3,
                        color="white" if abs(val) > 0.6 * lim else "black")
        ax.set_xticks(range(len(METRICS)), list(METRICS.values()), fontsize=7, rotation=30, ha="right",
                      rotation_mode="anchor")
        ax.set_yticks(range(len(order)), ["Global Server" if c == GLOBAL else short[c] for c in order], fontsize=7)
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cbar.ax.tick_params(labelsize=6.5)
        cbar.set_label("No DP − DP", fontsize=7)
        _panel(ax, "f", "Difference of the means")

        _version_legend(fig, y=0.985)
        r10, r13 = runs
        fig.text(0.5, 0.0, f"Bars: mean over the rounds (v10: {r10.rounds} rounds, v13: {r13.rounds} rounds); "
                 "whiskers: min to max over the rounds; numbers: mean. (f) Positive: higher without DP.",
                 ha="center", va="top", fontsize=7.2, style="italic")
        fig.tight_layout(rect=(0, 0.01, 1, 0.965), h_pad=1.6)
        _save(fig, path)


# --- tables ------------------------------------------------------------------------
def overall_summary(runs: list[Run], comm: pd.DataFrame, comp: pd.DataFrame,
                    bench: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for r in runs:
        cm, cp = comm.set_index("version").loc[r.version.key], comp.set_index("version").loc[r.version.key]
        row = {"Version": plain(r.version.label), "Rounds": r.rounds, "Converged": "yes" if r.converged else "no",
               "Comm per client-round (ms)": cm["comm_ms_per_client_round_mean"],
               "Up per client-round (KB)": cm["bytes_up_per_client_round_mean"] / 1e3,
               "Down per client-round (KB)": cm["bytes_down_per_client_round_mean"] / 1e3,
               "Comm until stop (s)": cm["comm_s_total_until_stop (clients in parallel)"],
               "Data until stop (MB)": cm["MB_total_until_stop"],
               "Comp per round (s)": cp["total per round (s, mean)"],
               "Comp per client-round (s)": cp["per client per round (s, mean)"],
               "Comp until stop (s)": cp["total computation until stop (s)"],
               **{f"Global Server {name.replace('κ', 'kappa')} (mean)": r.history[f"{m}[all]"].mean()
                  for m, name in METRICS.items()}}
        if bench is not None:
            row["DP time per client-round (ms)"] = bench["dp_s"].mean() * MS if r.version.key == "v10" else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def to_latex(table: pd.DataFrame, path: Path, caption: str) -> None:
    t = table.set_index("Version").T
    cols = list(t.columns)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", r"\label{tab:dp_comparison}",
             r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             "Quantity & " + " & ".join(c.replace("eps_r ~ U[2, 3]", r"$\varepsilon_r\sim U[2,3]$") for c in cols)
             + r" \\", r"\midrule"]
    for name, values in t.iterrows():
        cells = []
        for v in values:
            if isinstance(v, str):
                cells.append(v)
            else:
                v = float(v)
                cells.append(f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --- main --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="folder holding the two version folders")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--rtt-ms", type=float, default=50.0, help="network round-trip time (ms)")
    ap.add_argument("--uplink-mbps", type=float, default=20.0, help="client -> server bandwidth (Mbit/s)")
    ap.add_argument("--downlink-mbps", type=float, default=50.0, help="server -> client bandwidth (Mbit/s)")
    ap.add_argument("--benchmark-rounds", type=int, default=30, help="rounds of the DP overhead benchmark")
    ap.add_argument("--no-benchmark", action="store_true", help="skip the DP overhead benchmark")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # labels contain ε; Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    figs = a.out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    runs = [load_run(a.root, v) for v in VERSIONS]
    print("Versions:")
    for r in runs:
        print(f"  {r.version.key}: {r.version.folder} | DP {r.config.get('privacy', {}).get('mechanism')} | "
              f"{r.rounds} rounds ({'converged' if r.converged else 'not converged'}) x {len(r.clients)} clients")
    notes = check_comparable(runs)
    print("Same learning settings and ZKP; the versions differ only in the DP." if not notes
          else "WARNING, the runs differ:\n  " + "\n  ".join(notes))

    comm_tables = {r.version.key: communication_table(r, a) for r in runs}
    comp_tables = {r.version.key: computation_table(r) for r in runs}
    comm = communication_summary(runs, comm_tables)
    comp = computation_summary(runs, comp_tables)
    comm.to_csv(a.out / "communication_summary.csv", index=False)
    comp.to_csv(a.out / "computation_summary.csv", index=False)
    pd.concat(comm_tables.values()).to_csv(a.out / "communication_per_round.csv", index=False)
    pd.concat(comp_tables.values()).to_csv(a.out / "computation_per_round.csv", index=False)

    bench = None
    if not a.no_benchmark:
        print(f"\nDP overhead benchmark: v10's Laplace mechanism, {a.benchmark_rounds} rounds x "
              f"{len(runs[0].clients)} clients ...")
        bench = dp_benchmark(a.root, runs[0], a.benchmark_rounds)
        bench.to_csv(a.out / "dp_overhead_benchmark.csv", index=False)
        s = (bench.groupby("client", sort=False)["dp_s"].agg(["min", "max", "mean"]) * MS)
        print(s.rename(columns=lambda c: f"{c} (ms)").to_string(float_format=lambda v: f"{v:.4f}"))

    summary = overall_summary(runs, comm, comp, bench)
    summary.to_csv(a.out / "comparison_summary.csv", index=False)
    to_latex(summary, a.out / "comparison_summary.tex",
             "Communication and computation of FedRF-Diff with and without Laplace DP "
             "(seed 42, 4 clients, Schnorr ZKP).")

    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    print(f"\n=== Communication (RTT {a.rtt_ms:g} ms, up {a.uplink_mbps:g} Mbit/s, "
          f"down {a.downlink_mbps:g} Mbit/s, {ROUND_TRIPS} round trips per round) ===")
    print(comm.drop(columns="dp").set_index("version").T.to_string(float_format=fmt))
    print("\n=== Computation ===")
    print(comp.drop(columns="dp").set_index("version").T.to_string(float_format=fmt))
    print("\n=== Summary ===")
    print(summary.set_index("Version").T.to_string(float_format=fmt))

    metrics = metrics_summary(runs)
    metrics.to_csv(a.out / "metrics_summary.csv", index=False)
    metrics_latex(metrics, a.out / "metrics_summary.tex")
    print("\n=== Validation metrics averaged over the rounds (mean; v10: "
          f"{runs[0].rounds} rounds, v13: {runs[1].rounds} rounds) ===")
    wide = metrics.pivot_table(index=["client", "metric"], columns="version", values="mean", sort=False)
    wide["No DP - DP"] = wide["v13"] - wide["v10"]
    print(wide.rename(columns={"v10": "Laplace DP (v10)", "v13": "No DP (v13)"}).to_string(float_format=fmt))

    print("\nFigures (600-dpi PNG + vector PDF):")
    fig_communication(runs, comm_tables, figs / "fig1_communication_comparison", a)
    fig_computation(runs, comp_tables, figs / "fig2_computation_comparison")
    if bench is not None:
        fig_dp_overhead(bench, comp_tables["v10"], runs[0], figs / "fig3_dp_overhead")
    fig_end_to_end(runs, comp_tables, comm_tables, figs / "fig4_end_to_end")
    fig_metrics(runs, figs / "fig5_averaged_metrics")
    print(f"\nTables in {a.out.resolve()}")


if __name__ == "__main__":
    main()
