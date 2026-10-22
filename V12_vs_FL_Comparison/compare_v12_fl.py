"""Compare FedRF-Diff v12 with the Federated-Learning project: computation, communication, metrics.

    v12  Fed_Code_v12_ProposedPedersenZKP_LaplaceDP_eps2-3
         Diffusion-map latents + Procrustes alignment + global RF on the latents,
         Laplace DP on the latents (eps_t ~ U[2, 3]).
    FL   Federated-Learning_ProposedZKP
         Whole random forests exchanged as explicit trees, tree-level aggregation,
         Laplace DP on the leaf vectors (eps_t ~ U[2, 3]).

Both use the same proposed Pedersen ZKP for client authentication (the same
zkp.py, group and lambda = 2: chained commitment C_t = g^v_t h^r_t (C_{t-1})^lambda,
server-sent challenge), so the comparison isolates the learning designs.

Inputs are the two saved runs (outputs/run/ of each folder; make them with
``python -m fedrf_diff`` in each folder). The projects log different
quantities, so they are mapped onto shared categories:

  Client computation (per client and round)
    Encoding / serialization     v12: RF kernel + diffusion map + DP     FL: forest serialization
    ZKP proof generation         v12: proof generation                   FL: proof generation
    Local model update           v12: decoding + RF retraining           FL: local tuning (CV) + refinement
    Evaluation / convergence     v12: validation loss and counts         FL: three-part convergence test
  Server computation (per round)
    Aggregation / learning       v12: alignment + M_G + stopping rule    FL: tree-level aggregation
    Update validation            v12: none                               FL: structural tree validation
    ZKP verification             v12: all clients' proofs                FL: all clients' proofs

  Communication (per client and round; the server's row is what it receives and sends)
    bytes from each project's own cost table, and a modelled time
        time = round trips x RTT + 8 x bytes_up / uplink + 8 x bytes_down / downlink
    with 2 round trips per round for the (shared) interactive ZKP session in both
    (C_t and T up / c down; r_v and r_s up / verdict down), plus the learning
    messages: v12 2 more (latents and labels up / M_G, A_c and labels down;
    metrics up / ack) = 4, FL 1 more (convergence flag up / decision down) = 3;
    FL's forest travels with the responses and its global forest with the verdict.
    The server's communication time per round is that of its slowest client
    (clients communicate in parallel).

  Metrics: accuracy, precision, recall, specificity, F1, MCC and Cohen's kappa of
  each project's *global* model on every client's validation patients and on all
  of them pooled (Global Server), averaged over the rounds and in the last round.
  The two projects split the data differently (v12 keeps duplicate rows and sets
  100 landmark patients aside, FL drops duplicates), so their validation sets differ.

Terminal: tables of the averaged computation and communication per client and
Global Server, and of the averaged and last-round metrics. Files (--out,
default V12_vs_FL_Comparison/outputs): CSV and LaTeX tables, and figures as
600-dpi PNG + vector PDF.

Usage:
    python compare_v12_fl.py
    python compare_v12_fl.py --rtt-ms 20 --uplink-mbps 50 --downlink-mbps 100
"""

from __future__ import annotations

import argparse
import json
import sys
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
CLIENTS = ["Client 1(Hungary)", "Client 2(Switzerland)", "Client 3(VA Long Beach)", "Client 4(Cleveland)"]
ENTITIES = [*CLIENTS, GLOBAL]
KB = 1024.0
METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "MCC", "Cohen's Kappa"]
CLIENT_PARTS = ["Encoding / serialization", "ZKP proof generation", "Local model update",
                "Evaluation / convergence check"]
SERVER_PARTS = ["Aggregation / learning", "Update validation", "ZKP proof verification"]


@dataclass(frozen=True)
class Project:
    key: str
    label: str
    tick: str
    folder: str
    round_trips: int  # message exchanges per round between a client and the server
    color: str  # Okabe-Ito
    hatch: str
    marker: str


PROJECTS = (
    Project("v12", "FedRF-Diff v12 (latents + Procrustes, proposed Pedersen ZKP, Laplace DP)", "FedRF-Diff v12",
            "Fed_Code_v12_ProposedPedersenZKP_LaplaceDP_eps2-3", 4, "#009E73", "....", "^"),
    Project("FL", "Federated-Learning (tree-level aggregation, proposed Pedersen ZKP, Laplace DP)",
            "Federated-Learning", "Federated-Learning_ProposedZKP", 3, "#CC79A7", "\\\\\\\\", "D"),
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


def short(name: str) -> str:
    """'Client 3(VA Long Beach)' -> 'VA Long Beach'."""
    return name.split("(", 1)[1].rstrip(")") if "(" in name else name


# --- loading and harmonising -------------------------------------------------------
@dataclass
class Run:
    project: Project
    computation: pd.DataFrame  # Round, Entity, category columns (seconds), Total
    communication: pd.DataFrame  # Round, Entity, Sent (KB), Received (KB)
    metrics: pd.DataFrame  # Round, Entity, the seven metrics
    rounds: int
    val_sizes: dict


def _need(d: Path, files: list[str], folder: str) -> None:
    missing = [f for f in files if not (d / f).exists()]
    if missing:
        sys.exit(f"{d}: missing {', '.join(missing)}; run `python -m fedrf_diff` in {folder} first")


def zkp_fingerprint(root: Path, p: Project) -> dict:
    """What a project's fedrf_diff/zkp.py implements (read from the source, not the labels)."""
    src = (root / p.folder / "fedrf_diff" / "zkp.py").read_text(encoding="utf-8")
    seed = next((line.split("=", 1)[1].split("#")[0].strip() for line in src.splitlines()
                 if line.startswith("SEED =")), "?")
    return {"chained commitment C_t (C_{t-1})^lambda": "class ClientProver" in src and "C_prev" in src,
            "server-sent challenge": "def challenge(self)" in src,
            "Fiat-Shamir": "Fiat-Shamir" in src and "def challenge(self)" not in src,
            "group seed": seed}


def same_zkp(root: Path) -> tuple[bool, list[str]]:
    a, b = (zkp_fingerprint(root, p) for p in PROJECTS)
    lines = [f"  {PROJECTS[0].key}: {a}", f"  {PROJECTS[1].key}: {b}"]
    return a == b and a["chained commitment C_t (C_{t-1})^lambda"], lines


def load_v12(root: Path, p: Project) -> Run:
    d = root / p.folder / "outputs" / "run"
    _need(d, ["costs.csv", "history.csv", "final_metrics.csv"], p.folder)
    costs = pd.read_csv(d / "costs.csv")
    cl, sv = costs[costs["client"] != GLOBAL], costs[costs["client"] == GLOBAL]
    comp = pd.DataFrame({
        "Round": cl["round"], "Entity": cl["client"],
        "Encoding / serialization": cl["encode_dp_s"], "ZKP proof generation": cl["proof_generation_s"],
        "Local model update": cl["decode_update_s"], "Evaluation / convergence check": cl["evaluate_s"]})
    comp_s = pd.DataFrame({"Round": sv["round"], "Entity": GLOBAL, "Aggregation / learning": sv["learning_s"],
                           "Update validation": 0.0, "ZKP proof verification": sv["proof_verification_s"]})
    comm = pd.DataFrame({"Round": cl["round"], "Entity": cl["client"],
                         "Sent (KB)": cl["bytes_up"] / KB, "Received (KB)": cl["bytes_down"] / KB})
    # v12's server row: bytes_up = what the server sends, bytes_down = what it receives.
    comm_s = pd.DataFrame({"Round": sv["round"], "Entity": GLOBAL,
                           "Sent (KB)": sv["bytes_up"] / KB, "Received (KB)": sv["bytes_down"] / KB})
    hist = pd.read_csv(d / "history.csv")
    names = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "specificity": "Specificity",
             "f1": "F1 Score", "mcc": "MCC", "kappa": "Cohen's Kappa"}
    rows = []
    for _, h in hist.iterrows():
        for e in ENTITIES:
            key = "all" if e == GLOBAL else e
            rows.append({"Round": int(h["round"]), "Entity": e, **{n: h[f"{m}[{key}]"] for m, n in names.items()}})
    final = pd.read_csv(d / "final_metrics.csv").set_index("client")
    val = {c: int(final.loc[c, "n_val"]) for c in CLIENTS}
    val[GLOBAL] = sum(val.values())
    return Run(p, _finish(comp, comp_s), pd.concat([comm, comm_s], ignore_index=True), pd.DataFrame(rows),
               int(hist["round"].max()), val)


def load_fl(root: Path, p: Project) -> Run:
    d = root / p.folder / "outputs" / "run"
    files = ["communication_computation_cost.csv", "zkp_times.csv", "tree_transfer_times.csv",
             "validation_metrics.csv"]
    _need(d, files, p.folder)
    cost = pd.read_csv(d / "communication_computation_cost.csv")
    zkp = pd.read_csv(d / "zkp_times.csv")
    tr = pd.read_csv(d / "tree_transfer_times.csv")
    cl = cost[cost["Client"] != GLOBAL].merge(zkp, on=["Round", "Client"]).merge(tr, on=["Round", "Client"])
    ms = 1000.0
    comp = pd.DataFrame({
        "Round": cl["Round"], "Entity": cl["Client"],
        "Encoding / serialization": cl["Serialization Time (ms)"] / ms,
        "ZKP proof generation": cl["ZKP Proof Generation Time (ms)"] / ms,
        "Local model update": cl["Local Refinement Time (ms)"] / ms,  # includes the local tuning
        "Evaluation / convergence check": cl["Convergence Check Time (ms)"] / ms})
    sv = cost[cost["Client"] == GLOBAL].set_index("Round")
    per_round = cl.groupby("Round")[["Tree Validation Time (ms)", "ZKP Proof Verification Time (ms)"]].sum()
    comp_s = pd.DataFrame({"Round": sv.index, "Entity": GLOBAL,
                           "Aggregation / learning": sv["Aggregation Time (ms)"].to_numpy() / ms,
                           "Update validation": per_round.loc[sv.index, "Tree Validation Time (ms)"].to_numpy() / ms,
                           "ZKP proof verification":
                               per_round.loc[sv.index, "ZKP Proof Verification Time (ms)"].to_numpy() / ms})
    comm = pd.DataFrame({"Round": cl["Round"], "Entity": cl["Client"],
                         "Sent (KB)": cl["Uplink (KB)"], "Received (KB)": cl["Downlink (KB)"]})
    # FL's server row: Uplink = what all clients sent (server receives), Downlink = what it sends.
    comm_s = pd.DataFrame({"Round": sv.index, "Entity": GLOBAL,
                           "Sent (KB)": sv["Downlink (KB)"].to_numpy(), "Received (KB)": sv["Uplink (KB)"].to_numpy()})
    vm = pd.read_csv(d / "validation_metrics.csv").rename(
        columns={"Client": "Entity", "Matthews Correlation Coefficient (MCC)": "MCC"})
    val = _fl_val_sizes(root / p.folder)
    return Run(p, _finish(comp, comp_s), pd.concat([comm, comm_s], ignore_index=True),
               vm[["Round", "Entity", *METRICS]], int(cost["Round"].max()), val)


def _fl_val_sizes(folder: Path) -> dict:
    """Validation patients per client, from the project's README table (train / validation)."""
    sizes = {}
    readme = folder / "README.md"
    if readme.exists():
        for line in readme.read_text(encoding="utf-8").splitlines():
            for c in CLIENTS:
                if line.startswith(f"| {c} |"):
                    cells = [x.strip() for x in line.strip("|").split("|")]
                    if len(cells) > 2 and "/" in cells[2]:
                        sizes[c] = int(cells[2].split("/")[1])
    if len(sizes) == len(CLIENTS):
        sizes[GLOBAL] = sum(sizes.values())
    return sizes


def _finish(comp: pd.DataFrame, comp_s: pd.DataFrame) -> pd.DataFrame:
    comp = comp.copy()
    comp_s = comp_s.copy()
    comp["Total"] = comp[CLIENT_PARTS].sum(axis=1)
    comp_s["Total"] = comp_s[SERVER_PARTS].sum(axis=1)
    return pd.concat([comp, comp_s], ignore_index=True)


# --- tables ------------------------------------------------------------------------
def computation_table(runs: list[Run]) -> pd.DataFrame:
    """Mean, min and max per round of every category and the total, per entity and project."""
    rows = []
    for r in runs:
        for e in ENTITIES:
            sub = r.computation[r.computation["Entity"] == e]
            parts = CLIENT_PARTS if e != GLOBAL else SERVER_PARTS
            row = {"Project": r.project.key, "Entity": e}
            for col in [*parts, "Total"]:
                row[f"{col} mean (s)"] = sub[col].mean()
            row["Total min (s)"] = sub["Total"].min()
            row["Total max (s)"] = sub["Total"].max()
            row["Total over the run (s)"] = sub["Total"].sum()
            rows.append(row)
    return pd.DataFrame(rows)


def link_time(sent: float, received: float, round_trips: int, a: argparse.Namespace) -> float:
    """Seconds for one round of a client: latency + transmission of KB sent and received."""
    return (round_trips * a.rtt_ms / 1000.0 + 8 * sent * KB / (a.uplink_mbps * 1e6)
            + 8 * received * KB / (a.downlink_mbps * 1e6))


def communication_table(runs: list[Run], a: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    rows, per_round = [], {}
    for r in runs:
        c = r.communication.copy()
        cl = c["Entity"] != GLOBAL
        c.loc[cl, "Time (s)"] = [link_time(s, rcv, r.project.round_trips, a)
                                 for s, rcv in zip(c.loc[cl, "Sent (KB)"], c.loc[cl, "Received (KB)"])]
        slowest = c[cl].groupby("Round")["Time (s)"].max()
        c.loc[~cl, "Time (s)"] = c.loc[~cl, "Round"].map(slowest).to_numpy()
        per_round[r.project.key] = c
        for e in ENTITIES:
            sub = c[c["Entity"] == e]
            rows.append({"Project": r.project.key, "Entity": e,
                         "Sent mean (KB)": sub["Sent (KB)"].mean(), "Received mean (KB)": sub["Received (KB)"].mean(),
                         "Total mean (KB)": (sub["Sent (KB)"] + sub["Received (KB)"]).mean(),
                         "Time mean (ms)": sub["Time (s)"].mean() * 1000, "Time min (ms)": sub["Time (s)"].min() * 1000,
                         "Time max (ms)": sub["Time (s)"].max() * 1000,
                         "Data over the run (MB)": (sub["Sent (KB)"] + sub["Received (KB)"]).sum() / KB,
                         "Time over the run (s)": sub["Time (s)"].sum()})
    return pd.DataFrame(rows), per_round


def metric_tables(runs: list[Run]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(averaged over the rounds, last round): one row per entity and metric, both projects + difference."""
    avg, last = [], []
    a, b = runs
    for e in ENTITIES:
        ma = a.metrics[a.metrics["Entity"] == e]
        mb = b.metrics[b.metrics["Entity"] == e]
        la, lb = ma[ma["Round"] == ma["Round"].max()], mb[mb["Round"] == mb["Round"].max()]
        for m in METRICS:
            avg.append({"Entity": e, "Metric": m, a.project.key: ma[m].mean(), b.project.key: mb[m].mean(),
                        f"{a.project.key} min": ma[m].min(), f"{a.project.key} max": ma[m].max(),
                        f"{b.project.key} min": mb[m].min(), f"{b.project.key} max": mb[m].max()})
            last.append({"Entity": e, "Metric": m, a.project.key: float(la[m].iloc[0]),
                         b.project.key: float(lb[m].iloc[0])})
    avg, last = pd.DataFrame(avg), pd.DataFrame(last)
    for t in (avg, last):
        t[f"{b.project.key} - {a.project.key}"] = t[b.project.key] - t[a.project.key]
    return avg, last


def latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, digits: int = 3) -> None:
    """Plain booktabs table of a DataFrame (no jinja2 needed)."""
    cols = list(df.columns)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "".join("l" if df[c].dtype == object else "r" for c in cols) + "}",
             r"\toprule", " & ".join(str(c).replace("%", r"\%").replace("&", r"\&") for c in cols) + r" \\",
             r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, str):
                cells.append(v.replace("&", r"\&"))
            elif pd.isna(v):
                cells.append("--")
            else:
                cells.append(f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{digits}f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --- figures ----------------------------------------------------------------------
def _panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold")


def _save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {path.with_suffix('.png')} (+ .pdf)")


def _legend(fig, y: float) -> None:
    fig.legend(handles=[Patch(facecolor=p.color, edgecolor="black", hatch=p.hatch, linewidth=0.4, label=p.label)
                        for p in PROJECTS], loc="lower center", ncol=1, bbox_to_anchor=(0.5, y), fontsize=7.3)


def _fmt(v: float) -> str:
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 0.1:
        return f"{v:.2f}"
    if v >= 0.001:
        return f"{v:.3f}"
    return f"{v:.4f}" if v > 0 else "0"


def _grouped(ax, groups: list[str], values: dict, err: dict | None = None, log: bool = True,
             label_fmt=_fmt, fontsize: float = 5.8) -> None:
    """One group per x position, one bar per project; err = {key: (lo, hi)} absolute values."""
    x = np.arange(len(groups), dtype=float)
    w = 0.8 / len(PROJECTS)
    tops = []
    for j, p in enumerate(PROJECTS):
        v = np.asarray(values[p.key], dtype=float)
        pos = x + (j - (len(PROJECTS) - 1) / 2) * w
        kw = {}
        top = v
        if err is not None:
            lo, hi = (np.asarray(e, dtype=float) for e in err[p.key])
            kw = dict(yerr=[np.maximum(v - lo, 0), np.maximum(hi - v, 0)], capsize=1.5,
                      error_kw={"elinewidth": 0.6, "capthick": 0.6, "ecolor": "#333333"})
            top = np.maximum(hi, v)
        ax.bar(pos, v, w, color=p.color, hatch=p.hatch, edgecolor="black", linewidth=0.4, zorder=2, **kw)
        bottom = np.minimum(np.asarray(err[p.key][0], dtype=float), v) if err is not None else v
        for pp, val, t, b in zip(pos, v, top, bottom):
            if log:
                if val > 0:
                    ax.text(pp, t * 1.25, label_fmt(val), ha="center", va="bottom", fontsize=fontsize, rotation=90)
            elif val < 0:  # below a negative bar, clear of the zero line
                ax.text(pp, b - 0.03, label_fmt(val), ha="center", va="top", fontsize=fontsize, rotation=90)
            else:
                ax.text(pp, t + 0.02, label_fmt(val), ha="center", va="bottom", fontsize=fontsize, rotation=90)
        tops.append(top)
    ax.set_xticks(x, groups)
    if log:
        ax.set_yscale("log")


def fig_computation(runs: list[Run], path: Path) -> None:
    """(a) total computation per round per client and server; (b) client categories; (c) server categories."""
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(DOUBLE_COLUMN, 5.9))
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], width_ratios=[1.3, 1], hspace=0.42, wspace=0.25,
                                top=0.86, bottom=0.12)
        ax_a = fig.add_subplot(grid[0, :])
        ax_b, ax_c = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])

        vals, err = {}, {}
        for r in runs:
            g = r.computation.groupby("Entity")["Total"].agg(["mean", "min", "max"]).loc[ENTITIES]
            vals[r.project.key], err[r.project.key] = g["mean"], (g["min"], g["max"])
        _grouped(ax_a, [short(e) if e != GLOBAL else "Global Server" for e in ENTITIES], vals, err)
        ax_a.axvspan(len(CLIENTS) - 0.5, len(CLIENTS) + 0.5, color="#fff4e0", zorder=0, linewidth=0)
        ax_a.set_ylim(min(e[0].min() for e in err.values()) / 3, max(e[1].max() for e in err.values()) * 30)
        ax_a.set_ylabel("Computation per round (s, log)")
        _panel(ax_a, "a", "Computation per round: each client and the Global Server")

        for ax, letter, parts, entity_filter, title in (
                (ax_b, "b", CLIENT_PARTS, lambda d: d[d["Entity"] != GLOBAL], "Client side (mean per client)"),
                (ax_c, "c", SERVER_PARTS, lambda d: d[d["Entity"] == GLOBAL], "Server side")):
            vals, err = {}, {}
            for r in runs:
                d = entity_filter(r.computation)
                vals[r.project.key] = [d[p].mean() for p in parts]
                err[r.project.key] = ([d[p].min() for p in parts], [d[p].max() for p in parts])
            ticks = {"Encoding / serialization": "Encoding /\nserialization", "ZKP proof generation": "ZKP proof\ngeneration",
                     "Local model update": "Local model\nupdate",
                     "Evaluation / convergence check": "Evaluation /\nconvergence\ncheck",
                     "Aggregation / learning": "Aggregation /\nlearning", "Update validation": "Update\nvalidation",
                     "ZKP proof verification": "ZKP proof\nverification"}
            _grouped(ax, [ticks[p] for p in parts], vals, err)
            positive = [v for k in vals for v in np.asarray(err[k][0], dtype=float) if v > 0]
            ax.set_ylim(min(positive) / 3, max(np.max(err[k][1]) for k in err) * 40)
            ax.tick_params(axis="x", labelsize=7)
            ax.set_ylabel("Seconds per round (log)")
            _panel(ax, letter, title)
        # v12 has no update-validation step: mark the empty slot of its bar.
        ax_c.text(1 - 0.2, ax_c.get_ylim()[0] * 1.8, "none (v12)", ha="center", va="bottom", fontsize=6.0,
                  color="#333333", rotation=90)

        _legend(fig, 0.955)
        fig.text(0.5, 0.0, "Bars: mean over the 30 rounds; whiskers: min to max; numbers: mean (s). FL's local model "
                 "update includes its per-round hyper-parameter search (5-fold CV).", ha="center", va="top",
                 fontsize=7.2, style="italic")
        _save(fig, path)


def fig_communication(runs: list[Run], per_round: dict, path: Path, a: argparse.Namespace) -> None:
    """(a) KB sent, (b) KB received, (c) modelled communication time, per client and Global Server."""
    groups = [short(e) if e != GLOBAL else "Global\nServer" for e in ENTITIES]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(DOUBLE_COLUMN, 7.0))
        for ax, letter, col, title, unit in (
                (axes[0], "a", "Sent (KB)", "Data sent per round", "KB per round (log)"),
                (axes[1], "b", "Received (KB)", "Data received per round", "KB per round (log)"),
                (axes[2], "c", "Time (s)", "Communication time per round", "ms per round (log)")):
            vals, err = {}, {}
            scale = 1000.0 if col == "Time (s)" else 1.0
            for r in runs:
                c = per_round[r.project.key]
                g = c.groupby("Entity")[col].agg(["mean", "min", "max"]).loc[ENTITIES] * scale
                vals[r.project.key], err[r.project.key] = g["mean"], (g["min"], g["max"])
            _grouped(ax, groups, vals, err)
            ax.axvspan(len(CLIENTS) - 0.5, len(CLIENTS) + 0.5, color="#fff4e0", zorder=0, linewidth=0)
            ax.set_ylim(min(np.min(e[0]) for e in err.values()) / 3, max(np.max(e[1]) for e in err.values()) * 30)
            ax.set_ylabel(unit)
            _panel(ax, letter, title)
        _legend(fig, 0.985)
        fig.text(0.5, 0.0, f"Bars: mean over the rounds; whiskers: min to max. Global Server: all clients' data "
                 f"together; its time is the slowest client's. Link model: RTT {a.rtt_ms:g} ms, uplink "
                 f"{a.uplink_mbps:g} Mbit/s, downlink {a.downlink_mbps:g} Mbit/s; round trips per round: "
                 f"v12 {PROJECTS[0].round_trips}, FL {PROJECTS[1].round_trips}.",
                 ha="center", va="top", fontsize=7.0, style="italic", wrap=True)
        fig.tight_layout(rect=(0, 0.02, 1, 0.955))
        _save(fig, path)


def fig_metrics(runs: list[Run], table: pd.DataFrame, path: Path, title: str, with_range: bool) -> None:
    """(a) Global Server, (b)-(e) each client: the seven metrics; (f) difference FL - v12."""
    order = [GLOBAL, *CLIENTS]
    a, b = runs
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 2, figsize=(DOUBLE_COLUMN, 7.4))
        axes = axes.ravel()
        lowest = float(table[[c for c in table.columns if c.endswith("min")]
                             if with_range else [a.project.key, b.project.key]].min().min())
        # Room under the lowest value for the labels of negative bars.
        y_low = float(np.floor((lowest - 0.3) * 10) / 10) if lowest < 0 else -0.05
        for ax, letter, e in zip(axes, "abcde", order):
            t = table[table["Entity"] == e].set_index("Metric").loc[METRICS]
            vals = {p.key: t[p.key] for p in PROJECTS}
            err = {p.key: (t[f"{p.key} min"], t[f"{p.key} max"]) for p in PROJECTS} if with_range else None
            _grouped(ax, METRICS, vals, err, log=False, label_fmt=lambda v: f"{v:.3f}", fontsize=5.4)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xticks(range(len(METRICS)), [m.replace("Cohen's Kappa", "Cohen's κ") for m in METRICS],
                          fontsize=7, rotation=30, ha="right", rotation_mode="anchor")
            ax.set_ylim(y_low, 1.32)
            ax.set_yticks(np.arange(np.ceil(max(y_low, -1.0) * 5) / 5, 1.01, 0.2).round(1))
            if letter in "ace":
                ax.set_ylabel(title)
            n = f" (validation: v12 {a.val_sizes.get(e, '?')}, FL {b.val_sizes.get(e, '?')})"
            _panel(ax, letter, ("Global Server (pooled)" if e == GLOBAL else e) + n)
            ax.title.set_fontsize(8.2)

        ax = axes[5]
        diff = np.array([[table[(table["Entity"] == e) & (table["Metric"] == m)][f"{b.project.key} - {a.project.key}"]
                          .iloc[0] for m in METRICS] for e in order])
        lim = max(0.05, float(np.abs(diff).max()))
        im = ax.imshow(diff, cmap="PuOr", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(diff.shape[0]):
            for k in range(diff.shape[1]):
                ax.text(k, i, f"{diff[i, k]:+.3f}", ha="center", va="center", fontsize=6.3,
                        color="white" if abs(diff[i, k]) > 0.6 * lim else "black")
        ax.set_xticks(range(len(METRICS)), [m.replace("Cohen's Kappa", "Cohen's κ") for m in METRICS], fontsize=7,
                      rotation=30, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(order)), ["Global Server" if e == GLOBAL else short(e) for e in order], fontsize=7)
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cbar.ax.tick_params(labelsize=6.5)
        cbar.set_label("FL − v12", fontsize=7)
        _panel(ax, "f", "Difference (FL − v12)")

        _legend(fig, 0.975)
        note = ("Bars: mean over the 30 rounds; whiskers: min to max over the rounds; numbers: mean."
                if with_range else "Bars: value in the last round (30); numbers: value.")
        fig.text(0.5, 0.0, note + " Global model of each project on each client's validation patients; the two "
                 "projects use different validation sets (see the sizes in the titles).",
                 ha="center", va="top", fontsize=7.0, style="italic")
        fig.tight_layout(rect=(0, 0.01, 1, 0.955), h_pad=1.6)
        _save(fig, path)


# --- main ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="folder holding both project folders")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--rtt-ms", type=float, default=50.0, help="network round-trip time (ms)")
    ap.add_argument("--uplink-mbps", type=float, default=20.0, help="client -> server bandwidth (Mbit/s)")
    ap.add_argument("--downlink-mbps", type=float, default=50.0, help="server -> client bandwidth (Mbit/s)")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    figs = a.out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    fmt = lambda v: f"{v:.4f}"  # noqa: E731

    runs = [load_v12(a.root, PROJECTS[0]), load_fl(a.root, PROJECTS[1])]
    print("Projects:")
    for r in runs:
        print(f"  {r.project.key}: {r.project.folder} | {r.rounds} rounds | validation patients "
              + ", ".join(f"{short(c)} {r.val_sizes.get(c, '?')}" for c in CLIENTS))
    if runs[0].rounds != runs[1].rounds:
        print(f"  note: different numbers of rounds ({runs[0].rounds} vs {runs[1].rounds})")
    ok, lines = same_zkp(a.root)
    print("ZKP (read from each project's zkp.py): " + ("the same proposed Pedersen ZKP in both projects"
                                                       if ok else "WARNING: the projects use different ZKPs"))
    print("\n".join(lines))

    # Computation
    comp = computation_table(runs)
    comp.to_csv(a.out / "computation_summary.csv", index=False)
    pd.concat([r.computation.assign(Project=r.project.key) for r in runs]).to_csv(
        a.out / "computation_per_round.csv", index=False)
    print("\n=== Computation per round, averaged over the rounds (s): each client and the Global Server ===")
    wide = comp.pivot(index="Entity", columns="Project", values="Total mean (s)").loc[ENTITIES]
    wide["FL / v12"] = wide["FL"] / wide["v12"]
    rng = comp.pivot(index="Entity", columns="Project", values=["Total min (s)", "Total max (s)"]).loc[ENTITIES]
    for p in ("v12", "FL"):
        wide[f"{p} min"] = rng[("Total min (s)", p)]
        wide[f"{p} max"] = rng[("Total max (s)", p)]
    print(wide[["v12", "v12 min", "v12 max", "FL", "FL min", "FL max", "FL / v12"]].to_string(float_format=fmt))
    print("\nComputation by category (mean per round, s):")
    for side, parts, entities in (("Client side (mean over the 4 clients)", CLIENT_PARTS, CLIENTS),
                                  ("Server side", SERVER_PARTS, [GLOBAL])):
        rows = {}
        for r in runs:
            d = r.computation[r.computation["Entity"].isin(entities)]
            rows[r.project.key] = [d[p].mean() for p in parts]
        t = pd.DataFrame(rows, index=parts)
        print(f"{side}:")
        print(t.to_string(float_format=fmt))
    latex_table(comp[["Project", "Entity", "Total mean (s)", "Total min (s)", "Total max (s)",
                      "Total over the run (s)"]],
                a.out / "computation_summary.tex", "Computation per round (s) of FedRF-Diff v12 and "
                "Federated-Learning, per client and Global Server, averaged over 30 rounds.", "tab:v12_fl_comp")

    # Communication
    comm, per_round = communication_table(runs, a)
    comm.to_csv(a.out / "communication_summary.csv", index=False)
    pd.concat([d.assign(Project=k) for k, d in per_round.items()]).to_csv(
        a.out / "communication_per_round.csv", index=False)
    print(f"\n=== Communication per round, averaged over the rounds (RTT {a.rtt_ms:g} ms, up {a.uplink_mbps:g} "
          f"Mbit/s, down {a.downlink_mbps:g} Mbit/s; round trips v12 {PROJECTS[0].round_trips}, "
          f"FL {PROJECTS[1].round_trips}) ===")
    shown = comm.set_index(["Entity", "Project"]).loc[[(e, p.key) for e in ENTITIES for p in PROJECTS]]
    print(shown[["Sent mean (KB)", "Received mean (KB)", "Total mean (KB)", "Time mean (ms)", "Time min (ms)",
                 "Time max (ms)", "Data over the run (MB)"]].to_string(float_format=fmt))
    latex_table(comm[["Project", "Entity", "Sent mean (KB)", "Received mean (KB)", "Time mean (ms)",
                      "Data over the run (MB)"]],
                a.out / "communication_summary.tex", "Communication per round of FedRF-Diff v12 and "
                "Federated-Learning, per client and Global Server, averaged over 30 rounds.", "tab:v12_fl_comm")

    # Metrics
    avg, last = metric_tables(runs)
    avg.to_csv(a.out / "metrics_averaged.csv", index=False)
    last.to_csv(a.out / "metrics_last_round.csv", index=False)
    for title, t in ((f"Validation metrics averaged over the rounds (v12: {runs[0].rounds}, FL: {runs[1].rounds})",
                      avg), (f"Validation metrics of the last round (v12: round {runs[0].rounds}, "
                             f"FL: round {runs[1].rounds})", last)):
        print(f"\n=== {title} ===")
        print(t.set_index(["Entity", "Metric"])[["v12", "FL", "FL - v12"]].to_string(float_format=fmt))
    for name, t, cap in (("metrics_averaged", avg, "averaged over 30 rounds"), ("metrics_last_round", last,
                                                                                "of the last round")):
        latex_table(t[["Entity", "Metric", "v12", "FL", "FL - v12"]], a.out / f"{name}.tex",
                    f"Validation metrics of the global model {cap}: FedRF-Diff v12 vs Federated-Learning.",
                    f"tab:v12_fl_{name}")

    print("\nFigures (600-dpi PNG + vector PDF):")
    fig_computation(runs, figs / "fig1_computation_comparison")
    fig_communication(runs, per_round, figs / "fig2_communication_comparison", a)
    fig_metrics(runs, avg, figs / "fig3_averaged_metrics", "Mean over the rounds", with_range=True)
    fig_metrics(runs, last, figs / "fig4_last_round_metrics", "Last round", with_range=False)
    print(f"\nTables in {a.out.resolve()}")


if __name__ == "__main__":
    main()
