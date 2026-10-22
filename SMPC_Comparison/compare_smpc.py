"""Compare the two SMPC schemes of FedRF-Diff: v14 (additive secret sharing) vs v15 (proposed).

    v14  Fed_Code_v14_ProposedPedersenZKP_SMPC_NoDP
         n-out-of-n additive secret sharing mod 2^127 - 1: every client splits its
         vector into N shares, sends N - 1 of them to the other clients, adds the
         shares it holds and sends the partial sum to the server.
    v15  Fed_Code_v15_ProposedPedersenZKP_ProposedSMPC_NoDP
         Adaptive Prime Secure Aggregation with Self-Randomized Pairwise Masking:
         one-time Diffie-Hellman pairwise keys, a prime pool from a public seed, an
         adaptive prime p_t per round, quantization with S_q, pairwise PRG masks that
         cancel in the sum; every client sends the server one masked vector.

Both protect the same values (each client's weighted validation loss and
TP/FP/TN/FN counts; the server learns only their totals) in the same place of
the same pipeline (proposed ZKP, no DP), so the learning is identical and the
comparison isolates the SMPC.

Compared:

1. SMPC times in the saved runs: min, max, mean per client (per-round work;
   v15's one-time key setup separately) and for the Global Server.
2. SMPC communication: bytes, messages and round trips per client and round,
   and a modelled time
       time = round trips x RTT + 8 x bytes_up / uplink + 8 x bytes_down / downlink
   (v14: 2 round trips - shares to and from the other clients, partial sum to the
   server; v15: 1 - masked vector up; plus one key-exchange round trip once).
3. Computation: each run's total per round and the SMPC share of it.
4. Same-machine benchmark of the two smpc.py modules (loaded from the project
   folders) for N = 2 ... 32 clients: per-client and server time, bytes per
   client, one-time setup - fair timings and the scaling with N.
5. Metrics: the averaged and last-round metrics of both runs (identical up to
   quantization) and the per-round L_avg difference.

Outputs (--out, default SMPC_Comparison/outputs): CSV and LaTeX tables, four
figures (600-dpi PNG + vector PDF) and the terminal log.

Usage:
    python compare_smpc.py
    python compare_smpc.py --benchmark-rounds 50 --clients 2 4 8 16 32
    python compare_smpc.py --rtt-ms 20 --uplink-mbps 50 --downlink-mbps 100
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
import types
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
METRICS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "specificity": "Specificity",
           "f1": "F1 Score", "mcc": "MCC", "kappa": "Cohen's Kappa"}


@dataclass(frozen=True)
class Scheme:
    key: str
    label: str
    tick: str
    folder: str
    round_trips: int  # SMPC message exchanges per round between a client and the others/server
    color: str  # Okabe-Ito
    hatch: str
    marker: str


SCHEMES = (
    Scheme("v14", "Additive secret sharing (v14)", "Additive\nsharing (v14)",
           "Fed_Code_v14_ProposedPedersenZKP_SMPC_NoDP", 2, "#0072B2", "", "o"),
    Scheme("v15", "Proposed adaptive-prime pairwise masking (v15)", "Proposed\nmasking (v15)",
           "Fed_Code_v15_ProposedPedersenZKP_ProposedSMPC_NoDP", 1, "#D55E00", "////", "s"),
)

DOUBLE_COLUMN = 7.16
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
    return name.split("(", 1)[1].rstrip(")") if "(" in name else name


# --- loading ---------------------------------------------------------------------
@dataclass
class Run:
    scheme: Scheme
    smpc: pd.DataFrame  # harmonised: round, client, per_round_s, setup_s, server_s, bytes_up, bytes_down
    costs: pd.DataFrame
    history: pd.DataFrame
    final: pd.DataFrame
    config: dict

    @property
    def clients(self) -> list[str]:
        return [c for c in dict.fromkeys(self.smpc["client"]) if c != GLOBAL]


def load_run(root: Path, s: Scheme) -> Run:
    d = root / s.folder / "outputs" / "run"
    need = ["smpc_times.csv", "costs.csv", "history.csv", "final_metrics.csv", "config.json"]
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        sys.exit(f"{d}: missing {', '.join(missing)}; run `python -m fedrf_diff` in {s.folder} first")
    raw = pd.read_csv(d / "smpc_times.csv")
    h = pd.DataFrame({"round": raw["round"], "client": raw["client"],
                      "bytes_up": raw["bytes_up"], "bytes_down": raw["bytes_down"]})
    server = raw["client"] == GLOBAL
    if s.key == "v14":
        h["per_round_s"] = np.where(server, np.nan, raw["share_s"] + raw["sum_s"])
        h["setup_s"] = 0.0
        h["server_s"] = np.where(server, raw["reconstruct_s"], np.nan)
        h["messages_sent"] = np.where(server, 0, raw["shares_sent"] + 1)  # shares to peers + partial sum
        h["messages_received"] = np.where(server, 0, raw["shares_sent"])
    else:
        h["per_round_s"] = np.where(server, np.nan, raw["mask_s"] + raw["recovery_s"])
        h["setup_s"] = raw["key_setup_s"].fillna(0.0)
        h["server_s"] = np.where(server, raw["aggregate_s"], np.nan)
        first = raw["round"] == raw["round"].min()
        h["messages_sent"] = np.where(server, 0, 1 + first.astype(int))  # c_i (+ A_i once)
        n = raw.loc[~server, "client"].nunique()
        h["messages_received"] = np.where(server, 0, np.where(first, n - 1, 0))  # other A_j once
        h["prime_index"] = raw["prime_index"]
        h["prime"] = raw["prime"].astype(str)
    return Run(s, h, pd.read_csv(d / "costs.csv"), pd.read_csv(d / "history.csv"),
               pd.read_csv(d / "final_metrics.csv"), json.loads((d / "config.json").read_text()))


# --- tables ----------------------------------------------------------------------
def time_summary(runs: list[Run]) -> pd.DataFrame:
    rows = []
    for r in runs:
        for c in r.clients:
            g = r.smpc[r.smpc["client"] == c]
            ms = g["per_round_s"] * MS
            rows.append({"scheme": r.scheme.key, "entity": c, "what": "per-round SMPC work",
                         "min_ms": ms.min(), "max_ms": ms.max(), "mean_ms": ms.mean(),
                         "one_time_setup_ms": g["setup_s"].sum() * MS})
        g = r.smpc[r.smpc["client"] == GLOBAL]
        ms = g["server_s"] * MS
        rows.append({"scheme": r.scheme.key, "entity": GLOBAL,
                     "what": "reconstruction" if r.scheme.key == "v14" else "aggregation + dequantization",
                     "min_ms": ms.min(), "max_ms": ms.max(), "mean_ms": ms.mean(), "one_time_setup_ms": 0.0})
    return pd.DataFrame(rows)


def link_time(up: float, down: float, rt: int, a: argparse.Namespace) -> float:
    return rt * a.rtt_ms / MS + 8 * up / (a.uplink_mbps * 1e6) + 8 * down / (a.downlink_mbps * 1e6)


def communication_summary(runs: list[Run], a: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for r in runs:
        c = r.smpc[r.smpc["client"] != GLOBAL]
        first = c["round"] == c["round"].min()
        steady = c[~first] if (~first).any() else c
        setup_up = (c.loc[first, "bytes_up"].mean() - steady["bytes_up"].mean())
        setup_down = (c.loc[first, "bytes_down"].mean() - steady["bytes_down"].mean())
        up, down = steady["bytes_up"].mean(), steady["bytes_down"].mean()
        rows.append({
            "scheme": r.scheme.key, "label": r.scheme.label,
            "bytes_up_per_client_round": up, "bytes_down_per_client_round": down,
            "messages_sent_per_client_round": steady["messages_sent"].mean(),
            "messages_received_per_client_round": steady["messages_received"].mean(),
            "client_to_client_messages_per_round": (len(r.clients) * (len(r.clients) - 1)
                                                    if r.scheme.key == "v14" else 0),
            "round_trips_per_round": r.scheme.round_trips,
            "comm_ms_per_client_round": link_time(up, down, r.scheme.round_trips, a) * MS,
            "one_time_setup_bytes_up": max(setup_up, 0.0), "one_time_setup_bytes_down": max(setup_down, 0.0),
            "one_time_setup_comm_ms": (link_time(setup_up, setup_down, 1, a) * MS if r.scheme.key == "v15" else 0.0),
            "server_bytes_received_per_round": r.smpc.loc[r.smpc["client"] == GLOBAL, "bytes_down"].mean(),
            "smpc_bytes_total_run": int(c["bytes_up"].sum() + c["bytes_down"].sum()),
        })
    return pd.DataFrame(rows)


def computation_summary(runs: list[Run]) -> pd.DataFrame:
    rows = []
    for r in runs:
        c = r.costs
        per_round = c.groupby("round")["computation_s"].sum()
        smpc = c.groupby("round")["smpc_s"].sum()
        rows.append({"scheme": r.scheme.key, "rounds": int(c["round"].max()),
                     "total_computation_per_round_s": per_round.mean(),
                     "smpc_per_round_ms (all clients + server)": smpc.mean() * MS,
                     "smpc_share_of_computation_%": 100 * smpc.sum() / per_round.sum()})
    return pd.DataFrame(rows)


def metric_tables(runs: list[Run]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    a, b = runs
    clients = a.clients
    avg, last = [], []
    for e in [*clients, GLOBAL]:
        key = "all" if e == GLOBAL else e
        for m, name in METRICS.items():
            col = f"{m}[{key}]"
            avg.append({"entity": e, "metric": name, "v14": a.history[col].mean(), "v15": b.history[col].mean()})
            last.append({"entity": e, "metric": name, "v14": a.history[col].iloc[-1], "v15": b.history[col].iloc[-1]})
    avg, last = pd.DataFrame(avg), pd.DataFrame(last)
    for t in (avg, last):
        t["v15 - v14"] = t["v15"] - t["v14"]
    n = min(len(a.history), len(b.history))
    dl = float(np.max(np.abs(a.history["L_avg"].to_numpy()[:n] - b.history["L_avg"].to_numpy()[:n])))
    return avg, last, dl


def properties_table() -> pd.DataFrame:
    """The design differences, side by side (for the paper)."""
    rows = [
        ("Primitive", "n-out-of-n additive secret sharing", "self-randomized pairwise masking (PRG streams)"),
        ("Modulus", "fixed prime 2^127 - 1", "adaptive prime p_t = Pool[H(seed||t) mod 3], 61-bit pool"),
        ("Encoding", "fixed point 2^32 (reals), integers as is", "quantization S_q = 10^6"),
        ("One-time setup", "none", "pairwise Diffie-Hellman keys (2048-bit ZKP group)"),
        ("Client-to-client messages per round", "N - 1 shares sent, N - 1 received", "none"),
        ("Client-to-server messages per round", "1 partial sum", "1 masked vector"),
        ("Round trips per round", "2", "1"),
        ("Bytes per client and round (6 values, N = 4)", "384 up, 288 down", "48 up, 0 down"),
        ("Server learns", "only the sum", "only the sum"),
        ("Collusion threshold", "server + up to N - 2 clients learn nothing more",
         "server + up to N - 2 clients learn nothing more"),
        ("Security", "information-theoretic (uniform shares)", "computational (PRG, Diffie-Hellman)"),
        ("Drop-out during an aggregation", "not tolerated (all shares needed)",
         "recovery step: survivors reveal their pairwise terms with the dropped client"),
    ]
    return pd.DataFrame(rows, columns=["Property", SCHEMES[0].label, SCHEMES[1].label])


def latex(df: pd.DataFrame, path: Path, caption: str, label: str, digits: int = 4) -> None:
    cols = list(df.columns)
    esc = lambda x: str(x).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("^", r"\^{}")  # noqa: E731
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "".join("l" if df[c].dtype == object else "r" for c in cols) + "}",
             r"\toprule", " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, str):
                cells.append(esc(v))
            elif pd.isna(v):
                cells.append("--")
            else:
                cells.append(f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{digits}g}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --- same-machine benchmark ----------------------------------------------------------
def load_smpc_modules(root: Path) -> dict:
    """Import each project's fedrf_diff/smpc.py under its own package name."""
    mods = {}
    for s in SCHEMES:
        pkg_name = f"smpc_bench_{s.key}"
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(root / s.folder / "fedrf_diff")]  # lets "from . import zkp" resolve
        sys.modules[pkg_name] = pkg
        mods[s.key] = importlib.import_module(f"{pkg_name}.smpc")
    return mods


def benchmark(root: Path, ns: list[int], rounds: int, seed: int = 42) -> pd.DataFrame:
    """Per-client and server SMPC time and bytes for N clients, both schemes, same process."""
    mods = load_smpc_modules(root)
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        names = [f"Client {i + 1}" for i in range(n)]
        t0 = time.perf_counter()
        aggregator = mods["v15"].SecureAggregator(names)
        setup_wall = time.perf_counter() - t0
        setup_per_client = float(np.mean(list(aggregator.setup_s.values())))
        for t in range(1, rounds + 1):
            # Values of the size seen in the runs: n_c L_c ~ 50-200, counts 0-60.
            vectors = {c: [float(rng.uniform(20, 200)), float(rng.uniform(5, 60)),
                           *[int(v) for v in rng.integers(0, 60, 4)]] for c in names}
            order = ("v14", "v15") if t % 2 else ("v15", "v14")  # alternate to share machine conditions
            for key in order:
                if key == "v14":
                    out = mods["v14"].secure_aggregate(vectors)
                    per_client = [p.total_s for p in out.clients.values()]
                    server_s = out.reconstruct_s
                    up = next(iter(out.clients.values())).bytes_up
                    down = next(iter(out.clients.values())).bytes_down
                    setup = 0.0
                else:
                    out = aggregator.aggregate(t, vectors)
                    per_client = [p.total_s for p in out.clients.values()]
                    server_s = out.aggregate_s
                    up = next(iter(out.clients.values())).bytes_up
                    down = next(iter(out.clients.values())).bytes_down
                    setup = setup_per_client
                rows.append({"scheme": key, "clients": n, "round": t,
                             "client_ms": float(np.mean(per_client)) * MS, "server_ms": server_s * MS,
                             "bytes_up": up, "bytes_down": down, "setup_ms_per_client": setup * MS,
                             "setup_wall_ms": setup_wall * MS if key == "v15" else 0.0})
        print(f"  benchmark: N = {n} done", flush=True)
    return pd.DataFrame(rows)


def benchmark_summary(bench: pd.DataFrame) -> pd.DataFrame:
    g = bench.groupby(["scheme", "clients"])
    out = g.agg(client_ms_min=("client_ms", "min"), client_ms_max=("client_ms", "max"),
                client_ms_mean=("client_ms", "mean"), server_ms_mean=("server_ms", "mean"),
                bytes_up=("bytes_up", "mean"), bytes_down=("bytes_down", "mean"),
                setup_ms_per_client=("setup_ms_per_client", "mean")).reset_index()
    return out


# --- figures -------------------------------------------------------------------------
def _panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold")


def _save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {path.with_suffix('.png')} (+ .pdf)")


def _legend(fig, y: float, lines: bool = False) -> None:
    if lines:
        handles = [plt.Line2D([], [], color=s.color, marker=s.marker, label=s.label) for s in SCHEMES]
    else:
        handles = [Patch(facecolor=s.color, edgecolor="black", hatch=s.hatch, linewidth=0.4, label=s.label)
                   for s in SCHEMES]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, y))


def _label(v: float) -> str:
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}" if v >= 0.1 else f"{v:.3f}"


def _bars(ax, groups, values: dict, err: dict | None = None, log: bool = False, fmt=_label) -> None:
    x = np.arange(len(groups))
    w = 0.8 / len(SCHEMES)
    for j, s in enumerate(SCHEMES):
        v = np.asarray(values[s.key], dtype=float)
        pos = x + (j - (len(SCHEMES) - 1) / 2) * w
        kw, top = {}, v
        if err is not None:
            lo, hi = (np.asarray(e, dtype=float) for e in err[s.key])
            kw = dict(yerr=[np.maximum(v - lo, 0), np.maximum(hi - v, 0)], capsize=1.5,
                      error_kw={"elinewidth": 0.6, "capthick": 0.6, "ecolor": "#333333"})
            top = np.maximum(hi, v)
        ax.bar(pos, v, w, color=s.color, hatch=s.hatch, edgecolor="black", linewidth=0.4, zorder=2, **kw)
        for p, val, t in zip(pos, v, top):
            if val > 0:
                ax.text(p, t * (1.25 if log else 1.0) + (0 if log else 0.0), " " + fmt(val), ha="center",
                        va="bottom", fontsize=5.8, rotation=90)
    ax.set_xticks(x, groups)
    if log:
        ax.set_yscale("log")


def fig_times(runs: list[Run], ts: pd.DataFrame, path: Path) -> None:
    """(a) per-round SMPC work per client and the server; (b) one-time setup per client; (c) per round."""
    clients = runs[0].clients
    groups = [short(c) for c in clients] + ["Global\nServer"]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(DOUBLE_COLUMN, 5.4))
        grid = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0], hspace=0.5, wspace=0.28)
        ax_a, ax_b, ax_c = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
        vals, err = {}, {}
        for s in SCHEMES:
            t = ts[ts["scheme"] == s.key].set_index("entity").loc[[*clients, GLOBAL]]
            vals[s.key], err[s.key] = t["mean_ms"], (t["min_ms"], t["max_ms"])
        _bars(ax_a, groups, vals, err, log=True)
        ax_a.axvspan(len(clients) - 0.5, len(clients) + 0.5, color="#fff4e0", zorder=0, linewidth=0)
        ax_a.set_ylim(min(np.min(e[0]) for e in err.values()) / 3, max(np.max(e[1]) for e in err.values()) * 25)
        ax_a.set_ylabel("Time per round (ms, log)")
        _panel(ax_a, "a", "SMPC time per round: clients (share/sum vs mask) and server (reconstruct/aggregate)")

        for r in runs:
            s = r.scheme
            g = r.smpc[r.smpc["client"] != GLOBAL].groupby("round")["per_round_s"].mean() * MS
            ax_b.plot(g.index, g.values, color=s.color, marker=s.marker, label=s.label)
        ax_b.set_xlabel("Round")
        ax_b.set_ylabel("Mean client time (ms)")
        ax_b.set_ylim(bottom=0)
        _panel(ax_b, "b", "Per round (mean over clients)")

        setup = [ts[(ts["scheme"] == s.key) & (ts["entity"] != GLOBAL)]["one_time_setup_ms"].mean() for s in SCHEMES]
        for i, (s, v) in enumerate(zip(SCHEMES, setup)):
            ax_c.bar(i, v, 0.55, color=s.color, hatch=s.hatch, edgecolor="black", linewidth=0.4, zorder=2)
            ax_c.text(i, v + 0.05, "none" if v == 0 else f"{v:.2f} ms", ha="center", va="bottom", fontsize=7)
        ax_c.set_xticks(range(len(SCHEMES)), [s.tick for s in SCHEMES], fontsize=7)
        ax_c.set_ylabel("Per client (ms)")
        ax_c.set_ylim(0, max(setup) * 1.35 + 0.1)
        _panel(ax_c, "c", "One-time key setup")
        _legend(fig, 0.955)
        fig.text(0.5, 0.0, "(a) Bars: mean over the rounds; whiskers: min to max. v15's one-time Diffie-Hellman key "
                 "setup is excluded from (a), (b) and shown in (c).", ha="center", va="top", fontsize=7.2,
                 style="italic")
        _save(fig, path)


def fig_communication(comm: pd.DataFrame, path: Path, a: argparse.Namespace) -> None:
    c = comm.set_index("scheme")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.6))
        x = np.arange(len(SCHEMES))
        ticks = [s.tick for s in SCHEMES]
        ax = axes[0]
        w = 0.38
        for j, (col, lab, fill) in enumerate((("bytes_up_per_client_round", "Up", True),
                                               ("bytes_down_per_client_round", "Down", False))):
            vals = [c.loc[s.key, col] for s in SCHEMES]
            for i, (s, v) in enumerate(zip(SCHEMES, vals)):
                ax.bar(i + (j - 0.5) * w, v, w, zorder=2, linewidth=0.6,
                       color=s.color if fill else "white", edgecolor="black" if fill else s.color,
                       hatch=(s.hatch or None) if fill else "xxxx")
                ax.text(i + (j - 0.5) * w, v + 6, f"{v:.0f}", ha="center", va="bottom", fontsize=6.8)
        ax.set_xticks(x, ticks, fontsize=7)
        ax.set_ylabel("Bytes per client per round")
        ax.set_ylim(0, max(c["bytes_up_per_client_round"].max(), c["bytes_down_per_client_round"].max()) * 1.3)
        ax.legend(handles=[Patch(facecolor="#888888", edgecolor="black", label="Up"),
                           Patch(facecolor="white", edgecolor="#555555", hatch="xxxx", label="Down")],
                  loc="upper right", fontsize=6.8, handlelength=1.2)
        _panel(ax, "a", "SMPC bytes")

        ax = axes[1]
        for i, s in enumerate(SCHEMES):
            sent, recv = c.loc[s.key, "messages_sent_per_client_round"], c.loc[s.key, "messages_received_per_client_round"]
            ax.bar(i - 0.19, sent, 0.38, color=s.color, hatch=s.hatch or None, edgecolor="black", linewidth=0.4, zorder=2)
            ax.bar(i + 0.19, recv, 0.38, color="white", edgecolor=s.color, hatch="xxxx", linewidth=0.6, zorder=2)
            ax.text(i - 0.19, sent + 0.05, f"{sent:.0f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + 0.19, recv + 0.05, f"{recv:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, ticks, fontsize=7)
        ax.set_ylabel("Messages per client per round")
        ax.set_ylim(0, 5.5)
        ax.legend(handles=[Patch(facecolor="#888888", edgecolor="black", label="Sent"),
                           Patch(facecolor="white", edgecolor="#555555", hatch="xxxx", label="Received")],
                  loc="upper right", fontsize=6.8, handlelength=1.2)
        _panel(ax, "b", "SMPC messages")

        ax = axes[2]
        vals = [c.loc[s.key, "comm_ms_per_client_round"] for s in SCHEMES]
        lat = [s.round_trips * a.rtt_ms for s in SCHEMES]
        for i, (s, v, l_) in enumerate(zip(SCHEMES, vals, lat)):
            ax.bar(i, l_, 0.55, color=s.color, alpha=0.45, edgecolor="black", linewidth=0.4, zorder=2)
            ax.bar(i, v - l_, 0.55, bottom=l_, color=s.color, hatch="////", edgecolor="black", linewidth=0.4,
                   zorder=2)
            ax.text(i, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, ticks, fontsize=7)
        ax.set_ylabel("ms per client per round")
        ax.set_ylim(0, max(vals) * 1.3)
        ax.legend(handles=[Patch(facecolor="#bbbbbb", edgecolor="black", label="Latency"),
                           Patch(facecolor="white", edgecolor="black", hatch="////", label="Transmission")],
                  loc="upper right", fontsize=6.8, handlelength=1.2)
        _panel(ax, "c", "SMPC communication time")
        fig.text(0.5, -0.05, f"Steady state (after v15's one-time key exchange: {c.loc['v15', 'one_time_setup_bytes_up']:.0f} B up, "
                 f"{c.loc['v15', 'one_time_setup_bytes_down']:.0f} B down per client). Link model: RTT {a.rtt_ms:g} ms, "
                 f"uplink {a.uplink_mbps:g} Mbit/s, downlink {a.downlink_mbps:g} Mbit/s.",
                 ha="center", fontsize=7.2, style="italic")
        fig.tight_layout()
        _save(fig, path)


def fig_scaling(bench_sum: pd.DataFrame, path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.6))
        for s in SCHEMES:
            b = bench_sum[bench_sum["scheme"] == s.key]
            axes[0].plot(b["clients"], b["client_ms_mean"], color=s.color, marker=s.marker, label=s.label)
            axes[0].fill_between(b["clients"], b["client_ms_min"], b["client_ms_max"], color=s.color, alpha=0.15,
                                 linewidth=0)
            axes[1].plot(b["clients"], b["bytes_up"] + b["bytes_down"], color=s.color, marker=s.marker)
            axes[2].plot(b["clients"], b["server_ms_mean"], color=s.color, marker=s.marker)
        v15 = bench_sum[bench_sum["scheme"] == "v15"]
        axes[0].plot(v15["clients"], v15["setup_ms_per_client"], color=SCHEMES[1].color, linestyle=":",
                     marker="x", label="v15 one-time key setup")
        for ax, letter, title, ylabel in ((axes[0], "a", "Client time", "ms per client (log)"),
                                          (axes[1], "b", "Client bytes (up + down)", "Bytes per client per round (log)"),
                                          (axes[2], "c", "Server time", "ms per round (log)")):
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(sorted(bench_sum["clients"].unique()), [str(n) for n in sorted(bench_sum["clients"].unique())])
            ax.set_xlabel("Clients N")
            ax.set_ylabel(ylabel)
            _panel(ax, letter, title)
        setup_line = [h for h in axes[0].get_lines() if h.get_label() == "v15 one-time key setup"]
        axes[0].legend(handles=setup_line, loc="lower right", fontsize=6.2)
        _legend(fig, 0.96, lines=True)
        fig.text(0.5, -0.06, "Same-machine benchmark of both smpc.py modules (6 values per client); line: mean over "
                 "the rounds, band: min to max. v14's per-client bytes grow with N (N - 1 shares out and in); v15's "
                 "stay at one masked vector.", ha="center", fontsize=7.0, style="italic", wrap=True)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        _save(fig, path)


def fig_metrics(runs: list[Run], avg: pd.DataFrame, last: pd.DataFrame, path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.6), gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})
        names = list(METRICS.values())
        for ax, letter, table, title in ((axes[0], "a", avg, "Global Server, mean over the rounds"),
                                         (axes[1], "b", last, "Global Server, last round")):
            t = table[table["entity"] == GLOBAL].set_index("metric").loc[names]
            short_names = {"Accuracy": "Acc.", "Precision": "Prec.", "Recall": "Rec.", "Specificity": "Spec.",
                           "F1 Score": "F1", "MCC": "MCC", "Cohen's Kappa": "κ"}
            _bars(ax, [short_names[n] for n in names], {s.key: t[s.key] for s in SCHEMES}, fmt=lambda v: f"{v:.3f}")
            ax.set_ylim(0, 1.25)
            ax.tick_params(axis="x", labelsize=7.5)
            ax.set_ylabel("Value")
            _panel(ax, letter, title)
        ax = axes[2]
        a, b = runs
        n = min(len(a.history), len(b.history))
        diff = np.abs(a.history["L_avg"].to_numpy()[:n] - b.history["L_avg"].to_numpy()[:n])
        ax.semilogy(np.arange(1, n + 1), np.maximum(diff, 1e-16), color="#333333", marker="o")
        ax.axhline(0.5 / 10**6, color="#D55E00", linestyle="--", linewidth=1.0, label="quantization step / 2")
        ax.set_xlabel("Round")
        ax.set_ylabel("|L_avg(v15) - L_avg(v14)|")
        ax.legend(loc="upper right", fontsize=6.5)
        _panel(ax, "c", "Same learning")
        _legend(fig, 0.96)
        fig.text(0.5, -0.05, "Both runs learn the same model (same seed, proposed ZKP, no DP); the totals differ only "
                 "by the fixed-point / quantization rounding of the two SMPC schemes.", ha="center", fontsize=7.2,
                 style="italic")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        _save(fig, path)


# --- main ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--rtt-ms", type=float, default=50.0)
    ap.add_argument("--uplink-mbps", type=float, default=20.0)
    ap.add_argument("--downlink-mbps", type=float, default=50.0)
    ap.add_argument("--benchmark-rounds", type=int, default=50)
    ap.add_argument("--clients", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--no-benchmark", action="store_true")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    figs = a.out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    fmt = lambda v: f"{v:.4f}"  # noqa: E731

    runs = [load_run(a.root, s) for s in SCHEMES]
    print("Runs:")
    for r in runs:
        print(f"  {r.scheme.key}: {r.scheme.folder} | {r.history['round'].max()} rounds | "
              f"SMPC: {r.config.get('smpc', {}).get('scheme', '?')}")
    same = all(runs[0].config.get(k) == runs[1].config.get(k)
               for k in ("seed", "n_trees", "n_landmarks", "d_z", "max_rounds", "eps_z", "eps_l", "patience"))
    print("Same learning settings and ZKP; the runs differ only in the SMPC." if same
          else "WARNING: the learning settings differ")

    ts = time_summary(runs)
    comm = communication_summary(runs, a)
    comp = computation_summary(runs)
    avg, last, dl = metric_tables(runs)
    props = properties_table()
    ts.to_csv(a.out / "smpc_time_summary.csv", index=False)
    comm.to_csv(a.out / "smpc_communication_summary.csv", index=False)
    comp.to_csv(a.out / "computation_summary.csv", index=False)
    avg.to_csv(a.out / "metrics_averaged.csv", index=False)
    last.to_csv(a.out / "metrics_last_round.csv", index=False)
    props.to_csv(a.out / "smpc_properties.csv", index=False)

    print("\n=== SMPC times in the runs (ms): min / max / mean per round ===")
    shown = ts.pivot(index="entity", columns="scheme", values=["min_ms", "max_ms", "mean_ms"])
    shown = shown.reindex([*runs[0].clients, GLOBAL])
    shown.columns = [f"{sch} {stat.replace('_ms', '')}" for stat, sch in shown.columns]
    print(shown[[f"{s} {k}" for s in ("v14", "v15") for k in ("min", "max", "mean")]].to_string(float_format=fmt))
    setup = ts[(ts["scheme"] == "v15") & (ts["entity"] != GLOBAL)].set_index("entity")["one_time_setup_ms"]
    print("v15 one-time key setup (ms): " + ", ".join(f"{short(k)} {v:.3f}" for k, v in setup.items())
          + "  (v14: none)")
    print(f"\n=== SMPC communication per client and round (RTT {a.rtt_ms:g} ms, up {a.uplink_mbps:g} Mbit/s, "
          f"down {a.downlink_mbps:g} Mbit/s) ===")
    print(comm.drop(columns="label").set_index("scheme").T.to_string(float_format=fmt))
    print("\n=== Computation ===")
    print(comp.set_index("scheme").T.to_string(float_format=fmt))
    print("\n=== Design properties ===")
    print(props.to_string(index=False))

    bench_sum = None
    if not a.no_benchmark:
        print(f"\nSame-machine benchmark: N = {a.clients}, {a.benchmark_rounds} rounds each ...")
        bench = benchmark(a.root, a.clients, a.benchmark_rounds)
        bench.to_csv(a.out / "benchmark_smpc.csv", index=False)
        bench_sum = benchmark_summary(bench)
        bench_sum.to_csv(a.out / "benchmark_smpc_summary.csv", index=False)
        print("\n=== Benchmark (ms per client per round, bytes per client per round) ===")
        print(bench_sum.to_string(index=False, float_format=fmt))

    for title, t in (("averaged over the rounds", avg), ("of the last round", last)):
        print(f"\n=== Validation metrics {title} (v14, v15, v15 - v14) ===")
        print(t.set_index(["entity", "metric"]).to_string(float_format=lambda v: f"{v:.6f}"))
    print(f"\nmax |L_avg(v15) - L_avg(v14)| over the rounds: {dl:.3e}")

    latex(ts.drop(columns="what"), a.out / "smpc_time_summary.tex",
          "SMPC times (ms) per client and Global Server: v14 additive sharing vs v15 proposed masking.",
          "tab:smpc_times")
    latex(comm.drop(columns="label"), a.out / "smpc_communication_summary.tex",
          "SMPC communication per client and round: v14 vs v15.", "tab:smpc_comm")
    latex(props, a.out / "smpc_properties.tex", "Design properties of the two SMPC schemes.", "tab:smpc_props")
    if bench_sum is not None:
        latex(bench_sum, a.out / "benchmark_smpc_summary.tex",
              "Same-machine SMPC benchmark for N clients: v14 vs v15.", "tab:smpc_bench")
    latex(avg[avg["entity"] == GLOBAL].drop(columns="entity"), a.out / "metrics_global_server.tex",
          "Global Server metrics averaged over the rounds: v14 vs v15.", "tab:smpc_metrics", digits=6)

    print("\nFigures (600-dpi PNG + vector PDF):")
    fig_times(runs, ts, figs / "fig1_smpc_times")
    fig_communication(comm, figs / "fig2_smpc_communication", a)
    if bench_sum is not None:
        fig_scaling(bench_sum, figs / "fig3_smpc_scaling_benchmark")
    fig_metrics(runs, avg, last, figs / "fig4_metrics_same_learning")
    print(f"\nTables in {a.out.resolve()}")


if __name__ == "__main__":
    main()
