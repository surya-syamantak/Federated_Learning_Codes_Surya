"""Journal-quality figures (600-dpi PNG plus vector PDF of each).

Style: serif text (Times New Roman when installed), 8-9 pt type, the
colour-blind-safe Okabe-Ito palette with a distinct marker per client (so no
series relies on colour alone), light dashed grid, panel labels (a), (b), (c),
and a double-column width of 7.16 in.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DOUBLE_COLUMN = 7.16  # inches
GLOBAL = "Global Server"
CLIENT_STYLE = {  # Okabe-Ito colours, one marker per client
    "Client 1(Hungary)": ("#0072B2", "o"),
    "Client 2(Switzerland)": ("#E69F00", "s"),
    "Client 3(VA Long Beach)": ("#009E73", "^"),
    "Client 4(Cleveland)": ("#CC79A7", "D"),
    GLOBAL: ("#000000", "*"),
}
FALLBACK = [("#56B4E9", "v"), ("#D55E00", "P"), ("#F0E442", "X")]

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
    "grid.color": "#d9d9d9",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}


def _style(name: str, i: int = 0) -> dict:
    color, marker = CLIENT_STYLE.get(name, FALLBACK[i % len(FALLBACK)])
    kw = {"color": color, "marker": marker, "label": name}
    if name == GLOBAL:
        kw.update(linestyle="--", markersize=7, linewidth=1.6)
    return kw


def _bar_style(name: str, i: int = 0) -> dict:
    """Bar colour per client; the Global Server bars are hatched so they do not rely on colour."""
    color, _ = CLIENT_STYLE.get(name, FALLBACK[i % len(FALLBACK)])
    kw = {"color": color, "label": name, "edgecolor": "white", "linewidth": 0.3}
    if name == GLOBAL:
        kw.update(color="#555555", hatch="////")
    return kw


def _grouped_bars(ax, x: np.ndarray, series: list[tuple[str, np.ndarray]], width: float = 0.82) -> None:
    """One bar per series side by side at every x position."""
    w = width / len(series)
    for j, (name, values) in enumerate(series):
        ax.bar(x + (j - (len(series) - 1) / 2) * w, values, w, zorder=2, **_bar_style(name, j))
    ax.grid(axis="x", visible=False)


def _panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, path: Path) -> None:
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _round_ticks(ax, rounds) -> None:
    rounds = np.asarray(rounds)
    if len(rounds):
        ax.set_xticks(np.arange(1, rounds.max() + 1, max(1, int(np.ceil(rounds.max() / 8)))))
    ax.set_xlabel("Communication round $t$")


def _legend_below(fig, axes, ncol: int, bottom: float = 0.27) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=bottom)


def plot_convergence(history: pd.DataFrame, eps_z: float, eps_l: float, converged_round: int | None,
                     path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.4))
        panels = [("delta_z", eps_z, r"Representation change $\Delta Z^{(t)}$", r"$\epsilon_Z$", False),
                  ("delta_l", eps_l, r"Loss change $|\Delta L^{(t)}|$", r"$\epsilon_L$", False),
                  ("L_avg", None, r"Validation loss $L_{\mathrm{avg}}^{(t)}$", None, False)]
        for ax, letter, (col, eps, title, eps_label, log) in zip(axes, "abc", panels):
            d = history[["round", col]].dropna()
            if log:
                d = d[d[col] > 0]
            ax.plot(d["round"], d[col], color="#0072B2", marker="o")
            threshold = None
            if eps is not None and eps > 0:
                ax.axhline(eps, color="#D55E00", linestyle="--", linewidth=1.0,
                           label=f"Threshold {eps_label} = {eps:.3g}")
                threshold = eps
            if threshold is not None:
                # A free band at the top for the legend, so it never covers a data point or line.
                top = max(threshold, d[col].max() if len(d) else 0)
                ax.set_ylim(0, top * 1.3)
                ax.legend(loc="upper left")
            if converged_round is not None:
                ax.axvline(converged_round, color="#555555", linestyle=":", linewidth=1.0)
            if log:
                ax.set_yscale("log")
            elif col in ("delta_z", "delta_l") and threshold is None:
                ax.set_ylim(bottom=0)
            _round_ticks(ax, history["round"])
            _panel(ax, letter, title)
        fig.tight_layout()
        _save(fig, path)


def plot_zkp_times(zkp_log: pd.DataFrame, path: Path) -> None:
    panels = [("authentication_s", "ZKP authentication time"),
              ("proof_generation_s", "ZKP proof generation time"),
              ("proof_verification_s", "ZKP proof verification time")]
    # Grouped bars (one per client per round), one full-width panel per time.
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(DOUBLE_COLUMN, 6.6), sharex=True)
        times = zkp_log.pivot(index="round", columns="client", values=[c for c, _ in panels])
        rounds = times.index.to_numpy()
        clients = list(dict.fromkeys(zkp_log["client"]))
        for ax, letter, (col, title) in zip(axes, "abc", panels):
            _grouped_bars(ax, rounds, [(c, times[(col, c)].to_numpy() * 1000) for c in clients])
            ax.set_ylabel("Time (ms)")
            ax.set_ylim(bottom=0)
            _round_ticks(ax, rounds)
            ax.set_xticks(rounds)  # every round: the panels are full width
            _panel(ax, letter, title)
        for ax in axes[:-1]:
            ax.set_xlabel("")
        fig.tight_layout()
        _legend_below(fig, axes, ncol=4, bottom=0.1)
        _save(fig, path)


def plot_smpc_times(smpc_log: pd.DataFrame, path: Path) -> None:
    """(a) client share generation, (b) client share summation, (c) server reconstruction, per round."""
    clients = smpc_log[smpc_log["client"] != GLOBAL]
    server = smpc_log[smpc_log["client"] == GLOBAL]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(DOUBLE_COLUMN, 6.6), sharex=True)
        times = clients.pivot(index="round", columns="client", values=["share_s", "sum_s"])
        rounds = times.index.to_numpy()
        names = list(dict.fromkeys(clients["client"]))
        for ax, letter, (col, title) in zip(axes[:2], "ab", [("share_s", "SMPC share generation time (client)"),
                                                            ("sum_s", "SMPC share summation time (client)")]):
            _grouped_bars(ax, rounds, [(c, times[(col, c)].to_numpy() * 1000) for c in names])
            ax.set_ylabel("Time (ms)")
            ax.set_ylim(bottom=0)
            ax.set_xticks(rounds)
            _panel(ax, letter, title)
        ax = axes[2]
        _grouped_bars(ax, server["round"].to_numpy(), [(GLOBAL, server["reconstruct_s"].to_numpy() * 1000)],
                      width=0.6)
        ax.set_ylabel("Time (ms)")
        ax.set_ylim(bottom=0)
        _round_ticks(ax, server["round"])
        ax.set_xticks(server["round"].to_numpy())
        _panel(ax, "c", "SMPC reconstruction time (Global Server)")
        for a in axes[:-1]:
            a.set_xlabel("")
        fig.tight_layout()
        _legend_below(fig, axes, ncol=4, bottom=0.1)
        _save(fig, path)


def plot_computation_cost(costs: pd.DataFrame, path: Path) -> None:
    clients = costs[costs["client"] != GLOBAL]
    server = costs[costs["client"] == GLOBAL]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 3.0), gridspec_kw={"width_ratios": [1.05, 1.35, 0.8]})
        ax = axes[0]
        for i, (client, g) in enumerate(clients.groupby("client", sort=False)):
            ax.plot(g["round"], g["computation_s"], **_style(client, i))
        ax.plot(server["round"], server["computation_s"], **_style(GLOBAL))
        ax.set_ylabel("Computation time (s)")
        ax.set_ylim(bottom=0)
        _round_ticks(ax, costs["round"])
        _panel(ax, "a", "Computation cost per round")

        ax = axes[1]
        parts = [("encode_s", "RF kernel, diffusion map", "#56B4E9"),
                 ("proof_generation_s", "ZKP proof generation", "#E69F00"),
                 ("evaluate_s", "Validation", "#009E73"),
                 ("decode_update_s", "Decoding, RF retraining", "#CC79A7"),
                 ("smpc_s", "SMPC secret sharing", "#D55E00")]
        mean = clients.groupby("client", sort=False)[[p for p, _, _ in parts]].mean()
        left = np.zeros(len(mean))
        ypos = np.arange(len(mean))
        for col, label, color in parts:
            ax.barh(ypos, mean[col], left=left, color=color, label=label, height=0.6,
                    edgecolor="white", linewidth=0.5)
            left += mean[col].to_numpy()
        # Proof generation takes milliseconds next to about a second of local work, so its
        # segment is too thin to see: its value is printed under each client's name and
        # drawn on its own scale in panel (c).
        zkp_ms = mean["proof_generation_s"].to_numpy() * 1000
        smpc_ms = mean["smpc_s"].to_numpy() * 1000  # SMPC is even thinner: printed the same way
        ax.set_yticks(ypos, [c.replace("(", "\n(") + f"\nZKP {z:.2f} ms, SMPC {s:.3f} ms"
                             for c, z, s in zip(mean.index, zkp_ms, smpc_ms)], fontsize=6.2)
        ax.invert_yaxis()
        ax.set_xlabel("Mean time per round (s)")
        ax.set_xlim(0, 1.75 * left.max())  # free space on the right for the legend
        ax.grid(axis="y", visible=False)
        ax.legend(loc="center right", fontsize=6)
        _panel(ax, "b", "Client computation breakdown")

        ax = axes[2]
        bars = ax.bar(ypos, zkp_ms, 0.6, color="#E69F00", edgecolor="white", linewidth=0.5, zorder=2)
        ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=1.5)
        ax.set_xticks(ypos, [c.split("(")[0].strip().replace(" ", "\n") for c in mean.index], fontsize=7)
        ax.set_ylabel("Mean time per round (ms)")
        ax.set_ylim(0, 1.2 * zkp_ms.max() if zkp_ms.max() > 0 else 1.0)
        ax.grid(axis="x", visible=False)
        _panel(ax, "c", "ZKP proof generation")
        fig.tight_layout()
        _legend_below(fig, axes, ncol=5, bottom=0.28)
        _save(fig, path)


def plot_communication_cost(costs: pd.DataFrame, path: Path) -> None:
    clients = costs[costs["client"] != GLOBAL]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.7), sharex=True)
        for ax, letter, (col, title) in zip(axes, "ab", [("bytes_up", "Uplink (client to server)"),
                                                      ("bytes_down", "Downlink (server to client)")]):
            for i, (client, g) in enumerate(clients.groupby("client", sort=False)):
                ax.plot(g["round"], g[col] / 1024, **_style(client, i))
            ax.set_ylabel("Communication cost (KB)")
            ax.set_ylim(0, clients[col].max() / 1024 * 1.18)  # headroom above the top series
            _round_ticks(ax, costs["round"])
            spread = clients.groupby("round")[col].agg(lambda b: b.max() - b.min()).max()
            if spread < 0.05 * clients[col].max():  # all clients receive about the same
                ax.text(0.5, 0.06, "Same for every client: dominated by the global model $M_G$",
                        transform=ax.transAxes, ha="center", fontsize=7, color="#555555")
            _panel(ax, letter, f"{title} per round")
        fig.tight_layout()
        _legend_below(fig, axes, ncol=4)
        _save(fig, path)
