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
        ax.set_xticks(np.arange(1, rounds.max() + 1, max(1, int(np.ceil(rounds.max() / 12)))))
    ax.set_xlabel("Communication round $t$")


def _legend_below(fig, axes, ncol: int, bottom: float = 0.27) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=bottom)


def plot_convergence(history: pd.DataFrame, eps_z: float, eps_l: float, converged_round: int | None,
                     path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.4))
        panels = [("delta_z", eps_z, r"Representation change $\Delta Z^{(r)}$", r"$\epsilon_z$", True),
                  ("delta_l", eps_l, r"Loss change $|\Delta L^{(r)}|$", r"$\epsilon_l$", True),
                  ("L_glob", None, r"Validation loss $L_{\mathrm{glob}}^{(r)}$", None, False)]
        for ax, letter, (col, eps, title, eps_label, log) in zip(axes, "abc", panels):
            d = history[["round", col]].dropna()
            if log:
                d = d[d[col] > 0]
            ax.plot(d["round"], d[col], color="#0072B2", marker="o")
            if eps is not None and eps > 0:
                ax.axhline(eps, color="#D55E00", linestyle="--", linewidth=1.0,
                           label=f"Threshold {eps_label} = {eps:.3g}")
                ax.legend(loc="best")
            if converged_round is not None:
                ax.axvline(converged_round, color="#555555", linestyle=":", linewidth=1.0)
            if log:
                ax.set_yscale("log")
            _round_ticks(ax, history["round"])
            _panel(ax, letter, title)
        fig.tight_layout()
        _save(fig, path)


def plot_zkp_times(zkp_log: pd.DataFrame, path: Path) -> None:
    panels = [("authentication_s", "ZKP authentication time"),
              ("proof_generation_s", "ZKP proof generation time"),
              ("proof_verification_s", "ZKP proof verification time")]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.7), sharex=True)
        for ax, letter, (col, title) in zip(axes, "abc", panels):
            for i, (client, g) in enumerate(zkp_log.groupby("client", sort=False)):
                ax.plot(g["round"], g[col], **_style(client, i))
            ax.set_ylabel("Time (s)")
            ax.set_ylim(bottom=0)
            _round_ticks(ax, zkp_log["round"])
            _panel(ax, letter, title)
        fig.tight_layout()
        _legend_below(fig, axes, ncol=4)
        _save(fig, path)


def plot_accuracy_vs_dp_budget(sweep: pd.DataFrame, history: pd.DataFrame, clients: list[str],
                               main_epsilon: float | None, path: Path) -> None:
    """(a) mean accuracy over the rounds vs per-round budget; (b) accuracy vs budget spent in the main run."""
    names = [*clients, GLOBAL]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.9))
        ax = axes[0]
        labels = ["No DP" if pd.isna(e) else f"{e:g}" for e in sweep["epsilon_per_round"]]
        x = np.arange(len(sweep))
        for i, name in enumerate(names):
            col = f"mean_accuracy[{'All clients (pooled)' if name == GLOBAL else name}]"
            ax.plot(x, sweep[col], **_style(name, i))
        if main_epsilon is not None and (sweep["epsilon_per_round"] == main_epsilon).any():
            xi = int(np.flatnonzero(sweep["epsilon_per_round"].to_numpy() == main_epsilon)[0])
            ax.axvline(xi, color="#555555", linestyle=":", linewidth=1.0)
            ax.annotate("default", (xi, 0.03), xycoords=("data", "axes fraction"), xytext=(3, 0),
                        textcoords="offset points", fontsize=7, color="#555555")
        ax.set_xticks(x, labels)
        ax.set_xlabel(r"Privacy budget per round $\epsilon$ ($\delta = 10^{-5}$)")
        ax.set_ylabel("Mean accuracy")
        ax.set_ylim(0.3, 1.0)
        _panel(ax, "a", f"Accuracy averaged over {int(sweep['rounds'].iloc[0])} rounds")

        ax = axes[1]
        for i, name in enumerate(names):
            col = f"accuracy[{'all' if name == GLOBAL else name}]"
            ax.plot(history["epsilon_spent"], history[col], **_style(name, i))
        ax.set_xlabel(r"Cumulative privacy budget spent $\epsilon_r$ ($\delta = 10^{-5}$)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.3, 1.0)
        _panel(ax, "b", "Accuracy per round vs budget spent")
        fig.tight_layout()
        _legend_below(fig, axes, ncol=5, bottom=0.3)
        _save(fig, path)


def plot_computation_cost(costs: pd.DataFrame, path: Path) -> None:
    clients = costs[costs["client"] != GLOBAL]
    server = costs[costs["client"] == GLOBAL]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.9))
        ax = axes[0]
        for i, (client, g) in enumerate(clients.groupby("client", sort=False)):
            ax.plot(g["round"], g["computation_s"], **_style(client, i))
        ax.plot(server["round"], server["computation_s"], **_style(GLOBAL))
        ax.set_ylabel("Computation time (s)")
        ax.set_ylim(bottom=0)
        _round_ticks(ax, costs["round"])
        _panel(ax, "a", "Computation cost per round")

        ax = axes[1]
        parts = [("encode_dp_s", "RF kernel, diffusion map, DP", "#56B4E9"),
                 ("proof_generation_s", "ZKP proof generation", "#E69F00"),
                 ("evaluate_s", "Validation", "#009E73"),
                 ("decode_update_s", "Decoding, RF retraining", "#CC79A7")]
        mean = clients.groupby("client", sort=False)[[p for p, _, _ in parts]].mean()
        left = np.zeros(len(mean))
        ypos = np.arange(len(mean))
        for col, label, color in parts:
            ax.barh(ypos, mean[col], left=left, color=color, label=label, height=0.6,
                    edgecolor="white", linewidth=0.5)
            left += mean[col].to_numpy()
        ax.set_yticks(ypos, [c.replace("(", "\n(") for c in mean.index])
        ax.invert_yaxis()
        ax.set_xlabel("Mean time per round (s)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", fontsize=6.5)
        _panel(ax, "b", "Client computation breakdown")
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
            ax.set_ylim(bottom=0)
            _round_ticks(ax, costs["round"])
            _panel(ax, letter, f"{title} per round")
        fig.tight_layout()
        _legend_below(fig, axes, ncol=4)
        _save(fig, path)
