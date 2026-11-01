"""Compare the new-trees-only FL version with FL_ProposedZKP (whole forest) and FedRF-Diff v15.

    FL-NT  Federated-Learning_ProposedZKP_NewTrees      uploads only newly generated trees (+ kept ids)
    FL     Federated-Learning_ProposedZKP               uploads the complete forest F_k^(r-1) every round
    v15    Fed_Code_v15_ProposedPedersenZKP_ProposedSMPC_NoDP
           diffusion-map latents + proposed Pedersen ZKP + proposed SMPC, no DP

All three use the same proposed Pedersen ZKP (beta = 2). FL and FL-NT add Laplace DP on the
leaf vectors (eps_r ~ U[2, 3]); v15 has no DP. Inputs are the saved runs in outputs/run/ of each
folder. Outputs: outputs/comparison_summary.csv, outputs/per_round.csv and two figures.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "outputs"
SERVER = "Global Server"
METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "MCC", "Kappa"]
FL_METRICS = dict(zip(METRICS, ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score",
                                "Matthews Correlation Coefficient (MCC)", "Cohen's Kappa"]))
V15_METRICS = dict(zip(METRICS, ["accuracy", "precision", "recall", "specificity", "f1", "mcc", "kappa"]))
NAMES = {"FL-NT": "FL_ProposedZKP_NewTrees (new trees only)", "FL": "FL_ProposedZKP (whole forest)",
         "v15": "v15 ProposedPedersenZKP + ProposedSMPC"}


def fl_run(folder):
    run = ROOT / folder / "outputs/run"
    rounds = pd.read_csv(run / "rounds.csv")
    val = pd.read_csv(run / "validation_metrics.csv")
    g = val[val.Client == SERVER].set_index("Round")
    cost = pd.read_csv(run / "communication_computation_cost.csv")
    clients = cost[cost.Client != SERVER]
    zkp = pd.read_csv(run / "zkp_times.csv")
    transfer = pd.read_csv(run / "tree_transfer_times.csv")
    per_round = pd.DataFrame({
        "Round": rounds.Round, "Accuracy": g.loc[rounds.Round, "Accuracy"].to_numpy(),
        "Uplink (KB)": clients.groupby("Round")["Uplink (KB)"].sum().to_numpy(),
        "Downlink (KB)": clients.groupby("Round")["Downlink (KB)"].sum().to_numpy(),
        "Trees Uploaded": rounds["Trees Uploaded"], "Round Time (s)": rounds["Round Time (s)"]})
    summary = {
        "Rounds run": len(rounds), "Stop": rounds.Status.iloc[-1],
        "DP": "Laplace, eps ~ U[2, 3]",
        "What is uploaded": "trees",
        "Trees uploaded (total)": int(rounds["Trees Uploaded"].sum()),
        "Trees uploaded per client-round": transfer["Trees Sent"].mean(),
        "Uplink total (KB)": per_round["Uplink (KB)"].sum(),
        "Uplink per client-round (KB)": clients["Uplink (KB)"].mean(),
        "Downlink total (KB)": per_round["Downlink (KB)"].sum(),
        "Serialization per client-round (ms)": transfer["Serialization Time (ms)"].mean(),
        "Server tree validation per client-round (ms)": transfer["Tree Validation Time (ms)"].mean(),
        "ZKP authentication per client-round (ms)": zkp["ZKP Authentication Time (ms)"].mean(),
        "Mean round time (s)": rounds["Round Time (s)"].mean(),
        "Total time (s)": rounds["Round Time (s)"].sum(),
    }
    for m in METRICS:
        summary[f"Final {m}"] = g[FL_METRICS[m]].iloc[-1]
        summary[f"Mean {m}"] = g[FL_METRICS[m]].mean()
    return summary, per_round


def v15_run(folder="Fed_Code_v15_ProposedPedersenZKP_ProposedSMPC_NoDP"):
    run = ROOT / folder / "outputs/run"
    hist = pd.read_csv(run / "history.csv")
    g = pd.read_csv(run / "global_server_metrics.csv").set_index("round")
    cost = pd.read_csv(run / "costs.csv")
    zkp = pd.read_csv(run / "zkp_times.csv")
    per_round = pd.DataFrame({
        "Round": hist["round"], "Accuracy": g.loc[hist["round"], "accuracy"].to_numpy(),
        "Uplink (KB)": cost.groupby("round")["bytes_up"].sum().to_numpy() / 1024,
        "Downlink (KB)": cost.groupby("round")["bytes_down"].sum().to_numpy() / 1024,
        "Trees Uploaded": 0, "Round Time (s)": hist["round_seconds"]})
    summary = {
        "Rounds run": len(hist), "Stop": "converged" if len(hist) < 30 else "maximum rounds reached",
        "DP": "none", "What is uploaded": "diffusion-map latents",
        "Trees uploaded (total)": 0, "Trees uploaded per client-round": 0.0,
        "Uplink total (KB)": per_round["Uplink (KB)"].sum(),
        "Uplink per client-round (KB)": cost["bytes_up"].mean() / 1024,
        "Downlink total (KB)": per_round["Downlink (KB)"].sum(),
        "Serialization per client-round (ms)": 1000 * cost["encode_s"].mean(),
        "Server tree validation per client-round (ms)": float("nan"),
        "ZKP authentication per client-round (ms)": 1000 * zkp["authentication_s"].mean(),
        "Mean round time (s)": hist["round_seconds"].mean(),
        "Total time (s)": hist["round_seconds"].sum(),
    }
    for m in METRICS:
        summary[f"Final {m}"] = g[V15_METRICS[m]].iloc[-1]
        summary[f"Mean {m}"] = g[V15_METRICS[m]].mean()
    return summary, per_round


def main():
    OUT.mkdir(exist_ok=True)
    runs = {"FL-NT": fl_run("Federated-Learning_ProposedZKP_NewTrees"), "FL": fl_run("Federated-Learning_ProposedZKP"),
            "v15": v15_run()}
    summary = pd.DataFrame({NAMES[k]: s for k, (s, _) in runs.items()})
    summary.index.name = "Quantity"
    summary.to_csv(OUT / "comparison_summary.csv")
    pd.concat([r.assign(Version=NAMES[k]) for k, (_, r) in runs.items()]).to_csv(OUT / "per_round.csv", index=False)
    nt, fl = runs["FL-NT"][0], runs["FL"][0]
    with pd.option_context("display.width", 200, "display.max_colwidth", 40, "display.float_format", "{:.4g}".format):
        print(summary.to_string())
    print(f"\nNew trees only vs whole forest: uplink {nt['Uplink total (KB)']:.0f} KB vs {fl['Uplink total (KB)']:.0f} KB "
          f"({100 * (1 - nt['Uplink total (KB)'] / fl['Uplink total (KB)']):.1f} % less), trees uploaded "
          f"{nt['Trees uploaded (total)']} vs {fl['Trees uploaded (total)']}; final accuracy "
          f"{nt['Final Accuracy']:.4f} vs {fl['Final Accuracy']:.4f}")

    colors = {"FL-NT": "#2a78b5", "FL": "#e0832b", "v15": "#3a9a5b"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for k, (_, r) in runs.items():
        axes[0].plot(r.Round, r["Uplink (KB)"], marker="o", ms=3, color=colors[k], label=NAMES[k])
        axes[1].plot(r.Round, r["Uplink (KB)"].cumsum(), marker="o", ms=3, color=colors[k])
        axes[2].plot(r.Round, r.Accuracy, marker="o", ms=3, color=colors[k])
    for ax, title, ylabel in zip(axes, ["(a) Uplink per round, all clients", "(b) Cumulative uplink",
                                        "(c) Global model accuracy (Global Server)"],
                                 ["KB", "KB", "Accuracy"]):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Round r")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig1_uplink_accuracy.png", dpi=600)
    fig.savefig(OUT / "fig1_uplink_accuracy.pdf")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.27
    for i, k in enumerate(runs):
        vals = [runs[k][0][f"Final {m}"] for m in METRICS]
        ax.bar([j + (i - 1) * width for j in range(len(METRICS))], vals, width, color=colors[k], label=NAMES[k])
    ax.set_xticks(range(len(METRICS)), METRICS)
    ax.set_ylabel("Final round (Global Server)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_final_metrics.png", dpi=600)
    fig.savefig(OUT / "fig2_final_metrics.pdf")
    print(f"\nSaved {OUT / 'comparison_summary.csv'}, per_round.csv and fig1/fig2 (.png + .pdf)")


if __name__ == "__main__":
    main()
