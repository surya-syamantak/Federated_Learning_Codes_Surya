"""Console report: per-round tables and the end-of-run journal tables."""

import numpy as np
import pandas as pd

from .metrics import METRICS

SERVER = "Global Server"
ZKP_TIMES = ["ZKP Authentication Time (ms)", "ZKP Proof Generation Time (ms)", "ZKP Proof Verification Time (ms)"]
TRANSFER_TIMES = ["Serialization Time (ms)", "Tree Validation Time (ms)"]
BOUND_COLUMNS = ["Bounded Update", "Trees In Update", "Max Trees", "New Trees", "Max New Trees",
                 "Bound Used (%)", "Bound Check Time (ms)", "Bound Reason"]
LOCAL_COLUMNS = ["Client", "Trees in F_k", "Kept from G", "Loss L_k", "Rel. Loss Change", "Prediction Change",
                 "G/F Disagreement", "q_k", "b_k"]


def show(df):
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def stats_table(df, column, unit, parties):
    g = df.groupby("Client", sort=False)[column]
    show(pd.DataFrame({"Client": parties,
                       f"Min ({unit})": [g.min()[p] for p in parties],
                       f"Max ({unit})": [g.max()[p] for p in parties],
                       f"Mean ({unit})": [g.mean()[p] for p in parties]}))


def header(cfg, clients, schema, dp_text, zkp_group):
    print(f"FedRF-Diff | seed {cfg['seed']} | up to {cfg['max_rounds']} rounds | Proposed Pedersen ZKP "
          f"authentication: {zkp_group} group, beta = {cfg['zkp_lambda']}")
    print("Clients (train/validation patients): "
          + ", ".join(f"{c.name} {len(c.y_train)}/{len(c.y_val)}" for c in clients)
          + f"; schema {schema.version} ({schema.n_features} features, id {schema.schema_id})")
    print(f"Tree-level aggregation: F_k^(0) = {cfg['initial_trees']} trees; every round each client uploads only "
          f"its newly generated trees N_k^(r-1) (F_k^(0) in round 1, then H_k^(r-1)) as explicit trees plus the ids "
          f"of the trees in F_k^(r-1) (no feature/latent encoding, no tree is sent twice); the server validates them, "
          f"keeps every accepted tree in its repository and selects G^(r) = SelectTrees(U^(r), {cfg['aggregation_policy']}, B = {cfg['global_budget']}) without "
          f"averaging, and each client builds F_k^(r) = CapOrPrune(G^(r) + H_k^(r), B_k = {cfg['local_budget']}) "
          f"with {cfg['supplementary_trees']} supplementary trees. Vectors = leaf class-distribution vectors in an upload.")
    tuning = cfg["local_tuning"]
    if tuning:
        n_settings = 1
        for values in tuning["grid"].values():
            n_settings *= len(values)
        print(f"Local model selection: before growing F_k^(0) and every H_k^(r) each client runs a {tuning['folds']}-fold "
              f"stratified grid search ({n_settings} settings of {', '.join(tuning['grid'])}; {tuning['cv_trees']} trees "
              f"per CV forest; balanced accuracy) on its training patients; CV forests are never shared.")
    print(f"{dp_text}")
    print(f"Decentralized stopping: each client tests on its own validation patients relative loss change <= "
          f"{cfg['tau_loss']:g}, prediction change <= {cfg['tau_pred']:g} and G/F disagreement <= "
          f"{cfg['tau_agree']:g}; q_k counts consecutive passes and only b_k = [q_k >= {cfg['patience']}] is sent; "
          f"the server stops when all b_k are True or at t = {cfg['max_rounds']}")
    print(f"{SERVER} = the global model G^(r) scored on all clients' validation patients together (evaluation "
          f"harness only; the server receives no metrics).")


def round_report(rnd, zkp_rows, transfer_rows, round_metrics, local_rows, server_line, tuning_rows=()):
    print(f"\n{'=' * 30} Round {rnd} {'=' * 30}")
    if zkp_rows:
        rz = pd.DataFrame(zkp_rows).drop(columns="Round")
        print("ZKP client authentication (per client):")
        show(rz)
        print("\nZKP authentication status:")
        for _, r in rz.iterrows():
            status = ("Successful (proof valid, update accepted)" if r["Proof"] == "Valid"
                      else "Failed (proof invalid, update rejected)")
            print(f"  {r['Client']:<24}: Authentication {status}")
        print()
    if tuning_rows:
        print("Local model selection before growing H_k (stratified k-fold CV on each client's training patients):")
        show(pd.DataFrame(tuning_rows).drop(columns="Round"))
        print()
    print("Whole-tree transfer times (per client):")
    show(pd.DataFrame(transfer_rows)[["Client", *TRANSFER_TIMES]])
    tr = pd.DataFrame(transfer_rows)
    if "Bounded Update" in tr:
        print("\nBounded-update enforcement (per client):")
        show(tr[["Client", *BOUND_COLUMNS]])
    if round_metrics:
        print(f"\nValidation metrics of the global model G^({rnd}) after round {rnd}:")
        show(pd.DataFrame([{"Client": p, **m} for p, m in round_metrics.items()]))
    print(f"\nLocal three-part convergence test of F_k^({rnd}) (on each client's own validation patients; "
          f"only b_k is sent):")
    show(pd.DataFrame(local_rows)[LOCAL_COLUMNS])
    print(server_line)


def run_summary(res, names):
    parties = names + [SERVER]
    n = res.rounds_run
    print(f"\n=== Global model G^(r): metrics averaged over all {n} rounds ===")
    show(res.metrics.groupby("Client", sort=False)[METRICS].mean().reindex(parties).reset_index())
    print(f"\n=== Local models F_k^(r): metrics averaged over all {n} rounds ===")
    show(res.local.groupby("Client", sort=False)[METRICS].mean().reindex(names).reset_index())
    print(f"\n=== Final models after round {n} ===")
    last_g = res.metrics[res.metrics.Round == n].set_index("Client").reindex(parties)
    last_f = res.local[res.local.Round == n].set_index("Client").reindex(names)
    show(pd.concat([last_g[METRICS].assign(Model="G").reset_index(),
                    last_f[METRICS].assign(Model="F_k").reset_index()])[["Model", "Client", *METRICS]])

    print(f"\n=== Decentralized convergence summary over {n} rounds ===")
    g = res.local.groupby("Client", sort=False)
    show(pd.DataFrame({"Client": names,
                       "Loss Stable (rounds)": [int(g["Loss Stable"].sum()[c]) for c in names],
                       "Prediction Stable (rounds)": [int(g["Prediction Stable"].sum()[c]) for c in names],
                       "Agreement (rounds)": [int(g["Agreement"].sum()[c]) for c in names],
                       "Max q_k": [int(g["q_k"].max()[c]) for c in names],
                       "Flag True (rounds)": [int(g["b_k"].sum()[c]) for c in names],
                       "Final b_k": [bool(g["b_k"].last()[c]) for c in names]}))

    print(f"\n=== ZKP time analysis over {n} rounds (min, max and mean per client) ===")
    for col in ZKP_TIMES:
        print(f"{col}:")
        stats_table(res.zkp, col, "ms", names)

    print("\n=== ZKP authentication summary ===")
    valid = (res.zkp.Proof == "Valid").groupby(res.zkp.Client, sort=False).sum()
    summary = pd.DataFrame({"Client": names, "Rounds": n, "Successful": [int(valid[c]) for c in names]})
    summary["Failed"] = summary.Rounds - summary.Successful
    summary["ZKP Authentication"] = np.where(summary.Failed == 0, "Successful in every round",
                                             "Failed in " + summary.Failed.astype(str) + " round(s)")
    show(summary)
    ok = int(valid.sum())
    print(f"ZKP authentications: {ok}/{len(res.zkp)} successful (updates accepted), "
          f"{len(res.zkp) - ok} failed (updates rejected)")

    print("\n=== Server tree validation summary ===")
    g = res.transfer.groupby("Client", sort=False)
    trees = pd.DataFrame({"Client": names, "Trees Sent": [int(g["Trees Sent"].sum()[c]) for c in names],
                          "Trees Accepted": [int(g["Trees Accepted"].sum()[c]) for c in names]})
    trees["Trees Rejected"] = trees["Trees Sent"] - trees["Trees Accepted"]
    show(trees)
    print(f"Tree validation: {trees['Trees Accepted'].sum()}/{trees['Trees Sent'].sum()} uploaded trees accepted, "
          f"{trees['Trees Rejected'].sum()} rejected (schema, node references, split fields, leaf outputs, "
          f"size and depth limits, tree provenance, no re-sent trees)")
    if "Bounded Update" in res.transfer:
        print("\n=== Bounded-update enforcement summary ===")
        t = res.transfer
        rows = []
        for c in names:
            tc = t[t["Client"] == c]
            rows.append({"Client": c, "Within bound": int((tc["Bounded Update"] == "Within bound").sum()),
                         "Out of bound": int((tc["Bounded Update"] == "Out of bound").sum()),
                         "Max Bound Used (%)": tc["Bound Used (%)"].max(),
                         "Mean Check Time (ms)": tc["Bound Check Time (ms)"].mean()})
        bt = pd.DataFrame(rows)
        show(bt)
        checked = int((t["Bounded Update"] != "n/a").sum())
        print(f"Bounded updates: {bt['Within bound'].sum()}/{checked} within bound (accepted), "
              f"{bt['Out of bound'].sum()} out of bound (rejected); bounds: trees per update <= "
              f"max(initial_trees, local_budget), new tree ids <= max(initial_trees, supplementary_trees) "
              f"x rounds since the last accepted update")

    if not res.tuning.empty:
        print(f"\n=== Local tuning time analysis over {n} rounds (min, max and mean per client) ===")
        print("Tuning Time (ms):")
        stats_table(res.tuning, "Tuning Time (ms)", "ms", names)

    print(f"\n=== Whole-tree transfer time analysis over {n} rounds (min, max and mean per client) ===")
    for col in TRANSFER_TIMES:
        print(f"{col}:")
        stats_table(res.transfer, col, "ms", names)


def dp_sweep_report(avg, last, rounds):
    print(f"Accuracy of G^(r) averaged over the {rounds} rounds:")
    show(avg)
    print(f"Accuracy of G^({rounds}) after round {rounds}:")
    show(last)
