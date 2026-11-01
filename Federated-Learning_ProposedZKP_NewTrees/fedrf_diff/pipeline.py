"""Federated Random Forest with tree-level aggregation (Steps 1-7) and the DP-budget experiment.

    t = 0   every client trains F_k^(0); G^(0) is empty; q_k = 0
    t >= 1  2  clients authenticate (proposed ZKP) and upload only N_k^(r-1), the trees grown since
               their last accepted upload (F_k^(0), then H_k^(r-1); explicit trees + schema) and
               the ids of the trees in F_k^(r-1)
            3  server validates, keeps every accepted tree in its repository, pools
               U^(r) = G^(r-1) ∪ N^(r-1) ∪ kept trees, selects    G^(r) = SelectTrees(U^(r), A, B)
            4  server broadcasts G^(r), t and the schema
            5  clients refine                            F_k^(r) = CapOrPrune(TreePool(G^(r), H_k^(r)), B_k)
            6  clients test loss stability, prediction stability and global/local agreement on their
               own validation patients and send only b_k^(r) = [q_k >= P]
            7  server stops when every b_k is True ("all-client convergence") or t = R_max

The per-round validation tables are written by the experiment harness for the paper; in the
protocol itself no validation metric ever leaves a client.
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import report, zkp
from .alignment import agree_schema, federated_fill_values
from .client import FLAG_BYTES, HospitalClient
from .config import CONFIG
from .data import load_hospitals, mark_missing, save_client_csvs, split_hospitals
from .metrics import metrics_from_counts
from .privacy import PrivacyAccountant, describe, epsilon_schedule, laplace_scale
from .report import SERVER
from .server import FederatedServer, Limits, save_forest

CONVERGED = "all-client convergence"
MAX_ROUNDS = "maximum rounds reached"


@dataclass
class RunResult:
    rounds_run: int
    status: str
    metrics: pd.DataFrame  # G^(r) on every client's validation patients + Global Server (harness only)
    local: pd.DataFrame  # three-part test and F_k^(r) metrics per client and round
    tuning: pd.DataFrame  # local model selection per client and round
    zkp: pd.DataFrame
    transfer: pd.DataFrame
    cost: pd.DataFrame
    rounds: pd.DataFrame
    schedule: list
    epsilon_per_patient: float
    server: FederatedServer
    clients: list

    @property
    def converged(self):
        return self.status == CONVERGED


def ms(seconds):
    return 1000 * seconds


def bound_columns(b):
    """Transfer-table columns of the bounded-update check (n/a when the upload never reached it)."""
    if b is None or not b["enabled"]:
        return {"Bounded Update": "n/a", "Trees In Update": b["trees"] if b else 0,
                "New Trees": b["new"] if b else 0, "Max Trees": None, "Max New Trees": None,
                "Bound Used (%)": float("nan"), "Bound Check Time (ms)": 0.0, "Bound Reason": ""}
    return {"Bounded Update": "Within bound" if b["within"] else "Out of bound",
            "Trees In Update": b["trees"], "New Trees": b["new"], "Max Trees": b["max_trees"],
            "Max New Trees": b["max_new"], "Bound Used (%)": b["used_pct"],
            "Bound Check Time (ms)": b["check_s"] * 1000, "Bound Reason": b["reason"]}


def build_federation(cfg, with_zkp=True):
    """Schema agreement, hospital clients, federated imputation.

    The proposed ZKP needs no registration: in round 1 every client's previous
    commitment C_{r-1} is 1 on both sides.
    """
    schema = agree_schema()
    df = mark_missing(load_hospitals(cfg["raw_csv"]))
    clients = [HospitalClient(cid, hospital, train, val, schema, cfg, with_zkp)
               for cid, (hospital, train, val) in enumerate(split_hospitals(df, cfg["val_size"], cfg["seed"]), 1)]
    fill_values = federated_fill_values([c.imputation_stats() for c in clients])
    for c in clients:
        c.apply_preprocessing(fill_values)
    server = FederatedServer(schema, Limits.from_config(cfg), cfg["zkp_lambda"], cfg["aggregation_policy"],
                             cfg["global_budget"])
    return schema, clients, server


def dp_description(spec, cfg):
    if spec is None:
        return "No DP: leaf vectors are shared without noise"
    if isinstance(spec, (tuple, list)):
        draw = f"epsilon drawn uniformly from [{spec[0]:g}, {spec[1]:g}] for each round (seed {cfg['seed']}"
    else:
        draw = f"epsilon = {spec:g} for each round ("
    return (f"Laplace DP on every leaf vector of every newly grown tree: {draw}; pure DP, delta = 0), "
            f"L1 clip norm C = {cfg['dp_clip']:g}, Laplace scale b = 2C / epsilon (privacy.py)")


def run_federation(cfg, dp_spec, rounds, with_zkp=True, verbose=True, stop_early=True, impersonate=None):
    """Steps 1-7 for up to `rounds` rounds.

    ``impersonate`` maps a round to a client id (e.g. {2: "client_3"}): in that round an
    impostor replays the client's previous ZKP session instead of the client and uploads a
    forest in its name; the proposed ZKP rejects it (fresh challenge) and the upload is dropped.
    """
    schema, clients, server = build_federation(cfg, with_zkp)
    creators = {c.client_id for c in clients}
    schedule = epsilon_schedule(dp_spec, rounds + 1, cfg["seed"])  # index 0 = the initial forests
    privacy = None if dp_spec is None else {"mechanism": "laplace", "clip_l1": cfg["dp_clip"]}
    accountant = PrivacyAccountant()
    if verbose:
        report.header(cfg, clients, schema, dp_description(dp_spec, cfg), zkp.GROUP)

    # Step 1 · initialisation (t = 0)
    for c in clients:
        c.initialise(schedule[0])
    accountant.spend(schedule[0], cfg["initial_trees"])
    if verbose:
        print(f"\nRound 0 (initialisation): every client trained F_k^(0) with {cfg['initial_trees']} trees; "
              f"G^(0) is empty; q_k = 0 for every client.")

    zkp_rows, transfer_rows, cost_rows, metric_rows, local_rows, round_rows = [], [], [], [], [], []
    tuning_rows = []
    status, rnd = MAX_ROUNDS, 0
    for rnd in range(1, rounds + 1):
        t_round = time.perf_counter()
        epsilon = schedule[rnd]

        # Step 2 · every client uploads only its newly generated trees N_k^(r-1)
        up = {}
        for c in clients:
            u = up[c.cid] = {"n_trees": len(c.pending), "vectors": sum(t.n_leaves for t in c.pending),
                             "forest_trees": len(c.forest)}
            t0 = time.perf_counter()
            u["payload"] = c.serialize(rnd, privacy)
            u["ser"] = time.perf_counter() - t0

        # Step 3 · authentication (proposed ZKP session: C_r and T up, challenge c down,
        # responses y_v and y_s up, verdict down), structural validation, tree-level aggregation
        new_trees, kept_lists = [], []
        for c in clients:
            u = up[c.cid]
            if with_zkp:
                replay = c.prover.last_transcript if impersonate and impersonate.get(rnd) == c.client_id else None
                u["session"] = zkp.authenticate(server.verifier.pp, c.prover, server.verifier, rnd,
                                                vectors=u["vectors"], replay=replay)
                u["valid"] = u["session"].authenticated
            else:
                u["session"], u["valid"] = None, True
            t0 = time.perf_counter()
            trees, rejected = (server.receive(c.client_id, u["payload"], rnd, creators) if u["valid"]
                               else ([], [("<update>", "authentication failed")]))
            u["check"] = time.perf_counter() - t0
            u["bound"] = server.last_bound if u["valid"] else None
            u["accepted"], u["rejected"] = len(trees), rejected
            if u["valid"] and all(i != "<update>" for i, _ in rejected):
                c.acknowledge()  # verdict down: these trees are in the server's repository now
                kept_lists.append(server.kept[c.client_id])
            new_trees.append(trees)
        t0 = time.perf_counter()
        pool_size = server.aggregate(new_trees, kept_lists)
        t_agg = time.perf_counter() - t0
        global_forest = server.global_forest

        # Steps 4-5 · broadcast G^(r); every client refines F_k^(r)
        global_ids = {t.tree_id for t in global_forest}
        for c in clients:
            t0 = time.perf_counter()
            c.refine(global_forest, rnd, epsilon)
            up[c.cid]["refine"] = time.perf_counter() - t0
            up[c.cid]["kept_global"] = sum(t.tree_id in global_ids for t in c.forest)
            if c.tuning is not None:
                params, cv_score, fits = c.tuning
                tuning_rows.append({"Round": rnd, "Client": c.name,
                                    **{f"Best {k}": str(params[k]) for k in cfg["local_tuning"]["grid"]},
                                    "CV Balanced Accuracy": cv_score, "CV Fits": fits,
                                    "Tuning Time (ms)": ms(c.tuning_seconds)})
        eps_patient = accountant.spend(epsilon, cfg["supplementary_trees"])

        # Step 6 · decentralized three-part convergence test; only b_k leaves the client
        checks, global_counts = {}, {}
        for c in clients:
            t0 = time.perf_counter()
            checks[c.cid], local_counts, global_counts[c.cid] = c.convergence_check()
            up[c.cid]["conv"] = time.perf_counter() - t0
            local_rows.append({"Round": rnd, "Client": c.name, "Trees in F_k": len(c.forest),
                               "Kept from G": up[c.cid]["kept_global"], "Loss L_k": checks[c.cid].loss,
                               "Rel. Loss Change": checks[c.cid].rel_loss_change,
                               "Prediction Change": checks[c.cid].pred_change,
                               "G/F Disagreement": checks[c.cid].disagreement,
                               "Loss Stable": checks[c.cid].loss_stable, "Prediction Stable": checks[c.cid].pred_stable,
                               "Agreement": checks[c.cid].agrees, "q_k": checks[c.cid].patience,
                               "b_k": checks[c.cid].flag, **metrics_from_counts(local_counts)})
        flags = [checks[c.cid].flag for c in clients]

        # Step 7 · server stopping decision from the boolean flags only
        decision = server.stopping_decision(flags, rnd, rounds)
        round_time = time.perf_counter() - t_round

        # Harness bookkeeping (never part of the protocol messages)
        round_metrics = {}
        if global_forest:  # G^(r) is empty only if every upload was rejected
            round_metrics = {c.name: metrics_from_counts(global_counts[c.cid]) for c in clients}
            round_metrics[SERVER] = metrics_from_counts(np.sum(list(global_counts.values()), axis=0))
        for party, m in round_metrics.items():
            metric_rows.append({"Round": rnd, "Client": party, **m})
        global_kb = server.global_bytes / 1024
        for c in clients:
            u = up[c.cid]
            s = u["session"]
            if with_zkp:
                zkp_rows.append({"Round": rnd, "Client": c.name, "Proof": s.outcome,
                                 "Trees": u["n_trees"], "Vectors": u["vectors"],
                                 "ZKP Authentication Time (ms)": ms(s.authentication_s),
                                 "ZKP Proof Generation Time (ms)": ms(s.proof_generation_s),
                                 "ZKP Proof Verification Time (ms)": ms(s.proof_verification_s)})
            transfer_rows.append({"Round": rnd, "Client": c.name, "Serialization Time (ms)": ms(u["ser"]),
                                  "Tree Validation Time (ms)": ms(u["check"]), "Trees in F_k^(r-1)": u["forest_trees"],
                                  "Trees Sent": u["n_trees"],
                                  "Trees Accepted": u["accepted"], **bound_columns(u["bound"]),
                                  "Rejections": "; ".join(f"{i}: {r}" for i, r in u["rejected"])})
            zkp_up, zkp_down = (s.bytes_up, s.bytes_down) if s else (0, 0)  # C_r, T, y_v, y_s / c, verdict
            cost_rows.append({"Round": rnd, "Client": c.name,
                              "Uplink (KB)": (len(u["payload"]) + zkp_up + FLAG_BYTES) / 1024,
                              "Downlink (KB)": global_kb + zkp_down / 1024,
                              "Local Refinement Time (ms)": ms(u["refine"]),
                              "Local Tuning Time (ms)": ms(c.tuning_seconds) if c.tuning else 0.0,
                              "Convergence Check Time (ms)": ms(u["conv"]), "Aggregation Time (ms)": np.nan,
                              "Global Forest (KB)": np.nan})
        round_costs = pd.DataFrame(cost_rows[-len(clients):])
        cost_rows.append({"Round": rnd, "Client": SERVER, "Uplink (KB)": round_costs["Uplink (KB)"].sum(),
                          "Downlink (KB)": round_costs["Downlink (KB)"].sum(), "Local Refinement Time (ms)": np.nan,
                          "Local Tuning Time (ms)": np.nan,
                          "Convergence Check Time (ms)": np.nan, "Aggregation Time (ms)": ms(t_agg),
                          "Global Forest (KB)": global_kb})
        n_sent = sum(u["n_trees"] for u in up.values())
        n_accepted = sum(u["accepted"] for u in up.values())
        round_rows.append({"Round": rnd, "Trees Uploaded": n_sent, "Trees Rejected": n_sent - n_accepted,
                           "Candidate Pool |U|": pool_size, "Global Trees |G|": len(global_forest),
                           "Flags True": sum(flags), "Clients": len(clients), "Status": decision or "continue",
                           "Epsilon per Leaf Vector": epsilon,
                           "Laplace Scale b": None if epsilon is None else laplace_scale(epsilon, cfg["dp_clip"]),
                           "Epsilon per Patient (round)": eps_patient, "Epsilon per Patient (total)": accountant.total,
                           "Global Forest (KB)": global_kb, "Round Time (s)": round_time})

        if verbose:
            dp = ("DP off" if epsilon is None else
                  f"DP budget: epsilon = {epsilon:.3f} per leaf vector this round (pure DP); per patient "
                  f"{eps_patient:.2f} this round, {accountant.total:.2f} in total")
            line = (f"Server: U^(r) = {pool_size} candidate trees ({n_accepted}/{n_sent} uploaded trees valid) -> "
                    f"G^(r) = {len(global_forest)} trees (B = {cfg['global_budget']}, policy "
                    f"{cfg['aggregation_policy']}) | convergence flags {sum(flags)}/{len(clients)} True | "
                    f"decision: {decision or 'continue'} | round time {round_time:.1f} s | {dp}")
            report.round_report(rnd, zkp_rows[-len(clients):] if with_zkp else [], transfer_rows[-len(clients):],
                                round_metrics, local_rows[-len(clients):], line,
                                tuning_rows[-len(clients):] if tuning_rows else [])

        if decision is not None:
            status = decision
            if decision == CONVERGED and stop_early:
                break

    return RunResult(rnd, status, pd.DataFrame(metric_rows), pd.DataFrame(local_rows), pd.DataFrame(tuning_rows),
                     pd.DataFrame(zkp_rows),
                     pd.DataFrame(transfer_rows), pd.DataFrame(cost_rows), pd.DataFrame(round_rows),
                     schedule[:rnd + 1], accountant.total, server, clients)


def run_dp_sweep(cfg, rounds, names):
    """Accuracy of G^(r) for the same federation (same seed, no ZKP) under every DP budget."""
    avg_rows, last_rows = [], []
    for spec in cfg["dp_sweep"]:
        res = run_federation(cfg, spec, rounds, with_zkp=False, verbose=False, stop_early=False)
        eps = [e for e in res.schedule if e is not None]
        info = {"Epsilon per round": describe(spec),
                "Mean epsilon per round": float(np.mean(eps)) if eps else 0.0,
                "Mean Laplace scale b": float(np.mean([laplace_scale(e, cfg["dp_clip"]) for e in eps])) if eps else 0.0,
                "Epsilon spent per patient": res.epsilon_per_patient}
        acc = res.metrics.pivot(index="Round", columns="Client", values="Accuracy")
        avg_rows.append({**info, **{p: acc[p].mean() for p in names + [SERVER]}})
        last_rows.append({**info, **{p: acc[p].iloc[-1] for p in names + [SERVER]}})
    return pd.DataFrame(avg_rows), pd.DataFrame(last_rows)


def main(cfg=CONFIG):
    out_dir = cfg["out_dir"]
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_client_csvs(load_hospitals(cfg["raw_csv"]), cfg["clients_dir"])

    res = run_federation(cfg, cfg["dp_epsilon"], cfg["max_rounds"])
    names = [c.name for c in res.clients]
    report.run_summary(res, names)

    n = res.rounds_run
    if res.converged:
        print(f"\nStopped at round {n}: all-client convergence (every client passed loss stability, prediction "
              f"stability and global/local agreement for {cfg['patience']} consecutive rounds; seed {cfg['seed']}).")
    else:
        print(f"\nStopped after {n} rounds: maximum rounds reached without all-client convergence (seed {cfg['seed']}).")
    forest_dir = out_dir / "forests"
    forest_dir.mkdir(exist_ok=True)
    save_forest(forest_dir / "global_forest.json", res.server.global_forest, res.server.schema, n)
    for c in res.clients:
        save_forest(forest_dir / f"{c.client_id}_forest.json", c.forest, res.server.schema, n)
    print(f"Returned models saved as explicit tree structures in {forest_dir}: G^({n}) with "
          f"{len(res.server.global_forest)} trees and F_k^({n}) for every client "
          f"({', '.join(f'{c.name} {len(c.forest)}' for c in res.clients)} trees)")
    if cfg["dp_epsilon"] is not None:
        eps = res.schedule
        print(f"Privacy: every released leaf vector epsilon-DP with epsilon = {min(eps):.3f} to {max(eps):.3f}; "
              f"per-patient bound {res.epsilon_per_patient:.2f} for F_k^(0) and {n} supplementary forests "
              f"(basic composition, upper bound; split thresholds are not privatised)")

    res.metrics.to_csv(out_dir / "validation_metrics.csv", index=False)
    res.local.to_csv(out_dir / "local_convergence.csv", index=False)
    res.tuning.to_csv(out_dir / "local_tuning.csv", index=False)
    res.zkp.to_csv(out_dir / "zkp_times.csv", index=False)
    res.transfer.to_csv(out_dir / "tree_transfer_times.csv", index=False)
    res.cost.to_csv(out_dir / "communication_computation_cost.csv", index=False)
    res.rounds.to_csv(out_dir / "rounds.csv", index=False)

    if cfg["make_figures"]:
        from . import plotting  # matplotlib is only needed for figures

        print("\nFigures (600-dpi PNG + vector PDF):")
        for path in plotting.make_all(fig_dir, res, names, cfg):
            print(f"  saved {path} (+ .pdf)")

    sweep = None
    if cfg["run_dp_sweep"]:
        n_budgets = sum(spec is not None for spec in cfg["dp_sweep"])
        print(f"\nAccuracy of G^(r) vs DP budget: {n_budgets} budgets + No DP, {n} rounds each, seed {cfg['seed']} "
              f"(no ZKP; takes a few minutes) ...")
        sweep = run_dp_sweep(cfg, n, names)
        report.dp_sweep_report(*sweep, n)
        pd.concat([sweep[0].assign(Accuracy=f"mean over {n} rounds"),
                   sweep[1].assign(Accuracy=f"after round {n}")]).to_csv(out_dir / "dp_budget_accuracy.csv",
                                                                       index=False)
        if cfg["make_figures"]:
            print(f"  saved {plotting.fig_dp_budget(fig_dir, *sweep, names, n)} (+ .pdf)")
    print(f"\nCSV files in {out_dir} and figures in {fig_dir}")
    return res, sweep
