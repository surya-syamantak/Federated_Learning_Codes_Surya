"""Algorithm 1: iterative closed-loop FedRF-Diff with Gaussian DP and ZKP authentication."""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import privacy, zkp
from .client import Client
from .config import Config
from .convergence import ConvergenceMonitor, weighted_loss
from .data import load_heart_data, make_federated_split
from .metrics import METRICS, metrics_table
from .report import POOLED, format_round
from .server import Server

LABEL_BYTES = 1  # a 0/1 label is sent as one byte
FLOAT_BYTES = 8  # latent coordinates, losses and A_c are float64
METRIC_MESSAGE_BYTES = FLOAT_BYTES + 4 * 4  # log-loss + TP/FP/TN/FN (int32)


@dataclass
class RunResult:
    history: pd.DataFrame
    clients: list[Client]
    server: Server
    model: object  # final M_G
    converged: bool
    rounds_run: int
    epsilon: float | None  # per-round DP budget (None = no DP)
    rejected: dict[int, list[str]] = field(default_factory=dict)
    # Last round's validation metrics per client + pooled + macro average.
    final_metrics: pd.DataFrame | None = None
    # Every round's metrics table (as final_metrics, with a "round" column).
    round_metrics: pd.DataFrame | None = None
    # Global-server metrics per round (pooled over clients) + L_avg, Delta Z, Delta L.
    global_history: pd.DataFrame | None = None
    # One row per round and client: ZKP outcome, vectors, the three times, bytes.
    zkp_log: pd.DataFrame | None = None
    # One row per round and client (and "Global Server"): computation seconds,
    # bytes sent up and down, and the privacy budget spent so far.
    costs: pd.DataFrame | None = None

    def predict_proba(self, client_name: str, X: np.ndarray) -> np.ndarray:
        client = next(c for c in self.clients if c.name == client_name)
        return client.predict_proba(self.model, X)


def run(cfg: Config, verbose: bool = True, tamper: dict[int, str] | None = None,
        epsilon: float | None = privacy.EPSILON, zkp_enabled: bool = zkp.ENABLED) -> RunResult:
    """Run up to ``cfg.max_rounds`` rounds or until the stopping rule fires.

    ``epsilon`` is the per-round DP budget of each transmitted vector
    (privacy.EPSILON by default; None = no DP). ``tamper`` maps a round to a
    client that transmits a different Z~ from the one it committed to.
    """
    log = print if verbose else (lambda *a, **k: None)
    df = load_heart_data(cfg.data_path, cfg.drop_invalid_rows)
    fed = make_federated_split(df, cfg.n_landmarks, cfg.val_fraction, cfg.group_aware_val_split, cfg.seed)
    if cfg.n_landmarks < cfg.d_z:
        raise ValueError("need n_landmarks >= d_z for a unique Procrustes rotation")

    clients = [Client(d, fed.X_landmarks, cfg, i, epsilon) for i, d in enumerate(fed.clients)]
    sizes = {c.name: c.n_train for c in clients}
    server = Server(cfg.seed, cfg.global_n_trees, cfg.global_min_samples_leaf)
    mech = clients[0].dp
    monitor = ConvergenceMonitor(cfg.eps_z, cfg.eps_l, cfg.patience, cfg.eps_0,
                                 sigma=mech.noise_ratio, clip_norm=1.0)  # threshold uses sigma / C
    if zkp_enabled:
        pp = zkp.setup(cfg.d_z)
        provers = {c.name: zkp.Prover(pp) for c in clients}
        verifier = zkp.Verifier(pp)

    log(f"FedRF-Diff | seed {cfg.seed} | up to {cfg.max_rounds} rounds | "
        + (f"ZKP: {zkp.GROUP} group, lambda = {zkp.LAMBDA}" if zkp_enabled else "ZKP disabled"))
    log("Clients (train/validation patients): " + ", ".join(
        f"{c.name} {c.n_train}/{len(c.data.y_val)}" for c in clients)
        + f"; public landmarks: {len(fed.X_landmarks)}")
    if mech.enabled:
        log(f"Gaussian DP: epsilon = {epsilon:g} per round, delta = {mech.delta:g}, clip norm C = "
            f"{mech.clip_norm:g}, sigma = {mech.sigma:.4f} (privacy.py); stop when "
            f"Delta Z < {monitor.eps_z:.3f} and Delta L < {cfg.eps_l:g} for {cfg.patience} rounds")
    else:
        log(f"No DP noise; stop when Delta Z < {monitor.eps_z:.3f} and Delta L < {cfg.eps_l:g} "
            f"for {cfg.patience} rounds")
    log("Global Server = the global model scored on all clients' validation patients together.")

    rows, rejected, model, converged, final_metrics = [], {}, None, False, None
    round_tables, global_rows, zkp_rows, cost_rows = [], [], [], []
    for t in range(1, cfg.max_rounds + 1):
        t0 = time.perf_counter()

        # Stage 1: local RF kernel -> diffusion map -> Gaussian DP.
        messages = {c.name: c.encode() for c in clients}

        # Stage 1.5: ZKP authentication before the server accepts Z~_c.
        accepted = dict(messages)
        sessions: dict[str, zkp.Session] = {}
        if zkp_enabled:
            for name, msg in messages.items():
                Z_sent = None
                if tamper and tamper.get(t) == name:
                    Z_sent = msg.full_matrix + 1.0
                    # The server receives (and would aggregate) the altered matrix.
                    accepted[name] = replace(msg, Z_tilde=Z_sent[: msg.n], P=Z_sent[msg.n :])
                sessions[name] = zkp.authenticate(provers[name], verifier, name, msg.full_matrix,
                                                  Z_sent=Z_sent)
                if not sessions[name].authenticated:
                    del accepted[name]
            rejected[t] = [n for n in messages if n not in accepted]
            zkp_rows += [{"round": t, "client": n, "authenticated": s.authenticated,
                          "vectors": s.vectors, "authentication_s": s.authentication_s,
                          "proof_generation_s": s.proof_generation_s,
                          "proof_verification_s": s.proof_verification_s,
                          "bytes_up": s.bytes_up, "bytes_down": s.bytes_down}
                         for n, s in sessions.items()]
        if not accepted:
            log(format_round(t, sessions, None, None, zkp_enabled))
            log(f"Round {t}: every client failed authentication; stopping.")
            break

        # Stage 2: alignment, aggregation, global model.
        t_server = time.perf_counter()
        if server.P_ref is None:
            ref = cfg.reference_client if cfg.reference_client in accepted else next(iter(accepted))
            server.set_reference(accepted[ref].P)
        A = {name: server.align(msg) for name, msg in accepted.items()}
        Z_G = server.aggregate(accepted, A)
        model, pseudo_labels = server.train_global_model(Z_G, accepted)
        dz = monitor.delta_z(Z_G, {c.name: c.sigma_sent for c in clients if c.name in accepted})
        server_s = time.perf_counter() - t_server
        model_bytes = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))

        # Stage 3: client loss, decoding, local update (authenticated clients only).
        metrics = {}
        for c in clients:
            if c.name in accepted:
                metrics[c.name] = c.evaluate(model, A[c.name])
                c.decode_and_update(A[c.name], pseudo_labels[c.name])

        # Stage 4: aggregated loss and the stopping rule.
        t_stop = time.perf_counter()
        L_avg = weighted_loss({n: m["loss"] for n, m in metrics.items()}, sizes)
        dl = monitor.delta_l(L_avg)
        converged = monitor.step(dz, dl)
        server_s += time.perf_counter() - t_stop

        # Costs: computation (s) and communication (bytes) per client and server.
        verify_s = sum(s.proof_verification_s for s in sessions.values())
        for c in clients:
            s = sessions.get(c.name)
            ok = c.name in accepted
            up = (messages[c.name].full_matrix.size * FLOAT_BYTES + c.n_train * LABEL_BYTES
                  + (METRIC_MESSAGE_BYTES if ok else 0) + (s.bytes_up if s else 0))
            down = (s.bytes_down if s else 0) + (model_bytes + A[c.name].size * FLOAT_BYTES
                                                 + c.n_train * LABEL_BYTES if ok else 0)
            cost_rows.append({
                "round": t, "client": c.name,
                "computation_s": sum(c.timing.values()) + (s.proof_generation_s if s else 0.0),
                "encode_dp_s": c.timing.get("encode_s", 0.0),
                "proof_generation_s": s.proof_generation_s if s else 0.0,
                "evaluate_s": c.timing.get("evaluate_s", 0.0),
                "decode_update_s": c.timing.get("decode_update_s", 0.0),
                "bytes_up": up, "bytes_down": down,
                "epsilon_spent": c.dp.spent_epsilon(),
                "clip_norm": c.dp.last_clip if c.dp.enabled else None,
                "sigma": c.sigma_sent})
        cost_rows.append({"round": t, "client": "Global Server", "computation_s": server_s + verify_s,
                          "proof_verification_s": verify_s, "learning_s": server_s,
                          "bytes_up": sum(r["bytes_down"] for r in cost_rows[-len(clients):]),
                          "bytes_down": sum(r["bytes_up"] for r in cost_rows[-len(clients):]),
                          "epsilon_spent": mech.spent_epsilon()})

        row = {"round": t, "delta_z": dz, "L_avg": L_avg, "delta_l": dl,
               "patience": monitor.counter, "n_accepted": len(accepted),
               "zkp_seconds": sum(s.authentication_s for s in sessions.values()),
               "round_seconds": time.perf_counter() - t0, "epsilon_spent": mech.spent_epsilon()}
        for n, m in metrics.items():
            row[f"loss[{n}]"] = m["loss"]
            for k in METRICS:
                row[f"{k}[{n}]"] = m[k]
        final_metrics = metrics_table(metrics)
        pooled = final_metrics.loc[final_metrics["client"] == POOLED].iloc[0]
        for k in METRICS:
            row[f"{k}[all]"] = pooled[k]
        rows.append(row)
        round_tables.append(final_metrics.assign(round=t))
        global_rows.append({"round": t, **{k: int(pooled[k]) for k in ("n_val", "tp", "fp", "tn", "fn")},
                            **{k: pooled[k] for k in ("log_loss", *METRICS)},
                            "L_avg": L_avg, "delta_z": dz, "delta_l": dl,
                            "epsilon_spent": mech.spent_epsilon()})

        log(format_round(t, sessions, final_metrics, row, zkp_enabled,
                         eps_z=monitor.eps_z, patience=cfg.patience,
                         epsilon=epsilon, delta=mech.delta))
        if converged:
            log(f"Converged at round {t}.")
            break

    round_metrics = pd.concat(round_tables, ignore_index=True) if round_tables else None
    return RunResult(history=pd.DataFrame(rows), clients=clients, server=server, model=model,
                     converged=converged, rounds_run=len(rows), epsilon=epsilon,
                     rejected={t: r for t, r in rejected.items() if r},
                     final_metrics=final_metrics, round_metrics=round_metrics,
                     global_history=pd.DataFrame(global_rows),
                     zkp_log=pd.DataFrame(zkp_rows) if zkp_rows else None,
                     costs=pd.DataFrame(cost_rows))


def sweep_point(cfg: Config, epsilon: float | None, rounds: int) -> dict:
    """One learning run for the accuracy-vs-DP-budget figure (no ZKP, fixed rounds).

    The ZKP does not change any learned value, so it is skipped here. Every
    budget runs for the same number of rounds as the main run.
    """
    result = run(replace(cfg, max_rounds=rounds, patience=10**9), verbose=False,
                 epsilon=epsilon, zkp_enabled=False)
    return sweep_row(result)


def sweep_row(result: RunResult) -> dict:
    """Budget, noise and accuracy (mean over the rounds and final round) of one run.

    With one seed the final-round accuracy moves a few points from round to
    round, so the figure uses the mean over the rounds; both are saved.
    """
    names = [c for c in result.final_metrics["client"] if c != "Macro average"]
    final = result.final_metrics.set_index("client")["accuracy"]
    mean = result.round_metrics.groupby("client")["accuracy"].mean()
    mech = result.clients[0].dp
    return {"epsilon_per_round": result.epsilon, "sigma": mech.sigma,
            "epsilon_spent": mech.spent_epsilon(), "rounds": result.rounds_run,
            **{f"mean_accuracy[{c}]": mean[c] for c in names},
            **{f"accuracy[{c}]": final[c] for c in names}}
