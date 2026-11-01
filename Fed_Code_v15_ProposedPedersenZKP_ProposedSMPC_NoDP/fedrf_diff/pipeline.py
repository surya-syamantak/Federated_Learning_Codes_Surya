"""Algorithm 1: iterative closed-loop FedRF-Diff with ZKP client authentication and SMPC secure
aggregation (no differential privacy)."""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import smpc, zkp
from . import bounds
from .client import Client
from .config import Config
from .convergence import ConvergenceMonitor
from .data import load_heart_data, make_federated_split
from .metrics import METRICS, metrics_from_counts, metrics_table
from .report import POOLED, format_round
from .server import Server

LABEL_BYTES = 1  # a 0/1 label is sent as one byte
FLOAT_BYTES = 8  # latent coordinates and A_c are float64


@dataclass
class RunResult:
    history: pd.DataFrame
    clients: list[Client]
    server: Server
    model: object  # final M_G
    converged: bool
    rounds_run: int
    rejected: dict[int, list[str]] = field(default_factory=dict)
    # Last round's validation metrics per client + pooled + macro average. The client rows are
    # the evaluation harness (computed on each client, never sent); the pooled row is what the
    # server reconstructs from the SMPC totals.
    final_metrics: pd.DataFrame | None = None
    # Every round's metrics table (as final_metrics, with a "round" column).
    round_metrics: pd.DataFrame | None = None
    # Global-server metrics per round (from the SMPC totals) + L_avg, Delta Z, Delta L.
    global_history: pd.DataFrame | None = None
    # One row per round and client: ZKP outcome, vectors, the three times, bytes.
    zkp_log: pd.DataFrame | None = None
    # One row per round and authenticated client: bounded-update check (bounds.py).
    bounds_log: pd.DataFrame | None = None
    # One row per round and client (and "Global Server"): SMPC times and bytes.
    smpc_log: pd.DataFrame | None = None
    # One row per round and client (and "Global Server"): computation seconds
    # and bytes sent up and down.
    costs: pd.DataFrame | None = None

    def predict_proba(self, client_name: str, X: np.ndarray) -> np.ndarray:
        client = next(c for c in self.clients if c.name == client_name)
        return client.predict_proba(self.model, X)


def pooled_row(totals: dict) -> dict:
    """The Global Server row of the metrics table, from the SMPC totals only."""
    counts = {k: int(round(totals[k])) for k in ("tp", "fp", "tn", "fn")}  # dequantized, exact integers
    n_val = sum(counts.values())
    return {"client": POOLED, "n_val": n_val, **counts,
            "log_loss": totals["loss_x_n_val"] / n_val if n_val else float("nan"),
            **metrics_from_counts(**counts)}


def run(cfg: Config, verbose: bool = True, tamper: dict[int, str] | None = None,
        zkp_enabled: bool = zkp.ENABLED, bounds_enabled: bool = bounds.ENABLED, smpc_enabled: bool = smpc.ENABLED) -> RunResult:
    """Run up to ``cfg.max_rounds`` rounds or until the stopping rule fires.

    No differential privacy: every latent vector is sent as computed. The
    validation losses and confusion counts reach the server only through
    SMPC secure aggregation (``smpc_enabled=False`` sums them in the clear,
    for tests). ``tamper`` maps a round to a client impersonated that round:
    an impostor replays the client's previous authentication messages (and
    sends its own update), and must be rejected.
    """
    log = print if verbose else (lambda *a, **k: None)
    df = load_heart_data(cfg.data_path, cfg.drop_invalid_rows)
    fed = make_federated_split(df, cfg.n_landmarks, cfg.val_fraction, cfg.group_aware_val_split, cfg.seed)
    if cfg.n_landmarks < cfg.d_z:
        raise ValueError("need n_landmarks >= d_z for a unique Procrustes rotation")

    clients = [Client(d, fed.X_landmarks, cfg, i) for i, d in enumerate(fed.clients)]
    sizes = {c.name: c.n_train for c in clients}
    server = Server(cfg.seed, cfg.global_n_trees, cfg.global_min_samples_leaf)
    monitor = ConvergenceMonitor(cfg.eps_z, cfg.eps_l, cfg.patience, cfg.eps_0)
    if zkp_enabled:
        pp = zkp.setup()
        # Client authentication: each client holds its secret; the server keeps C_{t-1} per client.
        provers = {c.name: zkp.ClientProver(pp, c.name) for c in clients}
        verifier = zkp.ServerVerifier(pp)
    # Proposed SMPC, Phases 1-2 (one-time): pairwise keys K_ij and the prime pool.
    aggregator = smpc.SecureAggregator([c.name for c in clients]) if smpc_enabled else None

    log(f"FedRF-Diff | seed {cfg.seed} | up to {cfg.max_rounds} rounds | "
        + (f"Proposed Pedersen ZKP authentication: {zkp.GROUP} group, lambda = {zkp.LAMBDA}" if zkp_enabled
           else "ZKP disabled"))
    log("Clients (train/validation patients): " + ", ".join(
        f"{c.name} {c.n_train}/{len(c.data.y_val)}" for c in clients)
        + f"; public landmarks: {len(fed.X_landmarks)}")
    log("No differential privacy (latents sent without clipping or noise). "
        + (f"SMPC: adaptive prime secure aggregation with self-randomized pairwise masking of every "
           f"client's weighted validation loss and TP/FP/TN/FN counts (pairwise Diffie-Hellman keys, "
           f"prime pool of {smpc.POOL_SIZE} x {smpc.PRIME_BITS}-bit primes, S_q = {smpc.SCALE:g}); the server "
           f"learns only their totals (L_avg, Global Server metrics)."
           if smpc_enabled else "SMPC disabled: losses and counts are summed in the clear."))
    log(f"Stop when Delta Z < {monitor.eps_z:g} and Delta L < {cfg.eps_l:g} for {cfg.patience} rounds.")
    log("Global Server = the global model scored on all clients' validation patients together (SMPC totals); "
        "per-client metrics are the evaluation harness, computed on each client and never sent.")

    rows, rejected, model, converged, final_metrics = [], {}, None, False, None
    bound_rows: list = []
    round_tables, global_rows, zkp_rows, smpc_rows, cost_rows = [], [], [], [], []
    for t in range(1, cfg.max_rounds + 1):
        t0 = time.perf_counter()

        # Stage 1: local RF kernel -> diffusion map (no DP).
        messages = {c.name: c.encode() for c in clients}

        # Stage 1.5: ZKP client authentication before the server accepts an update.
        accepted = dict(messages)
        sessions: dict[str, zkp.Session] = {}
        if zkp_enabled:
            for name, msg in messages.items():
                replay = None
                if tamper and tamper.get(t) == name:
                    # An impostor replays the client's last session and sends its own update.
                    replay = provers[name].last_transcript
                    accepted[name] = replace(msg, Z_tilde=msg.Z_tilde + 1.0, P=msg.P + 1.0)
                sessions[name] = zkp.authenticate(pp, provers[name], verifier, t,
                                                  vectors=len(msg.full_matrix), replay=replay)
                if not sessions[name].authenticated:
                    del accepted[name]
            rejected[t] = [n for n in messages if n not in accepted]
            zkp_rows += [{"round": t, "client": n, "authenticated": s.authenticated,
                          "vectors": s.vectors, "authentication_s": s.authentication_s,
                          "proof_generation_s": s.proof_generation_s,
                          "proof_verification_s": s.proof_verification_s,
                          "bytes_up": s.bytes_up, "bytes_down": s.bytes_down}
                         for n, s in sessions.items()]
        # Stage 1.6: bounded-update enforcement (bounds.py) - every authenticated update must lie
        # inside the public bound that every honest update satisfies, or it is rejected this round.
        bound_info: dict = {}
        if bounds_enabled:
            for name in list(accepted):
                msg = accepted[name]
                bound = bounds.bound_none(len(msg.Z_tilde) + len(msg.P), len(fed.X_landmarks), cfg.d_z)
                bound_info[name] = (bounds.check_update(msg.Z_tilde, msg.P, msg.y, bound), bound)
                if not bound_info[name][0].within:
                    del accepted[name]
            bound_rows += [{"round": t, "client": n, "within": chk.within, "usage_pct": bounds.usage(chk, b),
                            "reasons": "; ".join(chk.reasons), "mode": b.mode, **chk.stats,
                            "bound_entry_max": b.entry_max, "bound_frob_sq_max": b.frob_sq_max,
                            "bound_landmark_norm_max": b.landmark_norm_max,
                            "bound_patient_mean_sq_max": b.patient_mean_sq_max, "check_s": chk.check_s}
                           for n, (chk, b) in bound_info.items()]
        rejected[t] = [n for n in messages if n not in accepted]
        bound_s = sum(chk.check_s for chk, _ in bound_info.values())
        if not accepted:
            log(format_round(t, sessions, None, None, zkp_enabled, bound_checks=bound_info,
                             codec={c.name: c.timing for c in clients}))
            log(f"Round {t}: no update was accepted (ZKP authentication or bounded-update check); stopping.")
            break

        # Stage 2: alignment, aggregation, global model.
        t_server = time.perf_counter()
        if server.P_ref is None:
            ref = cfg.reference_client if cfg.reference_client in accepted else next(iter(accepted))
            server.set_reference(accepted[ref].P)
        A = {name: server.align(msg) for name, msg in accepted.items()}
        Z_G = server.aggregate(accepted, A)
        model, pseudo_labels = server.train_global_model(Z_G, accepted)
        dz = monitor.delta_z(Z_G)
        server_s = time.perf_counter() - t_server
        model_bytes = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))

        # Stage 3: client loss, decoding, local update (authenticated clients only).
        # The metrics stay on the clients; only their SMPC-masked copies leave.
        metrics = {}
        for c in clients:
            if c.name in accepted:
                metrics[c.name] = c.evaluate(model, A[c.name])
                c.decode_and_update(A[c.name], pseudo_labels[c.name])

        # Stage 4: proposed SMPC (Phases 3-8) over the confirmed participants - the clients that
        # passed the ZKP this round - then the stopping rule on the totals.
        vectors = {n: [m["loss"] * sizes[n], m["loss"] * m["n_val"], m["tp"], m["fp"], m["tn"], m["fn"]]
                   for n, m in metrics.items()}
        if aggregator is not None:
            agg = aggregator.aggregate(t, vectors)
            totals, smpc_server_s = agg.totals, agg.aggregate_s
        else:  # in the clear (tests): the same totals without masks
            totals = {f: sum(v[i] for v in vectors.values()) for i, f in enumerate(smpc.FIELDS)}
            agg, smpc_server_s = None, 0.0
        first = aggregator is not None and not smpc_rows  # Phase 1 key setup is charged to the first round
        t_stop = time.perf_counter()
        L_avg = totals["loss_x_n_train"] / sum(sizes[n] for n in metrics)  # sum_c (n_c / sum n) L_c
        dl = monitor.delta_l(L_avg)
        converged = monitor.step(dz, dl)
        server_s += time.perf_counter() - t_stop + smpc_server_s

        # Costs: computation (s) and communication (bytes) per client and server.
        verify_s = sum(s.proof_verification_s for s in sessions.values())
        for c in clients:
            s = sessions.get(c.name)
            p = agg.clients.get(c.name) if agg else None
            ok = c.name in accepted
            setup_s = aggregator.setup_s[c.name] if first else 0.0
            up = (messages[c.name].full_matrix.size * FLOAT_BYTES + c.n_train * LABEL_BYTES
                  + (s.bytes_up if s else 0) + (p.bytes_up if p else 0)
                  + (aggregator.setup_bytes_up if first else 0))
            down = ((s.bytes_down if s else 0) + (p.bytes_down if p else 0)
                    + (aggregator.setup_bytes_down if first else 0)
                    + (model_bytes + A[c.name].size * FLOAT_BYTES + c.n_train * LABEL_BYTES if ok else 0))
            cost_rows.append({
                "round": t, "client": c.name,
                "computation_s": (sum(c.timing.values()) + (s.proof_generation_s if s else 0.0)
                                  + (p.total_s if p else 0.0) + setup_s),
                "encode_s": c.timing.get("encode_s", 0.0),
                "proof_generation_s": s.proof_generation_s if s else 0.0,
                "evaluate_s": c.timing.get("evaluate_s", 0.0),
                "decode_update_s": c.timing.get("decode_update_s", 0.0),
                "decode_s": c.timing.get("decode_s", float("nan")),
                "smpc_s": (p.total_s if p else 0.0) + setup_s,
                "bytes_up": up, "bytes_down": down})
        cost_rows.append({"round": t, "client": "Global Server", "computation_s": server_s + verify_s + bound_s,
                          "bound_check_s": bound_s,
                          "proof_verification_s": verify_s, "smpc_s": smpc_server_s,
                          "learning_s": server_s - smpc_server_s,
                          "bytes_up": sum(r["bytes_down"] for r in cost_rows[-len(clients):]),
                          "bytes_down": sum(r["bytes_up"] for r in cost_rows[-len(clients):])})
        if agg is not None:
            for n, p in agg.clients.items():
                key_s = aggregator.setup_s[n] if first else 0.0
                smpc_rows.append({
                    "round": t, "client": n, "prime_index": agg.prime_index, "prime": str(agg.prime),
                    "values": len(smpc.FIELDS), "key_setup_s": key_s, "mask_s": p.mask_s,
                    "recovery_s": p.recovery_s, "smpc_s": p.total_s + key_s,
                    "bytes_up": p.bytes_up + (aggregator.setup_bytes_up if first else 0),
                    "bytes_down": p.bytes_down + (aggregator.setup_bytes_down if first else 0)})
            smpc_rows.append({"round": t, "client": "Global Server", "prime_index": agg.prime_index,
                              "prime": str(agg.prime), "values": len(smpc.FIELDS), "key_setup_s": 0.0,
                              "mask_s": 0.0, "recovery_s": 0.0, "smpc_s": agg.aggregate_s,
                              "aggregate_s": agg.aggregate_s, "bytes_up": 0,
                              "bytes_down": sum(len(v) for v in agg.masked.values()) * smpc.ELEMENT_BYTES})

        row = {"round": t, "delta_z": dz, "L_avg": L_avg, "delta_l": dl,
               "patience": monitor.counter, "n_accepted": len(accepted),
               "n_out_of_bound": sum(not chk.within for chk, _ in bound_info.values()),
               "zkp_seconds": sum(s.authentication_s for s in sessions.values()),
               "smpc_seconds": (sum(p.total_s for p in agg.clients.values()) + agg.aggregate_s) if agg else 0.0,
               "smpc_prime": str(agg.prime) if agg else "",
               "round_seconds": time.perf_counter() - t0, "delta_z_threshold": monitor.eps_z}
        for n, m in metrics.items():  # evaluation harness (per client, never sent)
            row[f"loss[{n}]"] = m["loss"]
            for k in METRICS:
                row[f"{k}[{n}]"] = m[k]
        final_metrics = metrics_table(metrics)
        # The Global Server row comes from the SMPC totals, not from the clients' own counts.
        pooled = pooled_row(totals)
        idx = final_metrics.index[final_metrics["client"] == POOLED][0]
        for k, v in pooled.items():
            final_metrics.loc[idx, k] = v
        for k in METRICS:
            row[f"{k}[all]"] = pooled[k]
        rows.append(row)
        round_tables.append(final_metrics.assign(round=t))
        global_rows.append({"round": t, **{k: int(pooled[k]) for k in ("n_val", "tp", "fp", "tn", "fn")},
                            **{k: pooled[k] for k in ("log_loss", *METRICS)},
                            "L_avg": L_avg, "delta_z": dz, "delta_l": dl})

        log(format_round(t, sessions, final_metrics, row, zkp_enabled, bound_checks=bound_info,
                         codec={c.name: c.timing for c in clients},
                         eps_z=monitor.eps_z, patience=cfg.patience,
                         smpc=agg))
        if converged:
            log(f"Converged at round {t}.")
            break

    round_metrics = pd.concat(round_tables, ignore_index=True) if round_tables else None
    return RunResult(history=pd.DataFrame(rows), clients=clients, server=server, model=model,
                     converged=converged, rounds_run=len(rows),
                     rejected={t: r for t, r in rejected.items() if r},
                     final_metrics=final_metrics, round_metrics=round_metrics,
                     global_history=pd.DataFrame(global_rows),
                     zkp_log=pd.DataFrame(zkp_rows) if zkp_rows else None,
                     bounds_log=pd.DataFrame(bound_rows) if bound_rows else None,
                     smpc_log=pd.DataFrame(smpc_rows) if smpc_rows else None,
                     costs=pd.DataFrame(cost_rows))
