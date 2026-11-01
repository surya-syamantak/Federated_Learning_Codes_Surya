import json
from pathlib import Path

import numpy as np
import pytest

from fedrf_diff import Config, run
from fedrf_diff.plotting import plot_convergence

CLIENTS = ["Client 1(Hungary)", "Client 2(Switzerland)", "Client 3(VA Long Beach)", "Client 4(Cleveland)"]


def small_config(**overrides) -> Config:
    base = dict(n_trees=30, min_samples_leaf=10, n_landmarks=40, max_rounds=3, seed=1,
                global_n_trees=50)
    base.update(overrides)
    return Config(**base)


def test_pipeline_runs_without_dp_and_records_everything(tmp_path):
    result = run(small_config(), verbose=False)
    h = result.history
    # No DP: every latent vector (patients and landmarks) is sent exactly as computed.
    for c in result.clients:
        np.testing.assert_array_equal(c.Z_tilde, c.Z)
        assert not hasattr(c, "dp")
    assert not any(col.startswith(("epsilon", "laplace")) for col in h.columns)
    assert (h["delta_z_threshold"] == small_config().eps_z).all()
    assert list(h["round"]) == [1, 2, 3] and (h["n_accepted"] == 4).all()
    for metric in ("accuracy", "precision", "recall", "specificity", "f1", "mcc", "kappa"):
        assert f"{metric}[{CLIENTS[0]}]" in h and f"{metric}[all]" in h
    assert list(result.final_metrics["client"]) == [*CLIENTS, "All clients (pooled)", "Macro average"]

    z = result.zkp_log
    assert len(z) == 3 * 4 and z["authenticated"].all()
    for col in ("authentication_s", "proof_generation_s", "proof_verification_s"):
        assert (z[col] > 0).all()
    assert (z["authentication_s"] >= z["proof_verification_s"]).all()
    assert (z["authentication_s"] >= z["proof_generation_s"]).all()

    costs = result.costs
    assert set(costs["client"]) == {*CLIENTS, "Global Server"} and len(costs) == 3 * 5
    assert (costs["computation_s"] > 0).all() and (costs["bytes_up"] > 0).all()
    assert (costs[costs["client"] != "Global Server"]["bytes_down"] > 0).all()
    assert "epsilon_spent" not in costs

    proba = result.predict_proba(CLIENTS[0], result.clients[0].data.X_val[:5])
    assert proba.shape == (5,) and np.all((proba >= 0) & (proba <= 1))
    plot_convergence(h, 0.15, 0.02, None, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 0 and (tmp_path / "c.pdf").exists()


def test_no_privacy_module():
    assert not (Path(__file__).resolve().parent.parent / "fedrf_diff" / "privacy.py").exists()


def test_impostor_replaying_a_session_is_rejected():
    result = run(small_config(max_rounds=2), verbose=False, tamper={2: CLIENTS[2]})
    assert result.rejected == {2: [CLIENTS[2]]}
    z = result.zkp_log.set_index(["round", "client"])
    assert not z.loc[(2, CLIENTS[2]), "authenticated"] and z["authenticated"].sum() == 7
    from fedrf_diff.report import format_end_of_run
    assert "ZKP authentications: 7/8 successful (updates accepted), 1 failed" in format_end_of_run(result)
    assert list(result.history["n_accepted"]) == [4, 3]


def test_zkp_does_not_change_learning():
    a = run(small_config(max_rounds=2), verbose=False, zkp_enabled=False).history
    b = run(small_config(max_rounds=2), verbose=False).history
    np.testing.assert_allclose(a["L_glob"], b["L_glob"])


def test_cli_prints_rounds_averages_zkp_analysis_and_writes_figures(tmp_path, capsys):
    from fedrf_diff import __main__ as cli

    cli.main(["--n-trees", "20", "--min-samples-leaf", "10", "--n-landmarks", "40",
              "--global-n-trees", "30", "--max-rounds", "2", "--output-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "seed 42" in out and "No differential privacy" in out
    for t in (1, 2):
        assert f"Round {t} " in out
    for heading in ("Accuracy", "Precision", "Recall", "Specificity", "F1 Score",
                    "Matthews Correlation Coefficient (MCC)", "Cohen's Kappa",
                    "ZKP Authentication Time (ms)", "ZKP Proof Generation Time (ms)",
                    "ZKP Proof Verification Time (ms)", "Encoding Time (ms)", "Decoding Time (ms)",
                    *CLIENTS, "Global Server"):
        assert heading in out
    assert out.count(" Valid ") == 2 * 4 and "ZKP authentications: 8/8 successful" in out
    assert "Metrics averaged over all 2 rounds" in out and "ZKP time analysis over 2 rounds" in out
    assert out.count("Encoding and decoding times (per client):") == 2
    assert "Encoding and decoding time analysis over 2 rounds" in out
    assert "Min (ms)" in out and "Max (ms)" in out and "Mean (ms)" in out
    assert "epsilon" not in out.lower() and "DP budget" not in out
    assert out.count("Authentication Successful (proof valid, update accepted)") == 2 * 4
    assert "=== ZKP authentication summary ===" in out and "Successful in every round" in out
    assert "SMPC: adaptive prime secure aggregation with self-randomized pairwise masking (4 participants)" in out
    assert "learned only the totals" in out and "p_r = Pool[" in out
    assert "SMPC time analysis over 2 rounds" in out and "SMPC Masking Time (ms)" in out
    assert "One-time pairwise key setup" in out and "Adaptive prime p_r:" in out
    assert out.count("  saved ") == 5

    for name in ("history", "final_metrics", "global_server_metrics", "costs", "zkp_times", "smpc_times"):
        assert (tmp_path / f"{name}.csv").exists()
    assert not (tmp_path / "dp_budget_sweep.csv").exists()
    for fig in ("convergence", "zkp_times", "smpc_times", "computation_cost", "communication_cost"):
        assert (tmp_path / "figures" / f"{fig}.png").stat().st_size > 10_000
        assert (tmp_path / "figures" / f"{fig}.pdf").exists()
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["privacy"] == {"mechanism": "none"} and cfg["zkp"]["group"] == "test512"
    assert cfg["smpc"]["enabled"] and len(cfg["smpc"]["pool"]) == 3 and cfg["smpc"]["scale_S_q"] == 10**6


def test_cli_rejects_dp_and_zkp_options(capsys):
    from fedrf_diff.__main__ import main

    with pytest.raises(SystemExit):
        main(["--epsilon", "5"])
    assert "no differential privacy" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--zkp-group", "test512"])
    assert "zkp.py" in capsys.readouterr().err


def test_smpc_secure_aggregation_gives_the_server_only_exact_totals():
    from fedrf_diff import smpc
    from fedrf_diff.convergence import weighted_loss

    with_smpc = run(small_config(max_rounds=2), verbose=False)
    clear = run(small_config(max_rounds=2), verbose=False, smpc_enabled=False)
    # SMPC changes no learned value: the same losses, metrics and stopping decisions.
    np.testing.assert_allclose(with_smpc.history["L_glob"], clear.history["L_glob"], rtol=0, atol=1e-9)
    for k in ("accuracy[all]", "mcc[all]", "kappa[all]", "delta_z"):
        np.testing.assert_allclose(with_smpc.history[k].astype(float), clear.history[k].astype(float),
                                   equal_nan=True)
    # The Global Server row (from the SMPC totals) equals the sum of the clients' own counts.
    h = with_smpc.history.iloc[-1]
    table = with_smpc.final_metrics.set_index("client")
    pooled = table.loc["All clients (pooled)"]
    for k in ("tp", "fp", "tn", "fn"):
        assert int(pooled[k]) == int(table.loc[CLIENTS, k].sum())
    sizes = {c.name: c.n_train for c in with_smpc.clients}
    losses = {c: float(h[f"loss[{c}]"]) for c in CLIENTS}
    assert h["L_glob"] == pytest.approx(weighted_loss(losses, sizes), abs=1e-9)
    # Log and costs: one aggregation per round with the round's adaptive prime; every client sends
    # one masked vector (6 values x 8 bytes) and nothing else after the one-time key setup.
    log = with_smpc.smpc_log
    clients = log[log["client"] != "Global Server"]
    assert len(clients) == 2 * 4
    pool = smpc.prime_pool()
    for t, g in log.groupby("round"):
        idx, p = smpc.select_prime(int(t), pool)
        assert (g["prime_index"] == idx).all() and (g["prime"].astype(str) == str(p)).all()
    m = len(smpc.FIELDS) * smpc.ELEMENT_BYTES
    r1, r2 = clients[clients["round"] == 1], clients[clients["round"] == 2]
    assert (r2["bytes_up"] == m).all() and (r2["bytes_down"] == 0).all() and (r2["key_setup_s"] == 0).all()
    assert (r1["key_setup_s"] > 0).all() and (r1["bytes_up"] > m).all() and (r1["bytes_down"] > 0).all()
    costs = with_smpc.costs
    assert (costs[costs["client"] != "Global Server"]["smpc_s"] > 0).all()
    assert clear.smpc_log is None and (clear.costs["smpc_s"] == 0).all()


def test_rejected_client_takes_no_part_in_the_aggregation():
    result = run(small_config(max_rounds=2), verbose=False, tamper={2: CLIENTS[2]})
    log = result.smpc_log
    round2 = log[(log["round"] == 2) & (log["client"] != "Global Server")]
    assert sorted(round2["client"]) == sorted(set(CLIENTS) - {CLIENTS[2]})
    # The masks were built over the 3 confirmed participants, so they cancel without recovery
    # and the Global Server totals are exactly the 3 participants' sums.
    g = result.global_history.set_index("round").loc[2]
    table = result.round_metrics[(result.round_metrics["round"] == 2)].set_index("client")
    participants = [c for c in CLIENTS if c != CLIENTS[2]]
    for k in ("tp", "fp", "tn", "fn"):
        assert int(g[k]) == int(table.loc[participants, k].sum())
