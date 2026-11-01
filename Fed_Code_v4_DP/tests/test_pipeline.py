import json

import numpy as np
import pandas as pd
import pytest

from fedrf_diff import Config, privacy, run, zkp
from fedrf_diff.plotting import plot_convergence

CLIENTS = ["Client 1(Hungary)", "Client 2(Switzerland)", "Client 3(VA Long Beach)", "Client 4(Cleveland)"]


def small_config(**overrides) -> Config:
    base = dict(n_trees=30, min_samples_leaf=10, n_landmarks=40, max_rounds=3, seed=1,
                global_n_trees=50)
    base.update(overrides)
    return Config(**base)


def test_pipeline_runs_with_dp_and_records_everything(tmp_path):
    result = run(small_config(), verbose=False)
    h = result.history
    for c in result.clients:
        # Patient vectors: clipped to this round's C, then N(0, sigma^2) noise;
        # landmarks: clipped only.
        C = c.dp.last_clip
        sigma = c.dp.noise_ratio * C
        clipped = privacy.clip_rows(c.Z, C)
        noise = c.Z_tilde[: c.n_train] - clipped[: c.n_train]
        assert abs(noise.std() / sigma - 1) < 0.1
        np.testing.assert_allclose(c.Z_tilde[c.n_train:], clipped[c.n_train:])
        assert c.dp.releases == 3
    spent = privacy.epsilon_for_rho(3 * privacy.rho_for_epsilon(privacy.EPSILON, privacy.DELTA),
                                    privacy.DELTA)
    assert h["epsilon_spent"].iloc[-1] == pytest.approx(spent)
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

    proba = result.predict_proba(CLIENTS[0], result.clients[0].data.X_val[:5])
    assert proba.shape == (5,) and np.all((proba >= 0) & (proba <= 1))
    plot_convergence(h, 0.5, 0.02, None, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 0 and (tmp_path / "c.pdf").exists()


def test_zkp_binding_rejects_tampering_client(monkeypatch):
    monkeypatch.setattr(zkp, "BIND_PLAINTEXT", True)
    result = run(small_config(max_rounds=2), verbose=False, tamper={2: CLIENTS[2]})
    assert result.rejected == {2: [CLIENTS[2]]}
    z = result.zkp_log.set_index(["round", "client"])
    assert not z.loc[(2, CLIENTS[2]), "authenticated"] and z["authenticated"].sum() == 7
    from fedrf_diff.report import format_end_of_run
    assert "Authentications: 7/8 successful, 1 failed" in format_end_of_run(result)
    assert list(result.history["n_accepted"]) == [4, 3]


def test_zkp_does_not_change_learning():
    a = run(small_config(max_rounds=2), verbose=False, zkp_enabled=False).history
    b = run(small_config(max_rounds=2), verbose=False).history
    np.testing.assert_allclose(a["L_glob"], b["L_glob"])


def test_no_dp_run_sends_vectors_unchanged():
    result = run(small_config(max_rounds=2), verbose=False, epsilon=None, zkp_enabled=False)
    for c in result.clients:
        np.testing.assert_array_equal(c.Z_tilde, c.Z)
        assert c.dp.spent_epsilon() == 0.0


def test_cli_prints_rounds_averages_zkp_analysis_and_writes_figures(tmp_path, capsys, monkeypatch):
    from fedrf_diff import __main__ as cli

    monkeypatch.setattr(privacy, "EPSILON_SWEEP", (5.0, privacy.EPSILON, None))
    cli.main(["--n-trees", "20", "--min-samples-leaf", "10", "--n-landmarks", "40",
              "--global-n-trees", "30", "--max-rounds", "2", "--output-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "seed 42" in out
    for t in (1, 2):
        assert f"Round {t} " in out
    for heading in ("Accuracy", "Precision", "Recall", "Specificity", "F1 Score",
                    "Matthews Correlation Coefficient (MCC)", "Cohen's Kappa",
                    "ZKP Authentication Time (s)", "ZKP Proof Generation Time (s)",
                    "ZKP Proof Verification Time (s)", *CLIENTS, "Global Server"):
        assert heading in out
    assert out.count("Successful") == 2 * 4
    assert "Metrics averaged over all 2 rounds" in out and "ZKP time analysis over 2 rounds" in out
    assert "Min (s)" in out and "Max (s)" in out and "Mean (s)" in out
    assert "Accuracy vs DP budget" in out

    sweep = pd.read_csv(tmp_path / "dp_budget_sweep.csv")
    assert len(sweep) == 3 and (sweep["rounds"] == 2).all()
    for prefix in ("mean_accuracy", "accuracy"):
        assert sweep[f"{prefix}[All clients (pooled)]"].between(0, 1).all()
    for name in ("history", "final_metrics", "global_server_metrics", "costs", "zkp_times"):
        assert (tmp_path / f"{name}.csv").exists()
    for fig in ("convergence", "zkp_times", "accuracy_vs_dp_budget", "computation_cost",
                "communication_cost"):
        assert (tmp_path / "figures" / f"{fig}.png").stat().st_size > 10_000
        assert (tmp_path / "figures" / f"{fig}.pdf").exists()
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["privacy"]["epsilon_per_round"] == privacy.EPSILON and cfg["zkp"]["group"] == "test512"


def test_cli_rejects_dp_and_zkp_options(capsys):
    from fedrf_diff.__main__ import main

    with pytest.raises(SystemExit):
        main(["--sigma", "0.1"])
    assert "privacy.py" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--zkp-group", "test512"])
    assert "zkp.py" in capsys.readouterr().err
