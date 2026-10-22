import numpy as np
import pytest

from fedrf_diff import zkp
from fedrf_diff.alignment import orthogonal_procrustes
from fedrf_diff.config import DEFAULT_DATA_PATH
from fedrf_diff.convergence import ConvergenceMonitor, relative_change, weighted_loss
from fedrf_diff.data import CLIENT_NAMES, load_heart_data, make_federated_split, numeric_mask
from fedrf_diff.decoder import KNNDecoder, leaf_intersection_boxes, rf_compatible_points
from fedrf_diff.rf_kernel import ForestEncoder, train_forest


@pytest.fixture(scope="module")
def df():
    return load_heart_data(str(DEFAULT_DATA_PATH))


@pytest.fixture(scope="module")
def fed(df):
    return make_federated_split(df, n_landmarks=30, val_fraction=0.2, group_aware=True, seed=0)


@pytest.fixture(scope="module")
def forest_and_X(fed):
    c = fed.clients[0]
    return train_forest(c.X_train, c.y_train, 50, 5, "sqrt", 0), c.X_train


# --- data -------------------------------------------------------------------------
def test_load_keeps_duplicates_and_drops_two_invalid_rows(df):
    assert len(df) == 1188
    assert df.duplicated(subset=["patient_id"]).sum() == 272
    assert (df["resting bp s"] > 0).all() and (df["ST slope"] > 0).all()


def test_four_hospital_clients_with_statlog_in_cleveland(df):
    sizes = df.groupby("hospital").size().to_dict()
    assert sizes == {"Client 1(Hungary)": 294, "Client 2(Switzerland)": 123,
                     "Client 3(VA Long Beach)": 198, "Client 4(Cleveland)": 573}


def test_split_has_no_patient_overlap_and_excludes_landmarks(fed):
    assert [c.name for c in fed.clients] == list(CLIENT_NAMES)
    landmarks = set(fed.landmark_patient_ids)
    assert len(landmarks) == len(fed.X_landmarks) == 30
    for c in fed.clients:
        train, val = set(c.train_patient_ids), set(c.val_patient_ids)
        assert not train & val
        assert not (train | val) & landmarks


# --- RF kernel and diffusion map ------------------------------------------------------
def test_rf_kernel_is_symmetric_doubly_stochastic_psd(forest_and_X):
    forest, X = forest_and_X
    K = ForestEncoder(forest, X, d_z=3, tau=1.0).kernel()
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    np.testing.assert_allclose(K.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(K.sum(axis=0), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-10


def test_diffusion_map_and_nystrom_agree_on_reference_rows(forest_and_X):
    forest, X = forest_and_X
    enc = ForestEncoder(forest, X, d_z=3, tau=1.0).fit()
    assert enc.Z.shape == (len(X), 3)
    assert np.all(np.diff(enc.eigenvalues) <= 1e-12) and enc.eigenvalues.max() < 1.0
    np.testing.assert_allclose(enc.transform(X), enc.Z, atol=1e-8)
    np.testing.assert_allclose(enc.cross_kernel(X[:5]).sum(axis=1), 1.0)


# --- alignment ----------------------------------------------------------------
def test_procrustes_recovers_rotation_and_reflection():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(40, 4))
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    Q[:, 0] *= -1
    A = orthogonal_procrustes(P, P @ Q)
    np.testing.assert_allclose(A, Q, atol=1e-10)
    np.testing.assert_allclose(A.T @ A, np.eye(4), atol=1e-10)


# --- decoder ------------------------------------------------------------------
def test_leaf_boxes_contain_their_sample(forest_and_X):
    forest, X = forest_and_X
    lo, hi = leaf_intersection_boxes(forest, X, X.min(axis=0), X.max(axis=0))
    assert np.all(lo <= X + 1e-9) and np.all(X <= hi + 1e-9)


def test_decoder_outputs_valid_categories_and_exact_match(forest_and_X):
    forest, X = forest_and_X
    numeric = numeric_mask()
    X_tilde = rf_compatible_points(forest, X, numeric, X.min(axis=0), X.max(axis=0))
    Z = ForestEncoder(forest, X, d_z=3, tau=1.0).fit().Z
    X_hat = KNNDecoder(Z, X_tilde, numeric, k=10).decode(Z)
    for f in np.flatnonzero(~numeric):
        assert set(np.unique(X_hat[:, f])) <= set(np.unique(X[:, f]))
    # A query equal to a reference row reproduces that row's x~ (or a duplicate's).
    unique = np.unique(Z, axis=0, return_index=True)[1]
    np.testing.assert_allclose(X_hat[unique][:, numeric], X_tilde[unique][:, numeric], atol=1e-6)


# --- convergence --------------------------------------------------------------
def test_monitor_needs_k_consecutive_rounds():
    m = ConvergenceMonitor(eps_z=0.1, eps_l=0.01, patience=2, eps_0=1e-12)
    assert not m.step(None, None)
    assert not m.step(0.05, 0.001)
    assert not m.step(0.5, 0.001)  # resets
    assert not m.step(0.05, 0.001)
    assert m.step(0.05, 0.001)


def test_delta_z_and_weighted_loss():
    assert relative_change(np.ones((2, 2)) * 1.1, np.ones((2, 2)), 0.0) == pytest.approx(0.1)
    assert weighted_loss({"a": 1.0, "b": 0.0}, {"a": 3, "b": 1}) == pytest.approx(0.75)


# --- ZKP ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def pp():
    return zkp.setup(d_z=3, group="test512")


def test_generators_have_order_q(pp):
    for g in (*pp.g, pp.h):
        assert pow(int(g), int(pp.q), int(pp.p)) == 1 and g != 1


def test_honest_client_passes_chained_rounds(pp):
    rng = np.random.default_rng(0)
    prover, verifier = zkp.Prover(pp), zkp.Verifier(pp)
    for _ in range(3):
        Z = rng.normal(size=(5, 3))
        s = zkp.authenticate(prover, verifier, "c", Z)
        assert s.authenticated and s.vectors == 5
        assert s.authentication_s >= s.proof_generation_s + s.proof_verification_s - 1e-9
        assert zkp.authenticate(zkp.Prover(pp), zkp.Verifier(pp), "x", Z, bind_plaintext=True).authenticated


def test_forged_response_fails(pp):
    prover, verifier = zkp.Prover(pp), zkp.Verifier(pp)
    Z = np.random.default_rng(1).normal(size=(4, 3))
    C = prover.commit(Z)
    T, _ = prover.announce()
    e = verifier.challenge(len(C))
    responses = prover.respond(e)
    responses[2].s[0] = (responses[2].s[0] + 1) % pp.q
    assert not verifier.verify("c", C, T, e, responses)


def test_swapped_plaintext_only_caught_with_binding(pp):
    Z = np.random.default_rng(2).normal(size=(4, 3))
    sent = Z + 1.0
    assert zkp.authenticate(zkp.Prover(pp), zkp.Verifier(pp), "c", Z, bind_plaintext=False, Z_sent=sent).authenticated
    assert not zkp.authenticate(zkp.Prover(pp), zkp.Verifier(pp), "c", Z, bind_plaintext=True, Z_sent=sent).authenticated


def test_chain_mismatch_fails(pp):
    """A prover whose chain is out of sync with the server is rejected."""
    rng = np.random.default_rng(3)
    prover, verifier = zkp.Prover(pp), zkp.Verifier(pp)
    assert zkp.authenticate(prover, verifier, "c", rng.normal(size=(3, 3))).authenticated
    verifier.prev["c"] = [x * pp.g[0] % pp.p for x in verifier.prev["c"]]
    assert not zkp.authenticate(prover, verifier, "c", rng.normal(size=(3, 3))).authenticated


# --- global model -------------------------------------------------------------
def test_global_rf_pseudo_labels_are_out_of_bag():
    from fedrf_diff.client import LatentMessage
    from fedrf_diff.server import Server

    rng = np.random.default_rng(0)
    Z = {c: rng.normal(size=(60, 4)) for c in ("a", "b")}
    msgs = {c: LatentMessage(c, Z[c], Z[c][:5], rng.integers(0, 2, 60), 60) for c in Z}
    model, pseudo = Server(seed=0, n_trees=200, min_samples_leaf=1).train_global_model(Z, msgs)
    forest = model.forest
    assert {c: len(v) for c, v in pseudo.items()} == {"a": 60, "b": 60}
    X, y = np.vstack(list(Z.values())), np.concatenate([m.y for m in msgs.values()])
    # Random labels: in-sample predictions memorise y, out-of-bag ones cannot.
    assert (forest.predict(X) == y).mean() > 0.95
    assert 0 < model.predict_proba(X).min() and model.predict_proba(X).max() < 1
    assert (np.concatenate(list(pseudo.values())) == y).mean() < 0.7


# --- metrics ------------------------------------------------------------------
def test_metrics_match_sklearn():
    from sklearn import metrics as skm

    from fedrf_diff.metrics import confusion_counts, metrics_from_counts, metrics_table

    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
    m = metrics_from_counts(**confusion_counts(y, p))
    assert m["accuracy"] == pytest.approx(skm.accuracy_score(y, p))
    assert m["precision"] == pytest.approx(skm.precision_score(y, p))
    assert m["recall"] == pytest.approx(skm.recall_score(y, p))
    assert m["specificity"] == pytest.approx(skm.recall_score(y, p, pos_label=0))
    assert m["f1"] == pytest.approx(skm.f1_score(y, p))
    assert m["mcc"] == pytest.approx(skm.matthews_corrcoef(y, p))
    assert m["kappa"] == pytest.approx(skm.cohen_kappa_score(y, p))
    # Degenerate case (all predicted positive) must not divide by zero.
    assert metrics_from_counts(tp=5, fp=3, tn=0, fn=0)["mcc"] == 0.0

    a, b = confusion_counts(y[:80], p[:80]), confusion_counts(y[80:], p[80:])
    table = metrics_table({"a": {**a, "loss": 0.5}, "b": {**b, "loss": 0.3}})
    pooled = table.set_index("client").loc["All clients (pooled)"]
    assert pooled["mcc"] == pytest.approx(skm.matthews_corrcoef(y, p))
    assert pooled["log_loss"] == pytest.approx((0.5 * 80 + 0.3 * 120) / 200)


def test_summarize_runs_mean_and_std():
    import pandas as pd

    from fedrf_diff.metrics import summarize_runs

    t1 = pd.DataFrame({"client": ["A", "B"], "log_loss": [0.2, 0.4], "accuracy": [0.8, 0.6],
                       "precision": 0.5, "recall": 0.5, "specificity": 0.5, "f1": 0.5,
                       "mcc": [0.1, -0.1], "kappa": 0.0})
    t2 = t1.assign(accuracy=[0.9, 0.6], mcc=[0.3, 0.1])
    all_runs, summary = summarize_runs({1: t1, 2: t2})
    assert sorted(all_runs["seed"].unique()) == [1, 2] and len(all_runs) == 4
    s = summary.set_index("client")
    assert list(s.index) == ["A", "B"] and (s["n_runs"] == 2).all()
    assert s.loc["A", "accuracy_mean"] == pytest.approx(0.85)
    assert s.loc["A", "accuracy_std"] == pytest.approx(np.std([0.8, 0.9], ddof=1))
    assert s.loc["B", "mcc_mean"] == pytest.approx(0.0)


# --- privacy (Gaussian DP) ------------------------------------------------------
def test_gaussian_perturb_adds_noise_of_the_right_size():
    from fedrf_diff.privacy import gaussian_perturb

    Z = np.zeros((2000, 4))
    noisy = gaussian_perturb(Z, 0.05, np.random.default_rng(0))
    assert not np.allclose(noisy, Z)
    assert abs(noisy.mean()) < 0.005 and noisy.std() == pytest.approx(0.05, rel=0.05)
    assert np.array_equal(Z, np.zeros((2000, 4)))  # input left untouched


def test_gaussian_perturb_sigma_zero_and_negative():
    from fedrf_diff.privacy import gaussian_perturb

    Z = np.arange(12.0).reshape(3, 4)
    same = gaussian_perturb(Z, 0.0, np.random.default_rng(0))
    assert np.array_equal(same, Z) and same is not Z
    with pytest.raises(ValueError):
        gaussian_perturb(Z, -0.1, np.random.default_rng(0))


def test_gaussian_mechanism_calibration_clipping_and_accounting():
    from fedrf_diff import privacy

    for eps in (0.5, 1.0, 20.0, 100.0):
        rho = privacy.rho_for_epsilon(eps, 1e-5)
        assert privacy.epsilon_for_rho(rho, 1e-5) == pytest.approx(eps)
    # Classic check: sigma shrinks as the budget grows, and doubles with C.
    s1, s2 = privacy.gaussian_sigma(1.0, 1e-5, 1.0), privacy.gaussian_sigma(10.0, 1e-5, 1.0)
    assert s1 > s2 > 0 and privacy.gaussian_sigma(1.0, 1e-5, 2.0) == pytest.approx(2 * s1)
    Z = np.random.default_rng(0).normal(size=(500, 4)) * 3
    assert np.linalg.norm(privacy.clip_rows(Z, 1.0), axis=1).max() <= 1.0 + 1e-12
    fixed = privacy.GaussianMechanism(20.0, 1e-5, 1.0, adaptive=False)
    assert fixed.sigma == pytest.approx(privacy.gaussian_sigma(20.0, 1e-5, 1.0))
    mech = privacy.GaussianMechanism(20.0, 1e-5, 1.0)
    out = mech.privatize(np.zeros((4000, 4)), np.random.default_rng(1))
    assert out.std() == pytest.approx(mech.sigma_history[-1], rel=0.05)
    mech.privatize(np.zeros((10, 4)), np.random.default_rng(2))
    assert mech.spent_epsilon() == pytest.approx(privacy.epsilon_for_rho(2 * mech.rho, 1e-5))
    assert 20.0 < mech.spent_epsilon() < 40.0  # zCDP composition is sub-linear
    off = privacy.GaussianMechanism(None)
    assert np.array_equal(off.privatize(Z, np.random.default_rng(0)), Z) and off.spent_epsilon() == 0


def test_adaptive_clip_tracks_quantile_at_same_budget():
    from fedrf_diff import privacy

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(400, 4))
    Z *= (np.exp(rng.normal(size=(400, 1)) * 0.3) * 3.0 / np.linalg.norm(Z, axis=1, keepdims=True))
    target = np.quantile(np.linalg.norm(Z, axis=1), privacy.CLIP_QUANTILE)
    mech = privacy.GaussianMechanism(20.0, 1e-5, 1.0, adaptive=True)
    for _ in range(30):
        mech.privatize(Z, rng)
    assert mech.clip_norm == pytest.approx(target, rel=0.1)
    # sigma scales with C at a fixed ratio; the budget per round is unchanged.
    assert mech.sigma_history[-1] / mech.clip_history[-1] == pytest.approx(mech.noise_ratio)
    assert mech.rho_vec + mech.rho_count == pytest.approx(privacy.rho_for_epsilon(20.0, 1e-5))
    fixed = privacy.GaussianMechanism(20.0, 1e-5, 1.0, adaptive=False)
    fixed.releases = 30
    assert mech.spent_epsilon() == pytest.approx(fixed.spent_epsilon())
