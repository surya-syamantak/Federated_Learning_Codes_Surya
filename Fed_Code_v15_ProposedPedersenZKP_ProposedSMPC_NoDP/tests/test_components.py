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


# --- ZKP client authentication (proposed Pedersen) ------------------------------
@pytest.fixture(scope="module")
def pp():
    return zkp.setup(group="test512")


@pytest.mark.parametrize("group", sorted(zkp.GROUPS))
def test_group_primes_and_generators(group):
    from fedrf_diff.zkp import _is_prime

    pp = zkp.setup(group=group)
    p, q, g, h = int(pp.p), int(pp.q), int(pp.g), int(pp.h)
    assert (p.bit_length(), q.bit_length()) == zkp.GROUPS[group]
    assert _is_prime(p) and _is_prime(q) and (p - 1) % q == 0
    for x in (g, h):  # both generate the order-q subgroup
        assert x != 1 and pow(x, q, p) == 1
    assert g != h
    # Fixed-base tables agree with plain modular exponentiation.
    for e in (0, 1, 255, 256, q - 1, 2**200 + 12345):
        assert int(pp.g_pow(e)) == pow(g, e % q, p) and int(pp.h_pow(e)) == pow(h, e % q, p)
    zkp._setup.cache_clear()
    assert zkp.setup(group=group) == pp  # p, q, g, h are regenerated identically from SEED


def test_honest_client_follows_the_diagram_over_rounds(pp):
    p, q, g, h, lam = int(pp.p), int(pp.q), int(pp.g), int(pp.h), zkp.LAMBDA
    client, server = zkp.ClientProver(pp, "Client 1(Hungary)"), zkp.ServerVerifier(pp)
    assert client.C_prev == 1  # round 1: C_{r-1} initialised to 1
    for t in (1, 2, 3):
        C_prev = int(client.C_prev)
        C_t = int(client.commit(t))  # 1. C_r = g^{v_r} h^{ρ_r} (C_{r-1})^beta
        v_t, r_t = int(client._v), int(client._r)
        assert C_t == pow(g, v_t, p) * pow(h, r_t, p) * pow(C_prev, lam, p) % p
        T = int(client.announce())  # 2. T = g^k h^tau
        assert T == pow(g, int(client._k), p) * pow(h, int(client._tau), p) % p
        c = int(server.challenge())  # 3. random challenge
        k, tau = int(client._k), int(client._tau)
        r_v, r_s = (int(x) for x in client.respond(c))  # 4. responses
        assert r_v == (k + c * v_t) % q and r_s == (tau + c * r_t) % q
        C_base = C_t * pow(pow(C_prev, lam, p), -1, p) % p  # C_r / (C_{r-1})^beta
        assert pow(g, r_v, p) * pow(h, r_s, p) % p == T * pow(C_base, c, p) % p
        assert server.verify(client.client_id, C_t, T, c, r_v, r_s)
        server.decide(client.client_id, C_t, True)  # 5. Authentication Successful
        client.finish(True)
        assert client.C_prev == C_t and server.C_prev[client.client_id] == C_t  # chain moves on


def test_sessions_are_valid_and_record_times_and_scalar_messages(pp):
    client, server = zkp.ClientProver(pp, "c"), zkp.ServerVerifier(pp)
    for t in (1, 2):
        s = zkp.authenticate(pp, client, server, t, vectors=6)
        assert s.authenticated and s.outcome == "Valid" and s.vectors == 6
        assert s.authentication_s >= s.proof_generation_s > 0 and s.proof_verification_s > 0
        assert s.bytes_up == 2 * pp.p_bytes + 2 * pp.q_bytes and s.bytes_down == pp.q_bytes + 1
    # Fresh k, tau, ρ_r and c every session: transcripts never repeat.
    first = client.last_transcript
    zkp.authenticate(pp, client, server, 3)
    assert all(a != b for a, b in zip(first, client.last_transcript))


def test_replayed_wrong_secret_and_forged_sessions_fail(pp):
    client, server = zkp.ClientProver(pp, "c"), zkp.ServerVerifier(pp)
    assert zkp.authenticate(pp, client, server, 1).authenticated
    old = client.last_transcript
    # An impostor replaying the last session: the fresh challenge c makes it fail.
    replayed = zkp.authenticate(pp, client, server, 2, replay=old)
    assert not replayed.authenticated and replayed.outcome == "Invalid"
    assert server.C_prev["c"] == client.C_prev == old[0]  # chain kept after a failure
    # Responses that do not match the challenge (forged y_v or y_s).
    C_t, T = client.commit(2), client.announce()
    c = server.challenge()
    r_v, r_s = client.respond(c)
    assert not server.verify("c", C_t, T, c, (r_v + 1) % pp.q, r_s)
    assert not server.verify("c", C_t, T, c, r_v, (r_s + 1) % pp.q)
    assert not server.verify("c", C_t, T, (c + 1) % pp.q, r_v, r_s)
    assert not server.verify("c", C_t * pp.g % pp.p, T, c, r_v, r_s)  # altered commitment
    assert not server.verify("other client", C_t, T, c, r_v, r_s)  # wrong C_{r-1} in the chain
    assert not server.verify("c", C_t, 0, c, r_v, r_s)  # T outside the group
    assert server.verify("c", C_t, T, c, r_v, r_s)
    # A client whose commitment chain diverged from the server's is rejected.
    stranger = zkp.ClientProver(pp, "c")  # same name, unrelated C_{r-1} = 1 and secret
    C_s, T_s = stranger.commit(3), stranger.announce()
    c = server.challenge()
    assert not server.verify("c", C_s, T_s, c, *stranger.respond(c))


def test_secret_is_a_scalar_that_stays_on_the_client(pp):
    client = zkp.ClientProver(pp, "c")
    v1, v2 = client.secret(1), client.secret(2)
    assert 0 <= v1 < pp.q and v1 != v2 and v1 == client.secret(1)
    assert zkp.ClientProver(pp, "c").secret(1) != v1  # each client draws its own key


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


# --- Proposed SMPC: adaptive prime secure aggregation with pairwise masking ---------------
def test_the_diagrams_worked_example():
    """Round r = 5 of the diagram: p_r = 997, three clients, two values each."""
    from fedrf_diff import smpc

    p = 997
    Q = {1: [42, 18], 2: [31, 26], 3: [27, 15]}  # Phase 4 with S_quant = 100
    r = {(1, 2): [20, 15], (1, 3): [8, 4], (2, 3): [5, 10]}  # Phase 5 stream outputs
    masks = {}
    for i in (1, 2, 3):
        masks[i] = [sum(smpc.sign(i, j) * r[tuple(sorted((i, j)))][k] for j in (1, 2, 3) if j != i) % p
                    for k in range(2)]
    assert masks[1] == [28, 19] and masks[2] == [982, 992] and masks[3] == [984, 983]
    c = {i: [(q + m) % p for q, m in zip(Q[i], masks[i])] for i in Q}  # Phase 6
    assert c == {1: [70, 37], 2: [16, 21], 3: [14, 1]}
    agg = [sum(col) % p for col in zip(*c.values())]  # Phase 7
    assert agg == [100, 59] == [sum(col) for col in zip(*Q.values())]
    assert [sum(col) % p for col in zip(*masks.values())] == [0, 0]  # the masks cancel
    assert [a / 100 for a in agg] == [1.00, 0.59]  # Phase 8
    # The diagram's note: if C3 drops out after masking, r_13 and r_23 stay in the sum.
    partial = [sum(col) % p for col in zip(c[1], c[2])]
    assert partial != [sum(col) for col in zip(Q[1], Q[2])]


def test_prime_pool_and_adaptive_prime_selection():
    from fedrf_diff import smpc
    from fedrf_diff.zkp import _is_prime

    pool = smpc.prime_pool()
    assert pool == smpc.prime_pool() and len(set(pool)) == smpc.POOL_SIZE  # reproducible, distinct
    assert all(_is_prime(p) and p.bit_length() == smpc.PRIME_BITS for p in pool)
    choices = [smpc.select_prime(t, pool) for t in range(1, 31)]
    assert all(pool[i] == p for i, p in choices) and choices == [smpc.select_prime(t, pool) for t in range(1, 31)]
    assert len({i for i, _ in choices}) > 1  # the prime changes between rounds


def test_quantization_round_trips_and_is_fine_enough():
    from fedrf_diff import smpc

    p = smpc.prime_pool()[0]
    for v in (0.0, 1.0, 329.5308, 0.123456, 98, -7.5):
        assert smpc.dequantize(smpc.quantize(v) % p, p) == pytest.approx(v, abs=0.5 / smpc.SCALE)
    assert smpc.SCALE >= 10**6  # L_glob keeps 6 decimals (Delta L < 0.02 is the stopping test)


def test_secure_aggregator_reveals_only_exact_totals():
    from fedrf_diff import smpc

    names = ["A", "B", "C", "D"]
    agg = smpc.SecureAggregator(names)
    assert agg.key("A", "C") == agg.key("C", "A") and agg.key("A", "C") != agg.key("A", "B")
    vectors = {"A": [10.25, 3.5, 5, 1, 7, 2], "B": [4.0, 1.75, 2, 0, 9, 1],
               "C": [0.5, 0.125, 8, 3, 1, 0], "D": [7.0, 2.0, 1, 1, 1, 1]}
    out = agg.aggregate(5, vectors)
    for k, f in enumerate(smpc.FIELDS):
        assert out.totals[f] == pytest.approx(sum(v[k] for v in vectors.values()), abs=1e-9)
    _, p = smpc.select_prime(5, agg.pool)
    assert out.prime == p
    # The server receives only masked vectors: none equals the client's own quantized vector.
    for n, vec in vectors.items():
        assert out.masked[n] != [smpc.quantize(v) % p for v in vec]
    # The self-masks cancel over the confirmed participants.
    masks = [agg.mask(n, 5, p, names, len(smpc.FIELDS)) for n in names]
    assert [sum(col) % p for col in zip(*masks)] == [0] * len(smpc.FIELDS)
    # Fresh streams every round.
    assert agg.mask("A", 5, p, names, 6) != agg.mask("A", 6, p, names, 6)
    for cost in out.clients.values():
        assert cost.bytes_up == len(smpc.FIELDS) * smpc.ELEMENT_BYTES and cost.mask_s > 0
    with pytest.raises(ValueError):
        agg.aggregate(5, {"A": [1.0, 2.0]})


def test_dropout_after_masking_is_recovered():
    from fedrf_diff import smpc

    agg = smpc.SecureAggregator(["A", "B", "C"])
    vectors = {"A": [1.5, 0.5, 4, 1, 3, 0], "B": [2.25, 1.0, 2, 2, 2, 1], "C": [3.0, 1.5, 6, 0, 1, 2]}
    out = agg.aggregate(7, vectors, drop_after_masking=("C",))
    assert out.dropped == ["C"] and set(out.masked) == {"A", "B"}
    for k, f in enumerate(smpc.FIELDS):  # the exact sum over the survivors
        assert out.totals[f] == pytest.approx(vectors["A"][k] + vectors["B"][k], abs=1e-9)
    assert out.clients["C"].bytes_up == 0 and out.clients["A"].recovery_s > 0
    # Masks over a smaller confirmed set (C rejected before masking) need no recovery.
    out2 = agg.aggregate(8, {"A": vectors["A"], "B": vectors["B"]})
    assert out2.totals["tp"] == pytest.approx(6) and not out2.dropped


def test_values_too_large_for_the_prime_are_refused():
    from fedrf_diff import smpc

    agg = smpc.SecureAggregator(["A", "B"])
    huge = float(smpc.prime_pool()[0])  # times S_quant is far beyond p_r / (2N)
    with pytest.raises(OverflowError):
        agg.aggregate(1, {"A": [huge, 0, 0, 0, 0, 0], "B": [0, 0, 0, 0, 0, 0]})
