import copy
import json

import numpy as np
import pytest
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, log_loss as sk_log_loss, matthews_corrcoef,
                             precision_score, recall_score)

from fedrf_diff import zkp
from fedrf_diff.alignment import check_local_data
from fedrf_diff.config import CONFIG
from fedrf_diff.client import balanced_accuracy
from fedrf_diff.convergence import LocalConvergence, log_loss
from fedrf_diff.data import load_hospitals
from fedrf_diff.metrics import MCC, confusion_counts, metrics_from_counts
from fedrf_diff.privacy import PrivacyAccountant, epsilon_schedule, laplace_scale, privatize_leaves
from fedrf_diff.rf_kernel import forest_proba, train_local_forest
from fedrf_diff.server import (FederatedServer, InvalidUpdate, Limits, check_update, load_forest, save_forest,
                               select_trees)

CREATORS = {"client_1", "client_2", "client_3", "client_4"}


# ── Data and schema ─────────────────────────────────────────────────────────
def test_hospitals_are_horizontal_partitions_without_duplicates():
    df = load_hospitals(CONFIG["raw_csv"])
    sizes = df.hospital.value_counts().to_dict()
    assert sizes == {"Hungary": 293, "Switzerland": 123, "VA Long Beach": 199, "Cleveland": 303}
    assert not df.drop(columns="hospital").duplicated().any()


def test_clients_share_one_schema(federation):
    schema, clients, _ = federation
    assert [c.name for c in clients] == ["Client 1(Hungary)", "Client 2(Switzerland)",
                                         "Client 3(VA Long Beach)", "Client 4(Cleveland)"]
    for c in clients:
        assert c.X_train.shape[1] == c.X_val.shape[1] == schema.n_features
        assert not np.isnan(c.X_train).any()


def test_unknown_label_is_refused(federation):
    _, clients, _ = federation
    bad = clients[0].train_raw.copy()
    bad.iloc[0, bad.columns.get_loc("target")] = 2
    with pytest.raises(ValueError):
        check_local_data(bad)


# ── Proposed Pedersen ZKP (the same protocol as FedRF-Diff v12) ─────────────
@pytest.fixture(scope="module")
def pp():
    return zkp.setup()


def test_group_parameters_are_rederivable(pp):
    p, q, g, h = int(pp.p), int(pp.q), int(pp.g), int(pp.h)
    assert (p.bit_length(), q.bit_length()) == zkp.GROUPS["pedersen2048"] == (2048, 256)
    assert zkp.generate_group(2048, 256) == (p, q) and (p - 1) % q == 0
    assert all(x != 1 and pow(x, q, p) == 1 for x in (g, h)) and g != h
    assert zkp.SEED == "FedRF-Diff Pedersen ZKP group"  # the same group as FedRF-Diff v12


@pytest.mark.parametrize("lam", [1, 2, 3])
def test_honest_client_follows_the_diagram_over_rounds(pp, lam):
    p, q, g, h = int(pp.p), int(pp.q), int(pp.g), int(pp.h)
    client, server = zkp.ClientProver(pp, "client_1", lam), zkp.ServerVerifier(pp, lam)
    assert client.C_prev == 1  # round 1: C_{r-1} initialised to 1
    for t in (1, 2, 3):
        C_prev = int(client.C_prev)
        C_t = int(client.commit(t))  # 1. C_r = g^{v_r} h^{ρ_r} (C_{r-1})^beta
        v_t, r_t = int(client._v), int(client._r)
        assert C_t == pow(g, v_t, p) * pow(h, r_t, p) * pow(C_prev, lam, p) % p
        T = int(client.announce())  # 2. T = g^k h^tau
        k, tau = int(client._k), int(client._tau)
        c = int(server.challenge())  # 3. random challenge
        r_v, r_s = (int(x) for x in client.respond(c))  # 4. responses
        assert r_v == (k + c * v_t) % q and r_s == (tau + c * r_t) % q
        C_base = C_t * pow(pow(C_prev, lam, p), -1, p) % p  # C_r / (C_{r-1})^beta
        assert pow(g, r_v, p) * pow(h, r_s, p) % p == T * pow(C_base, c, p) % p
        assert server.verify("client_1", C_t, T, c, r_v, r_s)
        server.decide("client_1", C_t, True)  # 5. Authentication Successful
        client.finish(True)
        assert client.C_prev == C_t and server.C_prev["client_1"] == C_t  # chain moves on


def test_replayed_forged_and_out_of_chain_sessions_fail(pp):
    client, server = zkp.ClientProver(pp, "client_1"), zkp.ServerVerifier(pp)
    assert zkp.authenticate(pp, client, server, 1).authenticated
    old = client.last_transcript
    replayed = zkp.authenticate(pp, client, server, 2, replay=old)  # impostor replays round 1
    assert not replayed.authenticated and replayed.outcome == "Invalid"
    assert server.C_prev["client_1"] == client.C_prev == old[0]  # both keep C_{r-1} after a failure
    C_t, T = client.commit(2), client.announce()
    c = server.challenge()
    r_v, r_s = client.respond(c)
    assert not server.verify("client_1", C_t, T, c, (r_v + 1) % pp.q, r_s)  # forged response
    assert not server.verify("client_1", C_t, T, (c + 1) % pp.q, r_v, r_s)  # wrong challenge
    assert not server.verify("client_2", C_t, T, c, r_v, r_s)  # another client's chain
    # A server that uses another lambda divides out the wrong power of C_{r-1} (here != 1).
    other = zkp.ServerVerifier(pp, lam=3)
    other.C_prev = dict(server.C_prev)
    assert not other.verify("client_1", C_t, T, c, r_v, r_s)
    assert server.verify("client_1", C_t, T, c, r_v, r_s)


def test_federation_wires_the_proposed_zkp(federation):
    schema, clients, server = federation
    assert all(isinstance(c.prover, zkp.ClientProver) for c in clients)
    assert isinstance(server.verifier, zkp.ServerVerifier) and server.verifier.lam == clients[0].prover.lam == 2
    assert not hasattr(server, "public_keys")  # no registration step: C_0 = 1
    for rnd in (1, 2):
        sessions = [zkp.authenticate(server.verifier.pp, c.prover, server.verifier, rnd) for c in clients]
        assert all(s.authenticated for s in sessions)
        assert all(s.bytes_up == 2 * 256 + 2 * 32 and s.bytes_down == 32 + 1 for s in sessions)
    assert len(set(server.verifier.C_prev.values())) == len(clients)  # one chain per client


# ── Whole-tree export and server validation ─────────────────────────────────
def test_explicit_trees_reproduce_the_local_forest(federation):
    schema, clients, server = federation
    c = clients[3]
    n = c.cfg["initial_trees"]  # an honest F_k^(0) (bounded-update enforcement)
    trees = c.grow(0, n, None)
    seed = int(c._seed(0, 0).generate_state(1)[0])
    forest = train_local_forest(c.X_train, c.y_train, n, seed, c.cfg["rf_params"])
    np.testing.assert_allclose(forest_proba(trees, c.X_val), forest.predict_proba(c.X_val), atol=1e-12)
    c.forest = c.pending = trees
    received, rejected = server.receive(c.client_id, c.serialize(1), 1, CREATORS)
    assert rejected == []
    np.testing.assert_array_equal(forest_proba(received, c.X_val), forest_proba(trees, c.X_val))


def _valid_update(federation):
    schema, clients, server = federation
    c = clients[0]
    c.initialise(None)
    return json.loads(c.serialize(1)), c.client_id, schema


def _set_node(update, i, **fields):
    nodes = update["trees"][0]["nodes"]
    node = next(n for n in nodes if n["node_id"] == i)
    node.update(fields)


@pytest.mark.parametrize("corrupt, reason", [
    (lambda u: u["trees"][0].update(schema_version="features-v0"), "schema version"),
    (lambda u: u["trees"][0].update(classes=[1, 0]), "class order"),
    (lambda u: u["trees"][0].update(tree_id="client_9_r001_t000"), "tree id"),
    (lambda u: u["trees"][0].update(client_id="client_9", tree_id="client_9_r000_t000"), "unknown client"),
    (lambda u: _set_node(u, 0, feature=99), "unknown feature"),
    (lambda u: _set_node(u, 0, feature=True), "unknown feature"),
    (lambda u: _set_node(u, 0, threshold="1.5"), "threshold"),
    (lambda u: _set_node(u, 0, left_child=0), "child references"),
    (lambda u: _set_node(u, 0, right_child=10 ** 6), "child references"),
    (lambda u: u["trees"][0]["nodes"][1].update(node_id=0), "duplicate node id"),
    (lambda u: u["trees"][0]["nodes"].append({"node_id": len(u["trees"][0]["nodes"]), "leaf": True,
                                              "class_distribution": [0.5, 0.5]}), "unreachable"),
    (lambda u: _set_node(u, 0, right_child=u["trees"][0]["nodes"][0]["left_child"]), "child references"),
])
def test_malformed_tree_is_rejected_alone(federation, limits, corrupt, reason):
    update, client_id, schema = _valid_update(federation)
    n_trees = len(update["trees"])
    corrupt(update)
    trees, rejected, _ = check_update(json.dumps(update).encode(), schema, limits, client_id, 1, CREATORS)
    assert len(trees) == n_trees - 1
    assert len(rejected) == 1 and reason in rejected[0][1]


def test_invalid_leaf_and_shared_child_are_rejected(federation, limits):
    update, client_id, schema = _valid_update(federation)
    nodes = update["trees"][0]["nodes"]
    leaf = next(n for n in nodes if n.get("leaf"))
    leaf["class_distribution"] = [0.9, 0.3]
    # Second tree: an inner node points at a node that already has a parent
    nodes2 = update["trees"][1]["nodes"]
    inner = [n for n in nodes2 if not n.get("leaf") and n["node_id"] != 0]
    inner[0]["left_child"] = nodes2[0]["left_child"] if nodes2[0]["left_child"] != inner[0]["node_id"] else 0
    trees, rejected, _ = check_update(json.dumps(update).encode(), schema, limits, client_id, 1, CREATORS)
    assert len(rejected) == 2
    assert "probability distribution" in rejected[0][1]


def test_tree_too_deep_is_rejected(federation):
    update, client_id, schema = _valid_update(federation)
    shallow = Limits(10 ** 8, 500, 20_000, max_depth=2)
    trees, rejected, _ = check_update(json.dumps(update).encode(), schema, shallow, client_id, 1, CREATORS)
    assert rejected and all("depth" in r for _, r in rejected)


def test_malformed_update_is_rejected_whole(federation, limits):
    update, client_id, schema = _valid_update(federation)
    raw = json.dumps(update).encode()
    with_nan = copy.deepcopy(update)
    next(n for n in with_nan["trees"][0]["nodes"] if "threshold" in n)["threshold"] = float("nan")
    for payload, rnd, cid in [(raw, 2, client_id), (raw, 1, "client_2"), (b"{not json", 1, client_id),
                              (json.dumps(with_nan).encode(), 1, client_id)]:
        with pytest.raises(InvalidUpdate):
            check_update(payload, schema, limits, cid, rnd, CREATORS)
    wrong = copy.deepcopy(update)
    wrong["schema_id"] = "0" * 16
    with pytest.raises(InvalidUpdate):
        check_update(json.dumps(wrong).encode(), schema, limits, client_id, 1, CREATORS)
    with pytest.raises(InvalidUpdate):
        check_update(raw, schema, Limits(100, 500, 20_000, 64), client_id, 1, CREATORS)


def test_tree_provenance_is_enforced(federation):
    schema, clients, server = federation
    c1, c2 = clients[0], clients[1]
    c1.initialise(None)
    first, _ = server.receive(c1.client_id, c1.serialize(1), 1, CREATORS)
    # client 2 keeps two of client 1's trees (they came back inside G): it names them by id only
    c2.initialise(None)
    c2.forest = c2.forest + first[:2]
    again, rejected = server.receive(c2.client_id, c2.serialize(1), 1, CREATORS)
    assert len(again) == len(c2.pending) and rejected == []
    assert set(server.kept[c2.client_id]) >= {t.tree_id for t in first[:2]}
    # but it may neither re-send client 1's tree nor mint a new id in client 1's name
    c2.pending = first[:1]
    upload = json.loads(c2.serialize(2))
    upload["trees"].append(dict(upload["trees"][0], tree_id="client_1_r099_t000"))
    kept, rejected = server.receive(c2.client_id, json.dumps(upload).encode(), 2, CREATORS)
    assert kept == [] and [r for _, r in rejected] == ["tree uploaded for another client"] * 2


def test_select_trees_policies(federation):
    _, clients, _ = federation
    pool = []
    for c in clients:
        pool += c.grow(1, 3, None) + c.grow(2, 3, None)
    balanced = select_trees(pool, "client-balanced", 8)
    assert len(balanced) == 8 and {t.client_id for t in balanced} == CREATORS
    assert all(t.tree_id.split("_r")[1].startswith("002") for t in balanced)  # newest first per client
    assert [t.tree_id for t in select_trees(pool, "newest", 3)] == sorted((t.tree_id for t in pool), reverse=True)[:3]
    old = next(t for t in pool if t.tree_id == "client_1_r001_t000")
    consensus = select_trees(pool, "consensus", 4, votes={old.tree_id: 4})
    assert old in consensus and len(select_trees(pool, "consensus", 100)) == len(pool)
    with pytest.raises(ValueError):
        select_trees(pool, "average", 4)


def test_aggregation_pools_the_union_and_caps_at_b(federation):
    schema, clients, _ = federation
    server = FederatedServer(schema, Limits.from_config(CONFIG), 2, "client-balanced", 5)
    forests = [c.grow(0, 3, None) for c in clients]
    shared = forests[0][0]
    pool_size = server.aggregate(forests + [[shared]])  # a tree held by two clients counts once
    assert pool_size == 12 and len(server.global_forest) == 5
    assert server.stopping_decision([True] * 4, 3, 50) == "all-client convergence"
    assert server.stopping_decision([True, False, True, True], 50, 50) == "maximum rounds reached"
    assert server.stopping_decision([True, False, True, True], 3, 50) is None


def test_cap_or_prune_keeps_the_best_out_of_bag_trees(federation):
    schema, clients, _ = federation
    c, other = clients[0], clients[3]
    c.initialise(None)
    foreign = other.grow(0, 6, None)
    supplementary = c.refine(foreign, 1, None)
    pool = foreign + supplementary
    assert len(c.forest) == c.cfg["local_budget"] <= len(pool)
    scores = {t.tree_id: c._score(t) for t in pool}
    kept = {t.tree_id for t in c.forest}
    assert min(scores[i] for i in kept) >= max(scores[t.tree_id] for t in pool if t.tree_id not in kept)
    # a foreign tree is scored on every training patient, an own tree only on its out-of-bag patients
    f = foreign[0]
    assert scores[f.tree_id] == balanced_accuracy(c.y_train, c.predict(f.predict_proba(c.X_train)), c.classes)
    own = supplementary[0]
    oob = np.ones(len(c.y_train), bool)
    oob[c._in_bag[own.tree_id]] = False
    assert oob.sum() < len(c.y_train)
    assert scores[own.tree_id] == balanced_accuracy(c.y_train[oob], c.predict(own.predict_proba(c.X_train[oob])),
                                                    c.classes)


def test_saved_forest_reloads(federation, tmp_path, limits):
    schema, clients, _ = federation
    c = clients[2]
    trees = c.grow(0, 4, 2.0)
    save_forest(tmp_path / "forest.json", trees, schema, 1)
    loaded = load_forest(tmp_path / "forest.json", schema, limits)
    np.testing.assert_array_equal(forest_proba(loaded, c.X_val), forest_proba(trees, c.X_val))


# ── Laplace DP ──────────────────────────────────────────────────────────────
def test_dp_leaves_are_valid_distributions_and_structure_is_kept(federation):
    _, clients, _ = federation
    c = clients[1]
    trees, noisy = c.grow(1, 4, None), c.grow(1, 4, 2.0)
    for t, n in zip(trees, noisy):
        np.testing.assert_array_equal(t.threshold, n.threshold)
        np.testing.assert_array_equal(t.left, n.left)
        leaves = n.value[n.is_leaf]
        assert (leaves >= 0).all() and np.allclose(leaves.sum(axis=1), 1)
        assert not np.allclose(leaves, t.value[t.is_leaf])


def test_laplace_mechanism_uses_b_equal_2c_over_epsilon():
    eps, clip = 2.5, 1.0
    assert laplace_scale(eps, clip) == pytest.approx(0.8)
    value = np.random.default_rng(0).dirichlet([1, 1], 1000)
    is_leaf = np.arange(1000) % 3 != 0
    out = privatize_leaves(value, is_leaf, eps, clip, np.random.default_rng(7))
    noisy = np.clip(value[is_leaf] + np.random.default_rng(7).laplace(0, 0.8, (is_leaf.sum(), 2)), 0, None)
    totals = noisy.sum(axis=1, keepdims=True)
    expected = np.where(totals > 0, noisy / np.where(totals > 0, totals, 1), 0.5)
    np.testing.assert_allclose(out[is_leaf], expected)
    np.testing.assert_array_equal(out[~is_leaf], value[~is_leaf])


def test_epsilon_schedule_and_accountant():
    sched = epsilon_schedule((2.0, 3.0), 10, 42)
    assert all(2 <= e <= 3 for e in sched) and sched == epsilon_schedule((2.0, 3.0), 10, 42)
    assert epsilon_schedule((2.0, 3.0), 4, 42) == sched[:4]
    assert epsilon_schedule(3, 2, 42) == [3.0, 3.0] and epsilon_schedule(None, 2, 42) == [None, None]
    acc = PrivacyAccountant()
    assert acc.spend(2.0, 25) == 50.0 and acc.spend(None, 25) == 0.0 and acc.total == 50.0


# ── Metrics and convergence ─────────────────────────────────────────────────
def test_metrics_match_scikit_learn():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
    m = metrics_from_counts(confusion_counts(y, p))
    assert m["Accuracy"] == pytest.approx(accuracy_score(y, p))
    assert m["Precision"] == pytest.approx(precision_score(y, p))
    assert m["Recall"] == pytest.approx(recall_score(y, p))
    assert m["F1 Score"] == pytest.approx(f1_score(y, p))
    assert m[MCC] == pytest.approx(matthews_corrcoef(y, p))
    assert m["Cohen's Kappa"] == pytest.approx(cohen_kappa_score(y, p))
    # Global Server = summed counts, same as pooling every validation patient
    y2, p2 = rng.integers(0, 2, 50), rng.integers(0, 2, 50)
    pooled = metrics_from_counts(confusion_counts(y, p) + confusion_counts(y2, p2))
    assert pooled[MCC] == pytest.approx(matthews_corrcoef(np.r_[y, y2], np.r_[p, p2]))


def test_log_loss_matches_scikit_learn():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 80)
    p1 = rng.uniform(0.01, 0.99, 80)
    proba = np.c_[1 - p1, p1]
    assert log_loss(y, proba, np.array([0, 1])) == pytest.approx(sk_log_loss(y, proba))


def test_local_convergence_needs_all_three_tests_for_p_rounds():
    conv = LocalConvergence(tau_loss=0.05, tau_pred=0.05, tau_agree=0.15, patience=2)
    same = np.zeros(20, int)
    conv.initialise(0.50, same)
    one_flip = same.copy()
    one_flip[0] = 1  # 5 % of the patients
    four_flips = same.copy()
    four_flips[:4] = 1  # 20 %
    steps = [(0.51, same, same),  # all pass -> q = 1
             (0.52, same, same),  # q = 2 -> flag
             (0.60, same, same),  # loss jumps 15 % -> reset
             (0.60, one_flip, one_flip),  # 5 % changed, still within tau_pred -> q = 1
             (0.60, one_flip, four_flips),  # 15 % disagreement with G -> q = 2 -> flag
             (0.60, four_flips, four_flips),  # 15 % predictions changed -> reset
             (0.60, four_flips, same)]  # stable but 20 % disagreement with G -> reset
    results = [conv.update(loss, pred, g) for loss, pred, g in steps]
    assert [r.patience for r in results] == [1, 2, 0, 1, 2, 0, 0]
    assert [r.flag for r in results] == [False, True, False, False, True, False, False]
    assert results[2].rel_loss_change == pytest.approx(0.08 / 0.52)
    assert not results[6].agrees and results[6].disagreement == pytest.approx(0.2)


def test_local_tuning_picks_a_grid_setting_on_training_data(federation):
    _, clients, _ = federation
    c = clients[0]
    c.cfg = dict(c.cfg, local_tuning={"grid": {"min_samples_leaf": [1, 8]}, "folds": 3, "cv_trees": 5})
    trees = c.grow(1, 3, 2.0)
    params, cv_score, fits = c.tuning
    assert params["min_samples_leaf"] in (1, 8) and params["max_features"] == "sqrt"
    assert 0 <= cv_score <= 1 and fits == 6 and len(trees) == 3 and c.rf_params == params
    first = c.tuning
    c.tune(1)  # deterministic for the same data and round: served from the cache
    assert c.tuning == first
