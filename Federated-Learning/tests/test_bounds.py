"""Bounded-update enforcement (server.Limits bounds checked in FederatedServer.receive)."""

from fedrf_diff.pipeline import run_federation
from fedrf_diff.server import FederatedServer, Limits

CREATORS = {"client_1", "client_2", "client_3", "client_4"}


def test_bounds_follow_the_protocol_parameters(small_cfg):
    lim = Limits.from_config(small_cfg)
    assert lim.max_trees_per_client == max(small_cfg["initial_trees"], small_cfg["local_budget"]) == 6
    assert lim.new_trees_per_round == max(small_cfg["initial_trees"], small_cfg["supplementary_trees"]) == 4
    off = Limits.from_config(dict(small_cfg, bounded_update=False))
    assert off.max_trees_per_client is None and off.new_trees_per_round is None


def test_honest_upload_is_within_bound(federation):
    _, clients, server = federation
    c = clients[0]
    c.initialise(None)
    trees, rejected = server.receive(c.client_id, c.serialize(1), 1, CREATORS)
    b = server.last_bound
    assert len(trees) == 4 and rejected == []
    assert b["within"] and b["trees"] == 4 and b["new"] == 4 and b["used_pct"] <= 100 and b["check_s"] >= 0


def test_oversized_forest_is_rejected_and_registers_nothing(federation):
    _, clients, server = federation
    c = clients[0]
    c.initialise(None)
    honest = list(c.forest)
    c.forest = honest + c.grow(1, 3, None)  # 7 trees > max(initial_trees, local_budget) = 6
    trees, rejected = server.receive(c.client_id, c.serialize(1), 1, CREATORS)
    assert trees == [] and rejected[-1][0] == "<update>" and "bounded-update" in rejected[-1][1]
    assert not server.last_bound["within"] and server.known == {}
    c.forest = honest  # the honest forest is still accepted afterwards
    trees, rejected = server.receive(c.client_id, c.serialize(1), 1, CREATORS)
    assert len(trees) == 4 and rejected == [] and server.last_bound["within"]


def test_too_many_new_trees_per_round_is_rejected(federation):
    _, clients, server = federation
    c = clients[0]
    c.initialise(None)
    assert server.receive(c.client_id, c.serialize(1), 1, CREATORS)[1] == []
    c.forest = c.grow(2, 5, None)  # 5 new tree ids in one round > 4
    trees, rejected = server.receive(c.client_id, c.serialize(2), 2, CREATORS)
    assert trees == [] and "new tree ids" in rejected[-1][1]
    # a client whose round-2 upload was rejected may bring two rounds of new trees in round 3
    trees, rejected = server.receive(c.client_id, c.serialize(3), 3, CREATORS)
    assert len(trees) == 5 and rejected == [] and server.last_bound["max_new"] == 8


def test_disabled_bounds_accept_the_oversized_forest(small_cfg, federation):
    schema, clients, _ = federation
    server = FederatedServer(schema, Limits.from_config(dict(small_cfg, bounded_update=False)),
                             small_cfg["zkp_lambda"], small_cfg["aggregation_policy"], small_cfg["global_budget"])
    c = clients[0]
    c.initialise(None)
    c.forest = c.forest + c.grow(1, 3, None)
    trees, rejected = server.receive(c.client_id, c.serialize(1), 1, CREATORS)
    assert len(trees) == 7 and rejected == [] and not server.last_bound["enabled"]


def test_honest_federation_stays_within_bounds(small_cfg):
    res = run_federation(small_cfg, None, 3, with_zkp=False, verbose=False, stop_early=False)
    t = res.transfer
    assert (t["Bounded Update"] == "Within bound").all()
    assert (t["Bound Used (%)"] <= 100).all() and (t["Trees In Update"] <= t["Max Trees"]).all()
    assert (t["New Trees"] <= t["Max New Trees"]).all()
