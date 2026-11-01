"""Bounded-update enforcement (fedrf_diff/bounds.py and Stage 1.6 of the pipeline)."""

import math

import numpy as np
import pytest

from fedrf_diff import bounds, pipeline, run
from test_pipeline import CLIENTS, small_config


# --- the bounds themselves ----------------------------------------------------------
def _diffusion_like(n: int, d: int, seed: int = 0) -> np.ndarray:
    """sqrt(n) * V * Lambda with orthonormal V and 0 <= lambda <= 1, as the encoder produces."""
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(rng.normal(size=(n, d)))
    return math.sqrt(n) * V * rng.uniform(0.2, 1.0, size=d)


def test_no_dp_bound_accepts_honest_and_rejects_oversized_updates():
    n_pat, n_land, d = 200, 40, 4
    Z = _diffusion_like(n_pat + n_land, d)
    y = np.random.default_rng(1).integers(0, 2, n_pat)
    bound = bounds.bound_none(n_pat + n_land, n_land, d)
    ok = bounds.check_update(Z[:n_pat], Z[n_pat:], y, bound)
    assert ok.within and ok.outcome == "Within bound" and bounds.usage(ok, bound) <= 100
    big = Z * 30.0
    out = bounds.check_update(big[:n_pat], big[n_pat:], y, bound)
    assert not out.within and any("||Z||_F^2" in r for r in out.reasons)


@pytest.mark.parametrize("defect", ["nan", "columns", "landmarks", "labels", "label_count"])
def test_malformed_updates_are_out_of_bound(defect):
    n_pat, n_land, d = 60, 10, 4
    Z = _diffusion_like(n_pat + n_land, d, seed=2)
    Zp, P, y = Z[:n_pat].copy(), Z[n_pat:].copy(), np.zeros(n_pat, dtype=int)
    if defect == "nan":
        Zp[0, 0] = np.nan
    elif defect == "columns":
        Zp, P = Zp[:, :3], P[:, :3]
    elif defect == "landmarks":
        P = P[:5]
    elif defect == "labels":
        y[0] = 2
    else:
        y = y[:-1]
    assert not bounds.check_update(Zp, P, y, bounds.bound_none(n_pat + n_land, n_land, d)).within


@pytest.mark.parametrize("scale", [2.0, 1.0, 2 / 3, 0.4])  # Laplace b for epsilon = 1, 2, 3, 5 with C = 1
def test_laplace_bound_holds_for_honest_noisy_updates(scale):
    rng = np.random.default_rng(3)
    n_pat, n_land, d, C = 88, 100, 4, 1.0
    bound = bounds.bound_laplace(n_pat, n_land, d, C, scale)
    for _ in range(200):
        Z = rng.normal(size=(n_pat + n_land, d))
        Z = Z * np.minimum(1.0, C / np.abs(Z).sum(axis=1, keepdims=True))  # L1 clip
        Zp = Z[:n_pat] + rng.laplace(0.0, scale, size=(n_pat, d))
        assert bounds.check_update(Zp, Z[n_pat:], np.zeros(n_pat, int), bound).within
    # Three times larger than any honest release, or an unclipped landmark row: out of bound.
    assert not bounds.check_update(3 * Zp, Z[n_pat:], np.zeros(n_pat, int), bound).within
    P = Z[n_pat:].copy()
    P[0] *= 1.5 / np.abs(P[0]).sum()
    assert not bounds.check_update(Zp, P, np.zeros(n_pat, int), bound).within


def test_gaussian_bound_checks_the_declared_release_parameters():
    rng = np.random.default_rng(4)
    n_pat, n_land, d, C, ratio = 140, 100, 4, 1.3, 0.6
    Z = rng.normal(size=(n_pat + n_land, d))
    Z = Z * np.minimum(1.0, C / np.linalg.norm(Z, axis=1, keepdims=True))  # L2 clip
    Zp = Z[:n_pat] + rng.normal(0.0, ratio * C, size=(n_pat, d))
    y = np.zeros(n_pat, int)
    honest = bounds.bound_gaussian(n_pat, n_land, d, C, ratio * C, noise_ratio=ratio)
    assert bounds.check_update(Zp, Z[n_pat:], y, honest).within
    wrong_sigma = bounds.bound_gaussian(n_pat, n_land, d, C, 5 * ratio * C, noise_ratio=ratio)
    assert not bounds.check_update(Zp, Z[n_pat:], y, wrong_sigma).within
    huge_clip = 10 * math.sqrt((n_pat + n_land) * d)
    too_large = bounds.bound_gaussian(n_pat, n_land, d, huge_clip, ratio * huge_clip, noise_ratio=ratio)
    assert not bounds.check_update(Zp, Z[n_pat:], y, too_large).within


# --- Stage 1.6 in the pipeline ---------------------------------------------------------
def test_honest_run_stays_within_bounds_and_learns_the_same():
    with_bounds = run(small_config(max_rounds=2), verbose=False)
    log = with_bounds.bounds_log
    assert log is not None and len(log) == 2 * 4
    assert log["within"].all() and (log["usage_pct"] <= 100).all()
    assert (with_bounds.history["n_out_of_bound"] == 0).all() and with_bounds.rejected == {}
    server = with_bounds.costs[with_bounds.costs["client"] == "Global Server"]
    assert (server["bound_check_s"] > 0).all()
    without = run(small_config(max_rounds=2), verbose=False, bounds_enabled=False)
    assert without.bounds_log is None
    np.testing.assert_allclose(with_bounds.history["L_avg"], without.history["L_avg"])


def test_out_of_bound_update_is_rejected_before_aggregation(monkeypatch):
    """An update the check marks out of bound is dropped for that round and never aggregated."""
    real, calls = bounds.check_update, {"n": 0}

    def marked(Z, P, y, bound):
        calls["n"] += 1
        check = real(Z, P, y, bound)
        if calls["n"] == 4 + 2:  # round 2, second client in order
            return bounds.BoundCheck(within=False, reasons=["marked out of bound (test)"], stats=check.stats)
        return check

    monkeypatch.setattr(pipeline.bounds, "check_update", marked)
    result = run(small_config(max_rounds=2), verbose=False)
    assert result.rejected == {2: [CLIENTS[1]]}
    assert list(result.history["n_accepted"]) == [4, 3]
    assert list(result.history["n_out_of_bound"]) == [0, 1]
    log = result.bounds_log.set_index(["round", "client"])
    assert not log.loc[(2, CLIENTS[1]), "within"] and log["within"].sum() == 7
    assert f"loss[{CLIENTS[1]}]" not in result.history.columns or np.isnan(
        result.history.loc[1, f"loss[{CLIENTS[1]}]"])  # no metrics from the rejected client in round 2
