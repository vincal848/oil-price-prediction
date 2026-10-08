"""The evaluation must find nothing in noise and find a planted signal."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecast import (
    Origins,
    evaluate,
    newey_west_dm,
    protocol,
    sign_pnl,
    walk_forward,
)


def ar1_prices(n: int, phi: float, seed: int, permute: bool = False):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.01, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    if permute:
        r = rng.permutation(r)
    dates = pd.bdate_range("2000-01-03", periods=n)
    return 50 * np.exp(np.cumsum(r)), dates


def skill(prices, dates, start, h=1) -> dict:
    o = walk_forward(prices, dates, h, start, len(prices) - 1, k=5, alpha=1.0)
    return evaluate(o, o.r_hat)


def test_null_permuted_returns_have_no_skill():
    skills = []
    for seed in range(10):
        p, d = ar1_prices(4000, 0.1, seed, permute=True)
        skills.append(skill(p, d, 2000)["skill"])
    assert np.mean(skills) <= 0


def test_planted_ar1_is_found():
    p, d = ar1_prices(12000, 0.1, seed=0)
    s = skill(p, d, 2000)
    assert s["skill"] > 0
    assert s["dm_p"] < 0.05  # and the DM test agrees it is real


def test_no_lookahead_in_forecasts():
    """Scaling prices from day T must not move any forecast made at or before T."""
    p, d = ar1_prices(3000, 0.1, seed=1)
    base = walk_forward(p, d, 5, 1500, 2999)
    # T sits 5 days after a quarter start, so that quarter's refit has targets
    # (j + 5 > f) that reach past its refit date but not past T.
    quarter = d.to_period("Q")
    s = (
        int(
            np.flatnonzero(quarter[1:] != quarter[:-1])[
                np.searchsorted(np.flatnonzero(quarter[1:] != quarter[:-1]), 2000)
            ]
        )
        + 1
    )
    T = s + 5
    q = p.copy()
    q[T:] *= 3.0
    alt = walk_forward(q, d, 5, 1500, 2999)
    early = base.idx < T
    assert early.any()
    np.testing.assert_allclose(base.r_hat[early], alt.r_hat[early])


def test_horizon_targets_do_not_overlap():
    p, d = ar1_prices(1000, 0.0, seed=2)
    o = walk_forward(p, d, 20, 500, 999)
    assert (np.diff(o.idx) == 20).all()


def test_zero_forecast_is_persistence_exactly():
    p, d = ar1_prices(1000, 0.1, seed=3)
    o = walk_forward(p, d, 1, 500, 999)
    s = evaluate(o, np.zeros_like(o.r_hat))
    assert s["skill"] == 0 and s["dir_n"] == 0


def test_nonpositive_price_is_skipped_not_propagated():
    p, d = ar1_prices(1000, 0.1, seed=4)
    p[700] = -5.0
    o = walk_forward(p, d, 1, 500, 999)
    s = evaluate(o, o.r_hat)
    assert s["n"] == len(o.idx) - 2 and np.isfinite(s["rmse"])


def test_dm_detects_a_shift_and_not_noise():
    rng = np.random.default_rng(0)
    assert newey_west_dm(rng.normal(0, 1, 2000))[1] > 0.01
    assert newey_west_dm(rng.normal(0.2, 1, 2000))[1] < 0.001


def test_sign_pnl_charges_entry_and_roll():
    n = 252
    o = Origins(np.arange(n), np.ones(n), np.ones(n), np.zeros(n), np.ones(n))
    flat = sign_pnl(o, np.zeros(n, dtype=bool))
    assert flat["net_ann_pct"] == pytest.approx(-2e-4 / n * 252 * 100)  # one entry side
    rolls = np.zeros(n, dtype=bool)
    rolls[::21] = True
    assert sign_pnl(o, rolls)["net_ann_pct"] < flat["net_ann_pct"]


def test_protocol_counts_configs():
    p, d = ar1_prices(6300, 0.0, seed=5)
    out = protocol(p, d, np.zeros(len(p), dtype=bool))
    assert out["n_configs"] == 3 * (15 + 1)
    assert set(out["horizons"]) == {1, 5, 20}
    assert "sign_pnl" in out["horizons"][1]
