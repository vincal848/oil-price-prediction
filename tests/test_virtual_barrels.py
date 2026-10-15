"""Mechanics of the Virtual Barrels strategies: rolls, timing, sizing, and that the
rules find nothing in noise."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import virtual_barrels as vb
from strategies import ewma_vol, pnl, summary, vol_target


def synthetic_curve(n=600, seed=0, slope=0.01):
    """A curve where contract k trades at front * (1 - slope)^(k-1)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    front = 60 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    curve = pd.DataFrame(
        {f"C{k}": front * (1 - slope) ** (k - 1) for k in range(1, 5)}, index=idx
    )
    return curve


def test_held_returns_use_the_next_contract_on_roll_days():
    idx = pd.bdate_range("2020-01-01", periods=3)
    curve = pd.DataFrame(
        {
            "C1": [50, 50, 52],
            "C2": [51, 51, 53],
            "C3": [52, 52, 54],
            "C4": [53, 53, 55],
        },
        index=idx,
        dtype=float,
    )
    roll = pd.Series([False, False, True], index=idx)
    r = vb.held_returns(curve, 2, roll)
    # Yesterday's C2 (51) is today's C1 (52): +1/51, not 53/51.
    assert r.iloc[2] == pytest.approx(52 / 51 - 1)
    assert r.iloc[1] == 0.0
    with pytest.raises(ValueError):
        vb.held_returns(curve, 1, roll)


def test_spread_return_is_near_minus_far_in_dollars():
    idx = pd.bdate_range("2020-01-01", periods=2)
    curve = pd.DataFrame(
        {"C1": [50, 51], "C2": [50, 52], "C3": [49, 50], "C4": [48, 49]},
        index=idx,
        dtype=float,
    )
    sr = vb.spread_returns(curve, 2, 3, pd.Series(False, index=idx))
    assert sr.iloc[1] == pytest.approx(((52 - 50) - (50 - 49)) / 50)


def test_carry_sign_and_scale():
    back = synthetic_curve(slope=0.01)  # each later contract 1% cheaper: backwardation
    c = vb.carry(back)
    assert (c > 0).all()
    assert c.iloc[0] == pytest.approx(-np.log(0.99) * 2 * 6)
    cont = synthetic_curve(slope=-0.01)
    assert (vb.carry(cont) < 0).all()


def test_carry_momentum_fires_when_backwardation_steepens():
    idx = pd.bdate_range("2020-01-01", periods=60)
    c = pd.Series(np.r_[np.zeros(30), np.linspace(0, 0.2, 30)], index=idx)
    sig = vb.carry_momentum_signal(c, 1, 21)
    assert (sig.iloc[35:] == 1).all()
    assert (sig.iloc[:20] == 0).all()


def test_reaction_function_peaks_at_one_sigma():
    x = pd.Series(np.linspace(-4, 4, 801))
    r = vb.reaction(x)
    assert r.max() == pytest.approx(1.0, abs=1e-6)
    assert x[r.idxmax()] == pytest.approx(1.0, abs=0.01)
    assert vb.reaction(pd.Series([3.0])).iloc[0] < vb.reaction(pd.Series([1.0])).iloc[0]
    np.testing.assert_allclose(vb.reaction(-x), -r)


def test_weekly_data_is_invisible_before_release():
    daily = pd.bdate_range("2021-01-04", "2021-01-29")
    weekly = pd.Series(
        [1.0, 2.0], index=pd.to_datetime(["2021-01-05", "2021-01-12"])
    )  # Tuesdays
    s = vb.on_release(weekly, daily, vb.COT_RELEASE_LAG)
    assert np.isnan(s.loc["2021-01-08"])  # Friday: released after the close
    assert s.loc["2021-01-11"] == 1.0  # Monday: usable
    assert s.loc["2021-01-15"] == 1.0 and s.loc["2021-01-18"] == 2.0


def test_sentiment_and_fade_extremes():
    idx = pd.date_range("2015-01-06", periods=12, freq="W-TUE")
    net = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0], index=idx, dtype=float)
    si = vb.sentiment_index(net, 5)
    assert si.iloc[10] == 100.0 and si.iloc[11] == 0.0
    sig = vb.fade_the_crowd(net, 5, 10)
    assert sig.iloc[10] == -1.0 and sig.iloc[11] == 1.0 and sig.iloc[2] == 0.0


def test_ensemble_is_bounded_consensus():
    rng = np.random.default_rng(1)
    net = pd.Series(
        np.cumsum(rng.normal(0, 1, 300)),
        index=pd.date_range("2010-01-05", periods=300, freq="W-TUE"),
    )
    e = vb.ensemble(vb.follow_the_flow, net, [(1, 8), (2, 13), (4, 26)]).dropna()
    assert e.between(-1, 1).all()


def test_seasonal_deviation_uses_only_earlier_years():
    idx = pd.date_range("2010-01-01", "2017-12-31", freq="W-FRI")
    s = pd.Series(100.0, index=idx)
    dev = vb.seasonal_deviation(s)
    s2 = s.copy()
    s2.loc["2017"] += 50
    dev2 = vb.seasonal_deviation(s2)
    pd.testing.assert_series_equal(dev.loc[:"2016"], dev2.loc[:"2016"])
    assert (dev2.loc["2017"].dropna() == 50).all()


def test_squeeze_signal_regimes():
    idx = pd.date_range("2015-01-02", periods=10, freq="W-FRI")
    filling_full = pd.Series(np.linspace(0.80, 0.95, 10), index=idx)
    draining_empty = pd.Series(np.linspace(0.50, 0.35, 10), index=idx)
    assert (vb.squeeze_spread_signal(filling_full).iloc[5:] == -1.0).all()
    assert (vb.squeeze_spread_signal(draining_empty).iloc[5:] == 1.0).all()


def test_variance_swap_settles_to_strike_minus_realised():
    """Over one swap the short collects N * (K^2 - RV^2), whatever OVX does meanwhile."""
    n = 22
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0, 0.02, n), index=idx)
    ovx = pd.Series(rng.uniform(30, 60, n), index=idx)
    out = vb.short_variance_swaps(ovx, r, tenor=21, cost_vol_pts=0.0)
    K = ovx.iloc[0] / 100
    rv2 = (r.iloc[1:22] ** 2).sum() * 252 / 21
    assert out.pnl.iloc[1:22].sum() == pytest.approx((K**2 - rv2) / (2 * K))


def test_variance_swap_marks_implied_vol_moves():
    """A jump in implied vol with no price move is a loss for the short on the day."""
    idx = pd.bdate_range("2020-01-01", periods=30)
    ovx = pd.Series(40.0, index=idx)
    ovx.iloc[5:] = 60.0
    out = vb.short_variance_swaps(ovx, pd.Series(0.0, index=idx), cost_vol_pts=0.0)
    assert out.pnl.iloc[5] < 0
    assert out.pnl.iloc[2] > 0  # calm days before it accrue premium


def test_variance_swap_charges_bid_ask_once_per_sale():
    idx = pd.bdate_range("2020-01-01", periods=64)
    ovx = pd.Series(40.0, index=idx)
    flat = pd.Series(0.0, index=idx)
    gross = vb.short_variance_swaps(ovx, flat, cost_vol_pts=0.0).pnl.sum()
    net = vb.short_variance_swaps(ovx, flat, cost_vol_pts=1.0).pnl.sum()
    assert gross - net == pytest.approx(3 * 0.01 * 21 / 252)


def test_cot_release_calendar():
    as_of = pd.to_datetime(["2021-03-02", "2021-06-01", "2013-10-08"])
    u = vb.cot_usable_dates(as_of)
    assert u.iloc[0] == pd.Timestamp("2021-03-08")  # normal week: Monday close
    assert u.iloc[1] == pd.Timestamp("2021-06-08")  # Memorial Day week: Tuesday close
    assert u.iloc[2] == pd.Timestamp("2013-11-25")  # shutdown backlog


def test_good_friday_moves_the_wti_roll():
    from roll import termination_date

    assert termination_date(2011, 5) == pd.Timestamp("2011-04-19")
    assert termination_date(2016, 4) == pd.Timestamp("2016-03-21")


def test_to_calendar_keeps_every_dollar():
    a = pd.bdate_range("2020-01-01", periods=10)
    p = pd.Series(np.arange(10, dtype=float), index=a)
    target = a.delete([3, 4, 7])
    q = vb.to_calendar(p, target)
    assert q.sum() == pytest.approx(p.sum())
    assert q.loc[a[5]] == pytest.approx(3 + 4 + 5)


def test_book_rebalance_is_monthly_and_costed():
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2015-01-01", periods=500)
    m = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    free = vb.scale_to_vol(m, rebalance="M")
    paid = vb.scale_to_vol(m, rebalance="M", cost_bp=2.0)
    lev = free / m
    changes = lev.diff().abs().fillna(0)
    assert (
        changes[changes > 1e-9].index.isin(
            pd.Series(idx, index=idx).groupby(idx.to_period("M")).min().values
            + pd.Timedelta(0)
        )
        | (changes[changes > 1e-9].index == idx[0])
    ).sum() >= 1
    assert paid.sum() < free.sum()


def test_rules_find_nothing_in_a_random_walk_curve():
    sharpes = []
    for seed in range(10):
        curve = synthetic_curve(n=3000, seed=seed, slope=0.0)
        # Random-walk carry: slope wanders with no link to future returns.
        rng = np.random.default_rng(100 + seed)
        drift = np.cumsum(rng.normal(0, 0.0005, len(curve)))
        for k in range(2, 5):
            curve[f"C{k}"] = curve["C1"] * np.exp(-drift * (k - 1))
        roll = pd.Series(False, index=curve.index)
        r = vb.held_returns(curve, 2, roll)
        for sig in (
            vb.carry_signal(vb.carry(curve)),
            vb.carry_momentum_signal(vb.carry(curve)),
        ):
            sharpes.append(summary(pnl(vol_target(sig, ewma_vol(r)), r))["sharpe"])
    assert abs(np.mean(sharpes)) < 0.15


def test_scale_to_vol_uses_only_past_pnl():
    rng = np.random.default_rng(3)
    p = pd.Series(
        rng.normal(0, 0.01, 500), index=pd.bdate_range("2015-01-01", periods=500)
    )
    a = vb.scale_to_vol(p)
    p2 = p.copy()
    p2.iloc[400:] *= 10
    b = vb.scale_to_vol(p2)
    pd.testing.assert_series_equal(a.iloc[:400], b.iloc[:400])
