"""The rules must not see the future, must find a planted trend, and must find
nothing in a random walk."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import (
    carry_signal,
    daily_returns,
    ewma_vol,
    lookback_grid,
    pnl,
    rebalance,
    summary,
    trend_signal,
    vol_target,
)


def prices_from_returns(r: np.ndarray, start: str = "2000-01-03") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(r))
    return pd.Series(50 * np.exp(np.cumsum(r)), index=idx)


def regime_returns(
    n: int, seed: int, drift: float = 0.0015, block: int = 250
) -> np.ndarray:
    """Persistent up/down regimes: the textbook case where trend should work."""
    rng = np.random.default_rng(seed)
    signs = np.repeat(rng.choice([-1.0, 1.0], size=n // block + 1), block)[:n]
    return signs * drift + rng.normal(0, 0.015, n)


def test_negative_price_return_is_zero_not_nan():
    p = pd.Series(
        [18.27, -37.63, 10.01, 12.0], index=pd.bdate_range("2020-04-17", periods=4)
    )
    r = daily_returns(p)
    assert r.iloc[1] == 0.0 and r.iloc[2] == 0.0
    assert r.iloc[3] == pytest.approx(12.0 / 10.01 - 1)


def test_roll_days_are_zeroed():
    p = prices_from_returns(np.full(10, 0.01))
    roll = pd.Series(False, index=p.index)
    roll.iloc[5] = True
    r = daily_returns(p, roll)
    assert r.iloc[5] == 0.0 and r.iloc[6] > 0


def test_position_at_t_earns_return_of_t_plus_1():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.0, 0.10, -0.05, 0.02], index=idx)
    pos = pd.Series([1.0, 0.0, 1.0, 1.0], index=idx)
    out = pnl(pos, r, cost_bp=0.0)
    assert out.tolist() == pytest.approx([0.0, 0.10, 0.0, 0.02])


def test_no_lookahead_changing_future_prices_leaves_past_positions_alone():
    r = regime_returns(1500, seed=1)
    p = prices_from_returns(r)
    cut = 1000
    p2 = p.copy()
    p2.iloc[cut:] *= np.linspace(1.0, 3.0, len(p) - cut)
    for lb in (21, 252):
        for f in ("D", "W", "M"):
            a = rebalance(trend_signal(daily_returns(p), lb), f)
            b = rebalance(trend_signal(daily_returns(p2), lb), f)
            # Monthly holds the last month-end value, which is before the cut.
            assert a.iloc[: cut - 1].equals(b.iloc[: cut - 1])
    v1, v2 = ewma_vol(daily_returns(p)), ewma_vol(daily_returns(p2))
    pd.testing.assert_series_equal(v1.iloc[:cut], v2.iloc[:cut])


def test_carry_uses_yesterdays_spot():
    idx = pd.bdate_range("2020-01-01", periods=6)
    fut = pd.Series(50.0, index=idx)
    spot = pd.Series([49, 49, 49, 51, 51, 51], index=idx, dtype=float)
    sig = carry_signal(fut, spot, smooth=1)
    # Spot jumps above the future on day 3; the signal can only flip on day 4.
    assert sig.iloc[3] == -1.0 and sig.iloc[4] == 1.0


def test_monthly_rebalance_only_changes_on_month_ends():
    idx = pd.bdate_range("2021-01-01", "2021-06-30")
    sig = pd.Series(np.random.default_rng(0).choice([-1.0, 1.0], len(idx)), index=idx)
    held = rebalance(sig, "M")
    changes = held.index[held.diff().fillna(0) != 0]
    month_ends = pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().values
    assert set(changes).issubset(set(month_ends))


def test_vol_target_hits_target_on_iid_noise():
    rng = np.random.default_rng(3)
    r = pd.Series(
        rng.normal(0, 0.02, 4000), index=pd.bdate_range("2000-01-03", periods=4000)
    )
    pos = vol_target(pd.Series(1.0, index=r.index), ewma_vol(r), target=0.20)
    realised = summary(pnl(pos, r, cost_bp=0.0).iloc[300:])["ann_vol"]
    assert realised == pytest.approx(0.20, rel=0.10)


def test_trend_finds_planted_regimes():
    sharpes = []
    for seed in range(5):
        r = pd.Series(
            regime_returns(5000, seed), index=pd.bdate_range("2000-01-03", periods=5000)
        )
        pos = vol_target(rebalance(trend_signal(r, 63), "W"), ewma_vol(r))
        sharpes.append(summary(pnl(pos, r))["sharpe"])
    assert np.mean(sharpes) > 0.5


def test_trend_finds_nothing_in_a_random_walk():
    sharpes = []
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        r = pd.Series(
            rng.normal(0, 0.02, 5000), index=pd.bdate_range("2000-01-03", periods=5000)
        )
        pos = vol_target(rebalance(trend_signal(r, 63), "W"), ewma_vol(r))
        sharpes.append(summary(pnl(pos, r))["sharpe"])
    # Mean across seeds sits near zero (slightly negative after costs).
    assert abs(np.mean(sharpes)) < 0.15


def test_costs_scale_with_turnover():
    rng = np.random.default_rng(7)
    r = pd.Series(
        rng.normal(0, 0.02, 2000), index=pd.bdate_range("2000-01-03", periods=2000)
    )
    fast = rebalance(trend_signal(r, 5), "D")
    slow = rebalance(trend_signal(r, 252), "M")
    drag = lambda pos: (pnl(pos, r, cost_bp=0.0) - pnl(pos, r, cost_bp=10.0)).sum()
    assert drag(fast) > 5 * drag(slow) > 0


def test_lookback_grid_shape():
    r = pd.Series(
        regime_returns(1500, 0), index=pd.bdate_range("2000-01-03", periods=1500)
    )
    g = lookback_grid(r, lookbacks=(21, 63), freqs=("D", "M"))
    assert g.shape == (2, 2) and g.notna().all().all()


def test_splice_spread_reads_the_contract_spread():
    """A splice from a front at 50 to a next contract at 51 is a 2% contango."""
    from strategies import carry_signal_spread, splice_spread

    idx = pd.bdate_range("2020-01-01", periods=6)
    spot = pd.Series([50, 50.5, 50.2, 50.4, 50.1, 50.3], index=idx)
    fut = spot.copy()
    fut.iloc[3:] *= 1.02  # contract switch on day 3, next contract 2% above
    roll = pd.Series(False, index=idx)
    roll.iloc[3] = True
    sp = splice_spread(fut, spot, roll)
    assert np.isnan(sp.iloc[3])  # not known until the next day
    assert sp.iloc[4] == pytest.approx(np.log(1.02))
    assert carry_signal_spread(fut, spot, roll).iloc[4] == -1.0


def test_month_start_roll_mask():
    from roll import month_start_roll_mask

    idx = pd.bdate_range("2021-01-01", "2021-03-31")
    m = month_start_roll_mask(idx)
    assert list(idx[m.values]) == [
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-02-01"),
        pd.Timestamp("2021-03-01"),
    ]
