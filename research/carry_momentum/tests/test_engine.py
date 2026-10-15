"""Mechanics of the curve and the engine on a synthetic contract panel."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import curve
import engine


def synthetic_panel(n_days=500, slope=0.01, seed=0, vol=0.02):
    """Monthly contracts on a business-day calendar; contract k trades at
    spot * (1 - slope)^(months to expiry). Expiry on the 20th of the month before."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    spot = pd.Series(60 * np.exp(np.cumsum(rng.normal(0, vol, n_days))), index=dates)
    rows = []
    for y in range(2009, 2014):
        for m in range(1, 13):
            exp = pd.Timestamp(y, m, 1) - pd.Timedelta(days=11)
            name = f"NCL{m:02d}{y % 100:02d}"
            for d in dates[(dates <= exp) & (dates >= exp - pd.Timedelta(days=420))]:
                months = (exp - d).days / 30.4
                rows.append(
                    (
                        d,
                        name,
                        exp,
                        exp,
                        spot[d] * (1 - slope) ** months,
                        1000.0,
                        50000.0,
                    )
                )
    return pd.DataFrame(
        rows,
        columns=["date", "contract", "expiry", "expiry_ds", "settle", "volume", "oi"],
    ), spot


def test_ranks_order_by_expiry():
    p, _ = synthetic_panel(60)
    r = curve.ranked(p, 3)
    for _, g in r.groupby("date"):
        assert list(g.sort_values("rank").expiry) == sorted(g.expiry)


def test_constant_maturity_interpolates_log_linearly():
    p, _ = synthetic_panel(30, slope=0.0)
    cm = curve.constant_maturity(p, {"3m": 91})
    day = p[p.date == p.date.iloc[0]]
    assert cm["3m"].iloc[0] == pytest.approx(day.settle.iloc[0], rel=1e-9)  # flat curve


def test_gsci_roll_moves_one_fifth_a_day():
    dates = pd.bdate_range("2021-03-01", "2021-03-31")
    w = curve.gsci_roll_weights(dates)
    assert list(w.iloc[:4]) == [0, 0, 0, 0]
    assert list(w.iloc[4:9]) == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert (w.iloc[9:] == 1).all()


def test_held_return_never_crosses_contracts():
    """With a flat spot and a steep curve, a rolled long earns exactly the roll-down."""
    p, _ = synthetic_panel(400, slope=0.02, vol=0.0)
    er = curve.held_excess_return(p, curve.gsci_targets(p)).dropna()
    # Each day the held contract converges toward spot: returns are small and positive.
    assert (er.iloc[30:] >= -1e-12).all()
    assert er.iloc[30:].mean() > 0


def test_carry_sign():
    cm = pd.DataFrame({"1m": [60.0], "13m": [55.0]})
    assert engine.carry(cm).iloc[0] > 0  # backwardation


def test_signal_waits_for_window():
    c = pd.Series(np.arange(30, dtype=float))
    s = engine.carry_momentum(c, 1, 21)
    assert (s.iloc[:20] == 0).all() and (s.iloc[20:] == 1).all()


def test_engine_executes_one_day_after_signal_and_holds_whole_contracts():
    p, _ = synthetic_panel(300)
    px = p.pivot(index="date", columns="contract", values="settle")
    w2 = engine.leg_weights(p, 1)
    sig = pd.Series(0.0, index=px.index)
    sig.iloc[100:] = 1.0
    b = engine.run(sig, {"leg": (1.0, w2)}, px, spread_trade=False)
    assert (b.units.iloc[:101] == 0).all() and b.units.iloc[101] > 0
    assert np.allclose(b.holdings.values, np.round(b.holdings.values))
    assert np.allclose(b.holdings.sum(axis=1), b.units)
    # First P&L day is the one after the first holding day.
    assert (b.gross.iloc[:102] == 0).all()


def test_engine_rolls_are_charged_as_spreads():
    p, _ = synthetic_panel(400)
    px = p.pivot(index="date", columns="contract", values="settle")
    w2 = engine.leg_weights(p, 1)
    sig = pd.Series(1.0, index=px.index)
    b = engine.run(sig, {"leg": (1.0, w2)}, px, spread_trade=False)
    # After sizing settles, costs occur on roll days even with a constant signal.
    late = b.cost.iloc[200:]
    assert (late > 0).sum() > 0
    zero_cost = engine.run(
        sig, {"leg": (1.0, w2)}, px, costs=engine.Costs(0, 0), spread_trade=False
    )
    assert (zero_cost.cost == 0).all()


def test_spread_book_pnl_is_near_minus_far():
    p, _ = synthetic_panel(300)
    px = p.pivot(index="date", columns="contract", values="settle")
    w2, w7 = engine.leg_weights(p, 1), engine.leg_weights(p, 6)
    near = engine.unit_dollar_pnl(px, w2)
    far = engine.unit_dollar_pnl(px, w7)
    sig = pd.Series(1.0, index=px.index)
    b = engine.run(
        sig, {"near": (1.0, w2), "far": (-1.0, w7)}, px, costs=engine.Costs(0, 0)
    )
    expected = (b.units.shift(1).fillna(0) * (near - far)) / 10e6
    # Equal only up to the rounding of each leg across its two contracts mid-roll.
    assert np.corrcoef(b.gross.iloc[50:], expected.iloc[50:])[0, 1] > 0.99


def test_no_lookahead_in_engine():
    p, _ = synthetic_panel(400)
    px = p.pivot(index="date", columns="contract", values="settle")
    w2 = engine.leg_weights(p, 1)
    sig = pd.Series(np.sign(np.sin(np.arange(len(px)) / 15)), index=px.index)
    a = engine.run(sig, {"leg": (1.0, w2)}, px, spread_trade=False).returns
    px2 = px.copy()
    px2.iloc[250:] *= 2.0
    sig2 = sig.copy()
    sig2.iloc[250:] *= -1
    b = engine.run(sig2, {"leg": (1.0, w2)}, px2, spread_trade=False).returns
    pd.testing.assert_series_equal(a.iloc[:250], b.iloc[:250])


def test_capacity_scales_with_limits():
    p, _ = synthetic_panel(300)
    px = p.pivot(index="date", columns="contract", values="settle")
    vol = p.pivot(index="date", columns="contract", values="volume")
    oi = p.pivot(index="date", columns="contract", values="oi")
    w2 = engine.leg_weights(p, 1)
    b = engine.run(
        pd.Series(1.0, index=px.index), {"leg": (1.0, w2)}, px, spread_trade=False
    )
    c1 = engine.capacity(b, vol, oi)
    c2 = engine.capacity(b, vol, oi, adv_limit=0.10, oi_limit=0.04)
    assert c2["capacity"] == pytest.approx(2 * c1["capacity"])


def test_hysteresis_holds_inside_band_and_flips_outside():
    z = pd.Series([0.0, 0.6, 0.2, -0.3, -0.7, 0.1, np.nan, 0.8])
    assert engine.hysteresis(z, 0.5).tolist() == [0, 1, 1, 1, -1, -1, -1, 1]


def test_buffer_skips_small_rebalances_but_not_flips():
    t = pd.Series([10.0, 11.0, 12.0, 14.0, -3.0])
    assert engine.buffered(t, 0.25).tolist() == [10, 10, 10, 14, -3]
    assert engine.buffered(t, 0.0).tolist() == [10, 11, 12, 14, -3]


def test_buffer_cuts_turnover_on_a_real_book():
    p, _ = synthetic_panel(n_days=400)
    px = p.pivot(index="date", columns="contract", values="settle")
    w = engine.leg_weights(p, 1)
    sig = pd.Series(1.0, index=px.index)
    plain = engine.run(sig, {"leg": (1.0, w)}, px, spread_trade=False)
    buf = engine.run(sig, {"leg": (1.0, w)}, px, spread_trade=False, buffer=0.25)
    assert buf.units.diff().abs().sum() < plain.units.diff().abs().sum()
