"""Systematic futures rules on the energy complex: trend, carry, volatility targeting.

The forecasting work in this repo asks "what is tomorrow's price?" and finds the
answer is "today's price". These rules ask a different question: "what position
should I hold?" They do not need to forecast the next day. Trend bets that a move
already under way continues over weeks to months; carry collects the premium the
term structure pays for holding the contract; volatility targeting keeps the risk
roughly constant, which matters because oil volatility clusters.

Pure: no I/O. Every function takes daily prices (or returns) indexed by date and
returns a Series on the same index.

Timing convention, used everywhere: a signal computed at the close of day t sets
the position held over day t+1. `pnl` applies that one-day shift, so no rule can
trade on the close it was computed from.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
COST_BP = 2.0  # per side, per unit of notional traded (as in forecast.py)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def daily_returns(prices: pd.Series, roll_days: pd.Series | None = None) -> pd.Series:
    """Simple daily returns of a continuous front-month series.

    Returns touching a non-positive price (WTI on 2020-04-20) are set to 0: a
    percentage return is undefined there, and a real strategy would have rolled
    out of the expiring May contract days earlier.

    If `roll_days` is given (True on the first day after a contract splice), those
    returns are also set to 0. The splice mixes a real move with the spread between
    two contracts, and the spread part is not a return anyone earned.
    """
    p = pd.Series(prices, dtype=float)
    prev = p.shift(1)
    r = (p / prev - 1.0).where((p > 0) & (prev > 0))
    if roll_days is not None:
        r = r.mask(roll_days.reindex(r.index, fill_value=False).astype(bool))
    return r.fillna(0.0)


def ewma_vol(returns: pd.Series, com: float = 60.0) -> pd.Series:
    """Annualised ex-ante volatility, exponentially weighted (centre of mass in days).

    com = 60 follows Moskowitz, Ooi and Pedersen (2012). Uses data through day t.
    """
    var = (returns**2).ewm(com=com, min_periods=20).mean()
    return np.sqrt(var * TRADING_DAYS)


# ---------------------------------------------------------------------------
# Signals (computed at the close of t, using data through t)
# ---------------------------------------------------------------------------


def trend_signal(returns: pd.Series, lookback: int) -> pd.Series:
    """Time-series momentum: the sign of the trailing `lookback`-day return."""
    trailing = np.log1p(returns).rolling(lookback, min_periods=lookback).sum()
    return np.sign(trailing).fillna(0.0)


def carry_signal(futures: pd.Series, spot: pd.Series, smooth: int = 5) -> pd.Series:
    """Long in backwardation, short in contango.

    basis = log(front future / spot). Front above spot (contango) means the
    market pays you to store; holding the future then loses the roll as it
    converges down to spot, so the rule goes short. Spot is lagged one day,
    because FRED/EIA spot prints are published after the futures close.
    """
    s = spot.reindex(futures.index).ffill(limit=5).shift(1)
    ok = (futures > 0) & (s > 0)
    basis = np.log(futures.where(ok) / s.where(ok))
    return (-np.sign(basis.rolling(smooth, min_periods=1).mean())).fillna(0.0)


def splice_spread(
    futures: pd.Series, spot: pd.Series, roll_days: pd.Series
) -> pd.Series:
    """Front-to-next contract spread, read off each splice, carried forward.

    A continuous front-month series has no second contract in it, but it does
    switch contracts on every roll day. On that day the futures-minus-spot basis
    jumps by roughly log(next / front): the slope of the front of the curve. The
    day's own price move cancels because spot moves with it. Positive = contango.
    Known at the close of the splice day; used from the next day on.
    """
    s = spot.reindex(futures.index).ffill(limit=5)
    ok = (futures > 0) & (s > 0)
    jump = np.log(futures.where(ok) / s.where(ok)).diff()
    rolls = roll_days.reindex(futures.index, fill_value=False).astype(bool)
    return jump.where(rolls).ffill().shift(1)


def carry_signal_spread(
    futures: pd.Series, spot: pd.Series, roll_days: pd.Series
) -> pd.Series:
    """Carry from the front-to-next spread: long in backwardation, short in contango."""
    return (-np.sign(splice_spread(futures, spot, roll_days))).fillna(0.0)


def rebalance(signal: pd.Series, freq: str = "D") -> pd.Series:
    """Hold the signal fixed between rebalance dates.

    "D" daily, "W" weekly (last trading day of each week), "M" monthly (last
    trading day of each month). The value set on a rebalance date is carried
    forward until the next one.
    """
    if freq == "D":
        return signal
    key = {"W": signal.index.to_period("W"), "M": signal.index.to_period("M")}[freq]
    last_day = pd.Series(signal.index, index=signal.index).groupby(key).transform("max")
    on_rebalance = signal.index == last_day.values
    return signal.where(on_rebalance).ffill().fillna(0.0)


def vol_target(
    position: pd.Series, vol: pd.Series, target: float = 0.20, cap: float = 4.0
) -> pd.Series:
    """Scale a position so its ex-ante volatility is `target` (annualised)."""
    lev = (target / vol).clip(upper=cap)
    return (position * lev).fillna(0.0)


# ---------------------------------------------------------------------------
# P&L and summary statistics
# ---------------------------------------------------------------------------


def pnl(
    position: pd.Series,
    returns: pd.Series,
    cost_bp: float = COST_BP,
    roll_days: pd.Series | None = None,
) -> pd.Series:
    """Daily net strategy return. Position set at close t earns the return of t+1.

    Costs: `cost_bp` per side on every unit of notional traded, plus a close and
    reopen of the whole position on contract-roll days when `roll_days` is given.
    """
    held = position.shift(1).fillna(0.0)
    gross = held * returns
    traded = position.diff().abs().fillna(position.abs())
    cost = traded * cost_bp / 1e4
    if roll_days is not None:
        rolls = roll_days.reindex(position.index, fill_value=False).astype(bool)
        cost = cost + (held.abs() * 2 * cost_bp / 1e4).where(rolls, 0.0)
    return (gross - cost.shift(1).fillna(0.0)).rename(position.name)


def summary(r: pd.Series) -> dict[str, float]:
    """Annualised return, volatility, Sharpe, max drawdown and hit rate."""
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return {
            "ann_ret": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_dd": np.nan,
            "hit": np.nan,
        }
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    active = r[r.abs() > 1e-5]  # ignore days flat but for costs (< 0.1bp)
    return {
        "ann_ret": float(r.mean() * TRADING_DAYS),
        "ann_vol": float(r.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)),
        "max_dd": float(dd.min()),
        "hit": float((active > 0).mean()) if len(active) else np.nan,
    }


def turnover(position: pd.Series) -> float:
    """Average notional traded per year, in multiples of the position."""
    return float(position.diff().abs().mean() * TRADING_DAYS)


# ---------------------------------------------------------------------------
# The book of rules run in run_strategies.py
# ---------------------------------------------------------------------------


def build_positions(
    futures: pd.Series,
    returns: pd.Series,
    spot: pd.Series | None = None,
    target: float = 0.20,
    roll_days: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Positions for every rule on one contract. Keys are the rule names.

    With `roll_days`, carry uses the front-to-next spread read at each splice
    (`carry_signal_spread`); the front-vs-spot version is kept as "carry, vs spot"
    for comparison.
    """
    vol = ewma_vol(returns)
    one = pd.Series(1.0, index=returns.index)
    out = {
        "buy and hold": one,
        "buy and hold, vol-targeted": vol_target(one, vol, target),
    }
    for lb, label in [(21, "1m"), (63, "3m"), (252, "12m")]:
        sig = rebalance(trend_signal(returns, lb), "M")
        out[f"trend {label}"] = sig
        out[f"trend {label}, vol-targeted"] = vol_target(sig, vol, target)
    if spot is not None:
        vs_spot = rebalance(carry_signal(futures, spot), "M")
        out["carry, vs spot, vol-targeted"] = vol_target(vs_spot, vol, target)
        carry = (
            rebalance(carry_signal_spread(futures, spot, roll_days), "M")
            if roll_days is not None
            else vs_spot
        )
        out["carry"] = carry
        out["carry, vol-targeted"] = vol_target(carry, vol, target)
        blend = (rebalance(trend_signal(returns, 252), "M") + carry) / 2
        out["trend 12m + carry, vol-targeted"] = vol_target(blend, vol, target)
    return {k: v.rename(k) for k, v in out.items()}


def lookback_grid(
    returns: pd.Series,
    lookbacks=(5, 10, 21, 42, 63, 126, 189, 252),
    freqs=("D", "W", "M"),
    target: float = 0.20,
    roll_days: pd.Series | None = None,
) -> pd.DataFrame:
    """Net Sharpe of vol-targeted trend for every lookback x rebalance frequency."""
    vol = ewma_vol(returns)
    rows = {}
    for f in freqs:
        rows[f] = {
            lb: summary(
                pnl(
                    vol_target(rebalance(trend_signal(returns, lb), f), vol, target),
                    returns,
                    roll_days=roll_days,
                )
            )["sharpe"]
            for lb in lookbacks
        }
    return pd.DataFrame(rows).T
