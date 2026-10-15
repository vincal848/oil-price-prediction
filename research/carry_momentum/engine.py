"""Signal and position engine for the carry-momentum study (PROTOCOL.md).

Positions are held in whole 1,000-barrel contracts on a fixed notional book,
contract by contract, so that rolls, costs and capacity are computed on what
would actually be held. Pure: no I/O.

Timing (protocol "Execution"): the signal from day t's settlement sets a target;
the target is traded at day t+1's settlement (TAS) and so first earns from t+1
to t+2. Holdings at the close of day d therefore equal the target computed at
the close of day d-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import curve
import numpy as np
import pandas as pd

BBL = 1000  # barrels per CL contract
TICK = 10.0  # $ per contract per $0.01
FEES = 1.50  # exchange + clearing, $ per contract per side
TAS_PREMIUM = TICK / 2


@dataclass(frozen=True)
class Costs:
    """Per-contract costs ($). Defaults are the protocol's."""

    outright: float = TICK + FEES + TAS_PREMIUM  # one contract, one side
    spread: float = TICK + 2 * (
        FEES + TAS_PREMIUM
    )  # one exchange calendar spread: two contracts

    def scaled(self, k: float) -> Costs:
        return Costs(self.outright * k, self.spread * k)


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


def carry(cm: pd.DataFrame, near: str = "1m", far: str = "13m") -> pd.Series:
    """ln(F_near / F_far), annualised by the tenor gap. Positive in backwardation."""
    gap_years = (curve.TENORS_DAYS[far] - curve.TENORS_DAYS[near]) / 365
    ok = (cm[near] > 0) & (cm[far] > 0)
    return np.log(cm[near].where(ok) / cm[far].where(ok)) / gap_years


def carry_momentum(c: pd.Series, short: int = 1, long: int = 21) -> pd.Series:
    """sign(MA_short(carry) - MA_long(carry)); 0 until the long window fills."""
    fast = c.rolling(short, min_periods=short).mean()
    slow = c.rolling(long, min_periods=long).mean()
    return np.sign(fast - slow).fillna(0.0)


# ---------------------------------------------------------------------------
# Legs: which contracts make up one unit of a leg, per day
# ---------------------------------------------------------------------------


def leg_weights(panel: pd.DataFrame, offset: int) -> pd.DataFrame:
    """dates x contracts: share of one unit of the leg held in each contract at the close.

    offset=1 is the 2nd listed contract on the GSCI roll schedule, offset=6 the 7th.
    """
    t = curve.gsci_targets(panel, offset=offset)
    rows = []
    for d, r in t.iterrows():
        if r["from"] is None:
            continue
        rows.append((d, r["from"], 1 - r["w"]))
        rows.append((d, r["to"], r["w"]))
    w = (
        pd.DataFrame(rows, columns=["date", "contract", "w"])
        .groupby(["date", "contract"])
        .w.sum()
    )
    return w.unstack(fill_value=0.0)


def unit_dollar_pnl(prices: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """$ P&L per contract-unit of a leg from d-1 to d (weights at close of d-1)."""
    w = weights.reindex(index=prices.index, columns=prices.columns, fill_value=0.0)
    dp = prices.diff().fillna(0.0)
    return (w.shift(1).fillna(0.0) * dp).sum(axis=1) * BBL


# ---------------------------------------------------------------------------
# Book
# ---------------------------------------------------------------------------


@dataclass
class Book:
    returns: pd.Series  # daily net return on capital
    gross: pd.Series  # daily gross return on capital
    holdings: pd.DataFrame  # contracts held at each close, signed, by contract
    units: pd.Series  # spread or outright units held at each close
    cost: pd.Series  # $ cost per day


def run(
    signal: pd.Series,
    legs: dict[str, tuple[float, pd.DataFrame]],
    prices: pd.DataFrame,
    *,
    capital: float = 10e6,
    target_vol: float = 0.15,
    vol_com: float = 60.0,
    costs: Costs = Costs(),
    spread_trade: bool = True,
) -> Book:
    """Simulate the strategy on whole contracts.

    legs: name -> (sign, weights); a spread is {"near": (+1, w2), "far": (-1, w7)},
    an outright {"leg": (+1, w2)}. One "unit" is one contract in each leg.
    spread_trade: changes in units are executed as exchange calendar spreads
    (cost `costs.spread` per unit) rather than outright legs.
    """
    idx = prices.index
    unit_pnl = sum(sign * unit_dollar_pnl(prices, w) for sign, w in legs.values())
    # Ex-ante $ vol of one unit, from P&L through day t.
    vol = np.sqrt((unit_pnl**2).ewm(com=vol_com, min_periods=20).mean() * 252)
    size = (target_vol * capital / vol).replace([np.inf, -np.inf], np.nan)
    target = (signal.reindex(idx).fillna(0.0) * size).round().fillna(0.0)
    units = target.shift(1).fillna(0.0)  # traded at the next settlement

    # Contract holdings: split each leg's units across its from/to contracts.
    hold = {}
    for name, (sign, w) in legs.items():
        w = w.reindex(idx).fillna(0.0)
        n = units * sign
        # Round the 'to' share, give the rest to 'from', so totals stay whole.
        h = w.mul(n, axis=0)
        h = h.round()
        resid = n - h.sum(axis=1)
        first = w.idxmax(axis=1)
        for d in resid.index[resid != 0]:
            h.at[d, first.at[d]] += resid.at[d]
        hold[name] = h
    holdings = (
        pd.concat(hold.values(), axis=1)
        .T.groupby(level=0)
        .sum()
        .T.reindex(idx)
        .fillna(0.0)
    )

    dp = prices.reindex(columns=holdings.columns).diff().fillna(0.0)
    gross_dollars = (holdings.shift(1).fillna(0.0) * dp).sum(axis=1) * BBL

    # Costs: unit changes as spreads (or outright legs), plus rolls as spreads.
    d_units = units.diff().abs().fillna(units.abs())
    n_legs = len(legs)
    if spread_trade and n_legs > 1:
        unit_cost = d_units * costs.spread
    else:
        unit_cost = d_units * n_legs * costs.outright
    traded = holdings.diff().abs().fillna(holdings.abs()).sum(axis=1)
    rolled = ((traded - d_units * n_legs) / 2).clip(lower=0)
    cost = unit_cost + rolled * costs.spread

    return Book(
        returns=(gross_dollars - cost) / capital,
        gross=gross_dollars / capital,
        holdings=holdings,
        units=units,
        cost=cost,
    )


def capacity(
    book: Book,
    volume: pd.DataFrame,
    oi: pd.DataFrame,
    capital: float = 10e6,
    adv_limit: float = 0.05,
    oi_limit: float = 0.02,
    quantile: float | None = None,
) -> dict:
    """Largest book (in $ notional) meeting the participation limits.

    Ratios are computed at `capital` and scale linearly. quantile=None uses the
    worst day (the protocol's literal reading); a number uses that quantile of
    daily worst-contract ratios.
    """
    h = book.holdings
    adv = (
        volume.reindex(index=h.index, columns=h.columns)
        .rolling(20, min_periods=5)
        .mean()
    )
    o = oi.reindex(index=h.index, columns=h.columns)
    trade = h.diff().abs()
    tr = (trade / adv).where(trade > 0).max(axis=1)
    pr = (h.abs() / o).where(h != 0).max(axis=1)
    agg = (lambda s: s.max()) if quantile is None else (lambda s: s.quantile(quantile))
    t_ratio, p_ratio = (
        agg(tr.replace(np.inf, np.nan).dropna()),
        agg(pr.replace(np.inf, np.nan).dropna()),
    )
    cap = capital * min(adv_limit / t_ratio, oi_limit / p_ratio)
    return {
        "capacity": float(cap),
        "trade_over_adv": float(t_ratio),
        "position_over_oi": float(p_ratio),
        "days_untradeable": int(
            np.isinf((trade / adv).where(trade > 0).max(axis=1)).sum()
        ),
    }
