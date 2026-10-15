"""The WTI futures curve built from individual NYMEX CL contracts.

Input: one row per (date, contract) with settlement, volume, open interest and
last trading day, as delivered by LSEG Datastream through WRDS (see DATA.md).
Everything here is pure: no I/O, no global state.

Conventions
-----------
- A contract is *listed* on a date if it has a settlement and the date is on or
  before its last trading day.
- Rank 1 is the listed contract with the earliest expiry (EIA's "contract 1").
- Days to expiry are calendar days from the date to the last trading day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TENORS_DAYS = {"1m": 30, "2m": 61, "3m": 91, "6m": 182, "12m": 365, "13m": 395}


def rule_expiry(contract: str) -> pd.Timestamp:
    """CME last trading day for a Datastream mnemonic NCLMMYY (delivery month MM/YY).

    Uses the termination rule in the repository's roll.py (US federal holidays and
    Good Friday), which is checked against published CME dates in its tests.
    """
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from roll import termination_date

    mm, yy = int(contract[3:5]), int(contract[5:7])
    year = 1900 + yy if yy >= 80 else 2000 + yy
    return termination_date(year, mm)


def load_contracts(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Standardise raw Datastream rows into a tidy contract panel.

    Expiries come from the CME rule, not Datastream's LastTrdDate: several 2007
    contracts carry a last trading date a month early in Datastream (see
    DEVIATIONS.md). Datastream's value is kept as `expiry_ds` for the record.
    """
    d = pd.concat(frames, ignore_index=True)
    d = d.rename(
        columns={
            "Date_": "date",
            "Settlement": "settle",
            "Volume": "volume",
            "OpenInterest": "oi",
            "DSMnem": "contract",
            "LastTrdDate": "expiry",
            "FirstNoticeDate": "fnd",
        }
    )
    for c in ("date", "expiry", "fnd"):
        d[c] = pd.to_datetime(d[c])
    d = d.rename(columns={"expiry": "expiry_ds"})
    rule = {c: rule_expiry(c) for c in d.contract.unique()}
    d["expiry"] = d.contract.map(rule)
    d = d[d.date <= d.expiry].dropna(subset=["settle"])
    d = d.drop_duplicates(["date", "contract"], keep="last")
    cols = ["date", "contract", "expiry", "expiry_ds", "settle", "volume", "oi"]
    return d[cols].sort_values(["date", "expiry"])


def ranked(panel: pd.DataFrame, max_rank: int = 13) -> pd.DataFrame:
    """Add rank (1 = nearest listed expiry) and days to expiry."""
    p = panel.copy()
    p["rank"] = p.groupby("date").expiry.rank(method="first").astype(int)
    p["dte"] = (p.expiry - p.date).dt.days
    return p[p["rank"] <= max_rank]


def by_rank(panel: pd.DataFrame, field: str = "settle", max_rank: int = 13) -> pd.DataFrame:
    """Wide table: dates x rank 1..max_rank of `field`."""
    return ranked(panel, max_rank).pivot(index="date", columns="rank", values=field)


def constant_maturity(panel: pd.DataFrame, tenors: dict[str, int] = TENORS_DAYS) -> pd.DataFrame:
    """Log-linear interpolation of settlements at fixed days to expiry.

    For each date and tenor T, take the two listed contracts whose days to
    expiry bracket T and interpolate ln(price) linearly in days. Below the front
    contract's days to expiry, use the front contract (no extrapolation); beyond
    the last listed contract, NaN. Non-positive prices give NaN.
    """
    p = ranked(panel, 40)
    p = p[p.settle > 0]
    out = {}
    for date, g in p.groupby("date", sort=True):
        x = g.dte.to_numpy(float)
        y = np.log(g.settle.to_numpy(float))
        row = {}
        for name, T in tenors.items():
            if T <= x[0]:
                row[name] = np.exp(y[0])
            elif T > x[-1]:
                row[name] = np.nan
            else:
                row[name] = float(np.exp(np.interp(T, x, y)))
        out[date] = row
    return pd.DataFrame.from_dict(out, orient="index")[list(tenors)]


def gsci_roll_weights(dates: pd.DatetimeIndex, first_bd: int = 5, last_bd: int = 9) -> pd.Series:
    """Share of the position already rolled into the next contract, per date.

    Rolls in equal fifths on business days `first_bd`..`last_bd` of each month
    (S&P GSCI convention), counting the dataset's trading days. 0 before the
    window, 1 after it; resets each month.
    """
    dates = pd.DatetimeIndex(dates)
    bd = pd.Series(1, index=dates).groupby(dates.to_period("M")).cumsum()
    n = last_bd - first_bd + 1
    return ((bd - first_bd + 1).clip(0, n) / n).astype(float)


def gsci_targets(panel: pd.DataFrame, offset: int = 0) -> pd.DataFrame:
    """For each date: the contract held before and after this month's roll.

    In month M the GSCI holds the contract expiring in month M (CL contract for
    delivery M+1, which stops trading around the 20th of M) until the roll, then
    the one expiring in M+1. `offset` shifts both one listed month further out
    (offset=1 holds the next contract, as for the far leg of a spread).
    Returns columns `from`, `to`, `w` (share already in `to`, end of day).
    """
    p = ranked(panel, 40)
    dates = pd.DatetimeIndex(sorted(p.date.unique()))
    expiries = p.drop_duplicates("contract").set_index("contract").expiry.sort_values()
    rows = []
    for d in dates:
        month = d.to_period("M")
        live = expiries[expiries >= d]
        # Contracts expiring in this calendar month or later, nearest first.
        this_or_later = live[live.dt.to_period("M") >= month]
        if len(this_or_later) < offset + 2:
            rows.append((d, None, None))
            continue
        first = this_or_later.index[0]
        if expiries[first].to_period("M") == month:
            frm, to = this_or_later.index[offset], this_or_later.index[offset + 1]
        else:  # this month's contract already expired: already rolled
            frm = to = this_or_later.index[offset]
        rows.append((d, frm, to))
    t = pd.DataFrame(rows, columns=["date", "from", "to"]).set_index("date")
    t["w"] = gsci_roll_weights(t.index)
    t.loc[t["from"] == t["to"], "w"] = 1.0
    return t


def held_excess_return(panel: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    """Daily excess return of holding the target blend, no cross-contract jumps.

    The weight set at the close of day t-1 earns each contract's own return from
    t-1 to t. Returns on a contract with a non-positive price on either day are 0.
    """
    px = panel.pivot(index="date", columns="contract", values="settle").sort_index()
    ret = (px / px.shift(1) - 1).where((px > 0) & (px.shift(1) > 0))
    t = targets.reindex(px.index)
    w_from = (1 - t.w).shift(1)
    w_to = t.w.shift(1)
    c_from, c_to = t["from"].shift(1), t["to"].shift(1)
    out = []
    for d, cf, ct, wf, wt in zip(px.index, c_from, c_to, w_from, w_to):
        if cf is None or (isinstance(cf, float) and np.isnan(cf)):
            out.append(np.nan)
            continue
        rf = ret.at[d, cf] if cf in ret.columns else np.nan
        rt = ret.at[d, ct] if ct in ret.columns else np.nan
        r = (0 if np.isnan(rf) else wf * rf) + (0 if np.isnan(rt) else wt * rt)
        out.append(r)
    return pd.Series(out, index=px.index, name="excess_return")


def decompose(excess: pd.Series, spot: pd.Series) -> pd.DataFrame:
    """Split futures excess return into spot return and roll yield.

    roll yield = excess return - spot return (the standard identity). `spot` is a
    price series (e.g. the 1-month constant-maturity point, or a physical spot).
    """
    s = spot.reindex(excess.index)
    spot_ret = (s / s.shift(1) - 1).where((s > 0) & (s.shift(1) > 0))
    return pd.DataFrame({"excess": excess, "spot": spot_ret, "roll_yield": excess - spot_ret})
