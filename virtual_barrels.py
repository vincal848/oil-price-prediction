"""Strategies built from the frameworks in Bouchouev, *Virtual Barrels* (2023) and
his Oxford Institute for Energy Studies "Energy Quantamentals" series.

The book's argument is that the oil price is set as much by financial ("virtual")
barrels as by physical ones: storage hedgers, trend followers, slow inflation-hedging
money and option sellers all leave footprints in the curve, the positioning data and
the volatility surface. Each family below trades one of those footprints.

1. Curve: carry and carry-momentum (the slope of the curve, and its change).
2. Price momentum with Bouchouev's reaction function (position peaks at 1 sigma).
3. Positioning: follow the managed-money flow, fade crowded extremes, and the
   hedging-pressure premium paid by producers and merchants.
4. Inventories: the storage cycle, and a calendar-spread trade conditioned on
   Cushing filling or draining (the "model of the squeeze").
5. WTI-Brent convergence (the first trade in his quantamentals chapter).
6. The volatility risk premium, as a short variance swap on OVX.

Pure: no I/O. Inputs are pandas objects indexed by date.

Timing: a signal computed with data known at the close of day t sets the position
held over day t+1 (`strategies.pnl` does the shift). Weekly data enters on the first
close after it is published, not the day it describes (see the lags below).
"""

import numpy as np
import pandas as pd

from strategies import TRADING_DAYS, ewma_vol

# CFTC reports are as of Tuesday and released Friday at 3:30pm ET, after the 2:30pm
# oil settlement, so the first close that can use one is Monday's (a 4-day lag lands
# on Saturday and carries forward to the next trading day).
COT_RELEASE_LAG = pd.Timedelta(days=4)
# EIA weekly stocks are for the week ending Friday and released Wednesday 10:30am,
# Thursday in holiday weeks; a 6-day lag uses Thursday's close every week.
EIA_RELEASE_LAG = pd.Timedelta(days=6)


# ---------------------------------------------------------------------------
# The curve: roll-adjusted returns from constant-maturity contract series
# ---------------------------------------------------------------------------


def held_returns(curve: pd.DataFrame, k: int, roll_days: pd.Series) -> pd.Series:
    """Daily return of holding the k-th contract and rolling with it.

    `curve` has columns C1..C4 (EIA's contract 1-4 series). On an ordinary day the
    contract that was k-th yesterday is still k-th. On the first day after a
    front-month expiry every contract moves up one place, so yesterday's k-th is
    today's (k-1)-th. Using Ck/Ck(-1) on that day would compare two different
    contracts; the right return is C(k-1)/Ck(-1). k must be >= 2, since the front
    contract has no successor to compare against on its expiry.
    """
    if k < 2:
        raise ValueError(
            "hold contract 2 or later; contract 1 expires out of the series"
        )
    cur, prev_same = curve[f"C{k}"], curve[f"C{k}"].shift(1)
    rolled = curve[f"C{k - 1}"]
    roll = roll_days.reindex(curve.index, fill_value=False).astype(bool)
    num = cur.where(~roll, rolled)
    r = num / prev_same - 1
    return r.where((num > 0) & (prev_same > 0)).fillna(0.0)


def held_dollar_change(curve: pd.DataFrame, k: int, roll_days: pd.Series) -> pd.Series:
    """Daily $/bbl change of the contract held in slot k (roll-adjusted as above)."""
    roll = roll_days.reindex(curve.index, fill_value=False).astype(bool)
    num = curve[f"C{k}"].where(~roll, curve[f"C{k - 1}"])
    return (num - curve[f"C{k}"].shift(1)).fillna(0.0)


def spread_returns(
    curve: pd.DataFrame, near: int, far: int, roll_days: pd.Series
) -> pd.Series:
    """Return of a long-near / short-far calendar spread, per $ of near notional.

    Positive when the curve backwardates (near rises against far).
    """
    d = held_dollar_change(curve, near, roll_days) - held_dollar_change(
        curve, far, roll_days
    )
    return (d / curve[f"C{near}"].shift(1).abs()).fillna(0.0)


def total_return_index(returns: pd.Series) -> pd.Series:
    """Cumulative index of a roll-adjusted return series (for moving averages)."""
    return (1 + returns).cumprod()


# ---------------------------------------------------------------------------
# 1. Carry and carry-momentum
# ---------------------------------------------------------------------------


def carry(curve: pd.DataFrame, near: int = 2, far: int = 4) -> pd.Series:
    """Annualised slope of the curve: log(near/far) scaled to a year.

    Positive in backwardation. Bouchouev uses month 1 vs month 13, which removes
    seasonality; EIA publishes four contracts, so this uses 2 vs 4 (two months
    apart), skipping contract 1 whose last days before expiry are noisy.
    """
    ok = (curve[f"C{near}"] > 0) & (curve[f"C{far}"] > 0)
    slope = np.log(curve[f"C{near}"].where(ok) / curve[f"C{far}"].where(ok))
    return slope * 12 / (far - near)


def carry_signal(c: pd.Series) -> pd.Series:
    """Long in backwardation, short in contango."""
    return np.sign(c).fillna(0.0)


def carry_momentum_signal(c: pd.Series, short: int = 1, long: int = 21) -> pd.Series:
    """Buy when backwardation is accelerating or contango decelerating.

    sign(MA_short(carry) - MA_long(carry)); Bouchouev's best rule uses today's
    carry against its one-month average.
    """
    fast = c.rolling(short, min_periods=1).mean()
    slow = c.rolling(long, min_periods=long).mean()
    return np.sign(fast - slow).fillna(0.0)


# ---------------------------------------------------------------------------
# 2. Momentum and the reaction function
# ---------------------------------------------------------------------------


def ma_signal(index: pd.Series, short: int, long: int) -> pd.Series:
    """sign(MA_short - MA_long) of a price index; M(1,20) is price vs 20-day mean."""
    fast = index.rolling(short, min_periods=1).mean()
    slow = index.rolling(long, min_periods=long).mean()
    return np.sign(fast - slow).fillna(0.0)


def multi_horizon_momentum(index: pd.Series) -> pd.Series:
    """Equal-weight blend of M(1,5), M(5,20) and M(20,250)."""
    return (
        ma_signal(index, 1, 5) + ma_signal(index, 5, 20) + ma_signal(index, 20, 250)
    ) / 3


def reaction(x: pd.Series) -> pd.Series:
    """Bouchouev's reaction function: R(x) = x * exp((1 - x^2) / 2).

    Rises with the signal, peaks at |x| = 1 (one standard deviation) with
    R = +/-1, then declines: a very stretched signal means most of the move has
    happened, so the position is cut rather than added to.
    """
    return x * np.exp((1 - x**2) / 2)


def momentum_strength(
    index: pd.Series, long: int = 20, vol_window: int = 250
) -> pd.Series:
    """Distance of price above its moving average, in standard deviations."""
    gap = np.log(index / index.rolling(long, min_periods=long).mean())
    return gap / gap.rolling(vol_window, min_periods=60).std()


# ---------------------------------------------------------------------------
# 3. Positioning (CFTC disaggregated COT)
# ---------------------------------------------------------------------------


def on_release(weekly: pd.Series, daily_index: pd.DatetimeIndex, lag) -> pd.Series:
    """Map a weekly series onto trading days from the first close it can be used.

    `lag` is either a fixed Timedelta added to each date, or a Series mapping each
    as-of date to the first date it can be used (see `cot_usable_dates`). Missing
    weekly values stay missing rather than carrying a stale number forward.
    """
    s = weekly.copy()
    if isinstance(lag, pd.Series):
        s.index = pd.DatetimeIndex(lag.reindex(s.index).values)
    else:
        s.index = s.index + lag
    s = s[~s.index.duplicated(keep="last")]
    marker = pd.Series(np.arange(len(s)), index=s.index)
    pos = marker.reindex(daily_index.union(s.index)).ffill().reindex(daily_index)
    vals = s.to_numpy()
    return pd.Series(
        [vals[int(p)] if not np.isnan(p) else np.nan for p in pos], index=daily_index
    )


# Government shutdowns stopped COT publication; the backlog came out over the
# following weeks. Treat every report in the gap as usable only once the catch-up
# finished, which is later than reality and so cannot leak anything.
COT_SHUTDOWNS = [
    (
        pd.Timestamp("2013-09-24"),
        pd.Timestamp("2013-11-19"),
        pd.Timestamp("2013-11-25"),
    ),
    (
        pd.Timestamp("2018-12-18"),
        pd.Timestamp("2019-02-26"),
        pd.Timestamp("2019-03-11"),
    ),
]


def cot_usable_dates(as_of: pd.DatetimeIndex) -> pd.Series:
    """First close at which each COT report can be traded on.

    Normal weeks: as of Tuesday, released Friday 3:30pm ET, after the 2:30pm oil
    settlement, so usable at Monday's close. When a federal holiday falls in the
    release week the report comes out the following Monday afternoon, so usable
    at Tuesday's close. Reports caught in a shutdown use the catch-up date.
    """
    from pandas.tseries.holiday import USFederalHolidayCalendar

    as_of = pd.DatetimeIndex(as_of)
    hols = USFederalHolidayCalendar().holidays(
        as_of.min() - pd.Timedelta(days=7), as_of.max() + pd.Timedelta(days=14)
    )
    out = []
    for d in as_of:
        friday = d + pd.Timedelta(days=(4 - d.weekday()) % 7)
        week = pd.date_range(friday - pd.Timedelta(days=4), friday)
        usable = friday + pd.Timedelta(days=4 if week.isin(hols).any() else 3)
        for lo, hi, caught_up in COT_SHUTDOWNS:
            if lo <= d <= hi:
                usable = max(usable, caught_up)
        out.append(usable)
    return pd.Series(out, index=as_of)


def mm_net(cot: pd.DataFrame) -> pd.Series:
    """Managed-money net position (contracts)."""
    return cot["mm_long"] - cot["mm_short"]


def follow_the_flow(net: pd.Series, short: int, long: int) -> pd.Series:
    """Weekly: sign(MA_short(net) - MA_long(net)). Long when funds are buying."""
    return np.sign(
        net.rolling(short, min_periods=1).mean()
        - net.rolling(long, min_periods=long).mean()
    )


def sentiment_index(net: pd.Series, window: int) -> pd.Series:
    """Where the position sits in its `window`-week range, 0 to 100."""
    lo = net.rolling(window, min_periods=window).min()
    hi = net.rolling(window, min_periods=window).max()
    return 100 * (net - lo) / (hi - lo)


def fade_the_crowd(net: pd.Series, window: int, threshold: float) -> pd.Series:
    """Weekly: short when the crowd is in the top `threshold`% of its range, long in
    the bottom, flat in between."""
    si = sentiment_index(net, window)
    sig = pd.Series(0.0, index=net.index)
    sig[si > 100 - threshold] = -1.0
    sig[si < threshold] = 1.0
    return sig.where(si.notna(), 0.0)


def ensemble(signal_fn, net: pd.Series, grid) -> pd.Series:
    """Average a rule's signal over a parameter grid.

    Bouchouev draws parameters at random each week and reports the median path
    of a million simulations, rather than picking the best set. Averaging over
    the grid is the deterministic version of that: the position is the
    consensus of all the parameterisations.
    """
    return pd.concat([signal_fn(net, *p) for p in grid], axis=1).mean(axis=1)


def hedging_pressure(cot: pd.DataFrame) -> pd.Series:
    """Net short share of producers/merchants and swap dealers (Keynes, de Roon et al.).

    (shorts - longs) / (shorts + longs) for the commercial hedgers. Higher means
    more hedgers want to sell futures, so speculators are paid more to take the
    other side: a larger risk premium for being long.
    """
    longs = cot["pm_long"] + cot["sd_long"]
    shorts = cot["pm_short"] + cot["sd_short"]
    return (shorts - longs) / (shorts + longs)


def hedging_pressure_signal(hp: pd.Series, window: int = 156) -> pd.Series:
    """Weekly: long when hedging pressure is above its trailing 3-year mean."""
    return np.sign(hp - hp.rolling(window, min_periods=52).mean()).fillna(0.0)


# ---------------------------------------------------------------------------
# 4. Inventories and the squeeze
# ---------------------------------------------------------------------------


def seasonal_deviation(stocks: pd.Series, years: int = 5) -> pd.Series:
    """Stocks minus the average of the same ISO week over the previous `years` years.

    The trade press's "vs the five-year average". Uses only earlier years.
    """
    s = stocks.copy()
    # ISO week 53 occurs in only some years; fold it into 52 so it has a history.
    week = np.minimum(s.index.isocalendar().week.values, 52)
    year = s.index.isocalendar().year.values
    df = pd.DataFrame({"v": s.values, "w": week, "y": year}, index=s.index)
    avg = []
    for i, (w, y) in enumerate(zip(week, year)):
        past = df[(df.w == w) & (df.y < y) & (df.y >= y - years)]
        avg.append(past.v.mean() if len(past) >= years - 1 else np.nan)
    return s - pd.Series(avg, index=s.index)


def inventory_signal(dev: pd.Series, weeks: int = 4) -> pd.Series:
    """Weekly: long when the surplus vs the seasonal norm is shrinking (draws),
    short when it is building."""
    return (-np.sign(dev.diff(weeks))).fillna(0.0)


def cushing_utilisation(stocks: pd.Series, window: int = 156) -> pd.Series:
    """Cushing stocks as a share of their trailing 3-year maximum.

    A stand-in for stocks / working capacity: EIA's capacity survey is
    semi-annual and capacity roughly doubled after 2010, so a fixed number
    would misplace the boundaries the squeeze model is about.
    """
    return stocks / stocks.rolling(window, min_periods=52).max()


def squeeze_spread_signal(
    util: pd.Series, weeks: int = 4, lo: float = 0.5, hi: float = 0.8
) -> pd.Series:
    """Weekly position in the near/far calendar spread from the squeeze model.

    Near full storage the front spread is pushed into deep contango; near empty
    it blows out into backwardation; in between it sits at storage cost. So:
    short the spread (bet on contango) when Cushing is filling and above `hi`,
    long it (bet on backwardation) when draining and below `lo`, and otherwise
    follow the direction of the fill.
    """
    filling = util.diff(weeks)
    sig = -np.sign(filling)
    sig[(util > hi) & (filling > 0)] = -2.0
    sig[(util < lo) & (filling < 0)] = 2.0
    return (sig / 2).fillna(0.0)


# ---------------------------------------------------------------------------
# 5. WTI-Brent convergence
# ---------------------------------------------------------------------------


def convergence_signal(
    wti: pd.Series, brent: pd.Series, window: int = 63, entry: float = 1.0
) -> pd.Series:
    """Fade the WTI-Brent spread when it is stretched relative to recent history.

    z = (s - mean) / std of s = log(WTI / Brent) over `window` days. Position in
    the spread (long WTI, short Brent) is -1 when z > entry, +1 when z < -entry,
    otherwise scaled linearly; the trade is on arbitrage between the two
    benchmarks through shipping, which caps how far they can drift apart.
    """
    s = np.log(wti.where(wti > 0) / brent.where(brent > 0))
    z = (s - s.rolling(window, min_periods=window).mean()) / s.rolling(
        window, min_periods=window
    ).std()
    return (-(z / entry).clip(-1, 1)).fillna(0.0)


# ---------------------------------------------------------------------------
# 6. Volatility risk premium
# ---------------------------------------------------------------------------


def short_variance_swaps(
    ovx: pd.Series,
    returns: pd.Series,
    tenor: int = 21,
    target: float = 0.15,
    cost_vol_pts: float = 1.0,
    lookback: int = 252,
    cap: float = 4.0,
) -> pd.DataFrame:
    """Daily marked-to-market P&L of a ladder-free short variance swap programme.

    Every `tenor` trading days, at the close, a one-month variance swap is sold
    at strike K = OVX/100, with vega notional so that 1 vol point = 0.01, i.e.
    variance notional 1/(2K). It is marked each day at

        value = N * [ n/T * RV_n^2 + (T-n)/T * IV_t^2 - K^2 ],  N = 1/(2K)

    where RV_n^2 is the annualised realised variance over the n days elapsed and
    IV_t = OVX_t/100 prices the remaining days. The short's daily P&L is minus the
    change in value; at expiry it has paid out K^2 - RV^2 in full. Marking to
    market matters: booking only the daily accrual hides the swings in OVX and
    understates the risk by about half.

    The swap is sized once, when it is sold, to hit `target` annual vol using the
    realised vol of the programme's P&L over the previous `lookback` days, capped
    at `cap`. `cost_vol_pts` of bid-ask is charged on each sale.

    `returns` should be front-month returns: OVX is the implied vol of USO
    options, and USO holds the front month. Returns columns: pnl (unscaled, per
    unit vega), scaled, leverage.
    """
    idx = returns.index
    iv = (ovx / 100).reindex(idx).ffill()
    r2 = (returns**2 * TRADING_DAYS).to_numpy()
    ivv = iv.to_numpy()
    raw = np.zeros(len(idx))
    lev = np.zeros(len(idx))
    t, current_lev = 0, 0.0
    while t + 1 < len(idx):
        K = ivv[t]
        if not np.isfinite(K) or K <= 0:
            t += 1
            continue
        N = 1 / (2 * K)
        past = raw[max(0, t - lookback) : t]
        if t >= 63 and past.std() > 0:
            current_lev = min(target / (past.std() * np.sqrt(TRADING_DAYS)), cap)
        value_prev = 0.0  # sold at fair value K^2 at the close of day t
        acc = 0.0
        end = min(t + tenor, len(idx) - 1)
        for n, day in enumerate(range(t + 1, end + 1), start=1):
            acc += r2[day]
            remaining = ivv[day] ** 2 if np.isfinite(ivv[day]) else K**2
            value = N * (acc / tenor + (tenor - n) / tenor * remaining - K**2)
            raw[day] = -(value - value_prev)
            lev[day] = current_lev
            value_prev = value
        raw[t + 1] -= cost_vol_pts / 100 * tenor / TRADING_DAYS  # bid-ask on the sale
        t = end
    out = pd.DataFrame({"pnl": raw, "leverage": lev}, index=idx)
    out["scaled"] = out.pnl * out.leverage
    return out


def scale_to_vol(
    pnl: pd.Series,
    target: float = 0.15,
    com: float = 60.0,
    cap: float = 10.0,
    rebalance: str | None = None,
    cost_bp: float = 0.0,
    gross: pd.Series | None = None,
) -> pd.Series:
    """Rescale a P&L stream to `target` annual vol using its own lagged EWMA vol.

    With `rebalance="M"` the leverage is reset only at month ends. `cost_bp` is
    charged on each change in leverage times `gross` (the stream's gross
    exposure, 1 if not given), so re-sizing a book is not free.
    """
    lev = (target / ewma_vol(pnl, com).shift(1)).clip(upper=cap)
    if rebalance == "M":
        month_end = (
            pd.Series(pnl.index, index=pnl.index)
            .groupby(pnl.index.to_period("M"))
            .transform("max")
        )
        lev = lev.where(pnl.index == month_end.values).ffill().shift(1)
    lev = lev.fillna(0.0)
    g = 1.0 if gross is None else gross.reindex(pnl.index).fillna(0.0)
    cost = lev.diff().abs().fillna(lev.abs()) * g * cost_bp / 1e4
    return (pnl * lev - cost).fillna(0.0)


def to_calendar(pnl: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Move a P&L stream onto another trading calendar without losing any of it.

    P&L on a day missing from `index` is added to the next day that is in it.
    """
    cum = pnl.cumsum()
    on = cum.reindex(index.union(cum.index)).ffill().reindex(index).fillna(0.0)
    return on.diff().fillna(on.iloc[0] if len(on) else 0.0)
