"""Contract roll dates for NYMEX WTI, and the tools to measure what they cost.

`CL=F` is Yahoo's continuous front-month series: when the front contract expires,
the series splices onto the next one. The day-over-day change across a splice mixes
a real price move with the spread between two different contracts, and no model can
forecast the second part. Anything scored on this series is therefore scored partly
on a quantity that is not a price move at all.

Roll dates are derived from the CME contract rule rather than detected from the data,
so they are known in advance and cannot be fitted to whatever happens to look like a
jump.
"""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# NYMEX observes the US federal holiday calendar for this purpose. Close enough:
# the rule only needs to count three business days back.
BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def is_business_day(ts: pd.Timestamp | str) -> bool:
    """Business day on the NYMEX calendar, holidays included.

    A weekday check is not enough. The 25th of May 2020 was a Monday but also
    Memorial Day, and treating it as a business day moves that contract's
    termination a day late.
    """
    ts = pd.Timestamp(ts).normalize()
    return len(pd.bdate_range(ts, ts, freq=BDAY)) == 1


def termination_date(year: int, month: int) -> pd.Timestamp:
    """Last trading day of the WTI contract delivering in (year, month).

    CME rule: trading terminates 3 business days before the 25th calendar day of
    the month preceding delivery; if that 25th is not a business day, 4 business
    days before it instead.

    Verified against eight published CME termination dates in
    tests/test_metrics_and_roll.py.
    """
    # The 25th that matters is in the month BEFORE delivery. Reading the rule off
    # the delivery month puts every roll a month late, which quietly turns the
    # whole analysis into a comparison of arbitrary days.
    prior_year, prior_month = (year - 1, 12) if month == 1 else (year, month - 1)
    twenty_fifth = pd.Timestamp(year=prior_year, month=prior_month, day=25)

    steps = 3 if is_business_day(twenty_fifth) else 4
    return (twenty_fifth - steps * BDAY).normalize()


def termination_dates(start: pd.Timestamp | str, end: pd.Timestamp | str) -> pd.DatetimeIndex:
    """Every WTI termination date in range, as a DatetimeIndex."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    # Delivery months run one month ahead of the termination, so widen the span.
    for year in range(start.year - 1, end.year + 2):
        for month in range(1, 13):
            t = termination_date(year, month)
            if start <= t <= end:
                out.append(t)
    return pd.DatetimeIndex(sorted(set(out)))


def roll_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean Series: True on the first trading day after each termination.

    That is the day whose change spans two different contracts, so it is the day the
    splice actually lands on.
    """
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return pd.Series([], dtype=bool, index=index)

    flagged = set()
    for t in termination_dates(index.min(), index.max()):
        after = index[index > t]
        if len(after):
            flagged.add(after[0])
    return pd.Series([i in flagged for i in index], index=index)


def roll_diagnostics(prices: pd.Series) -> dict[str, float]:
    """How much the roll actually distorts this series.

    Returns a dict. Written to answer the question with numbers rather than to
    assume the answer: the roll is a real defect in the data, but whether it is
    large enough to change a conclusion is an empirical matter.
    """
    prices = pd.Series(prices).dropna()
    changes = prices.diff().dropna()
    mask = roll_mask(changes.index)

    on, off = changes[mask], changes[~mask]
    total_sq = float((changes ** 2).sum())

    return {
        "n_days": len(changes),
        "n_roll_days": int(mask.sum()),
        "roll_share_of_days": float(mask.mean()),
        "mean_abs_change_roll": float(on.abs().mean()),
        "mean_abs_change_other": float(off.abs().mean()),
        "volatility_ratio": float(on.abs().mean() / off.abs().mean()),
        "roll_share_of_variance": float((on ** 2).sum() / total_sq) if total_sq else np.nan,
    }
