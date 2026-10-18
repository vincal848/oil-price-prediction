"""Calendar strips and inventory timing for the crack study."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cracks


def test_winter_strip_holds_dec_to_feb_only_in_aug_to_oct():
    idx = pd.bdate_range("2020-07-30", "2020-11-03")
    w, on = cracks.strip_weights(idx, [cracks.WINTER])
    aug = w.loc["2020-08-14"]
    assert aug[aug > 0].round(4).to_dict() == {
        "NCL1220": 0.3333, "NCL0121": 0.3333, "NCL0221": 0.3333
    }
    assert on.loc["2020-07-31"] == 0 and on.loc["2020-08-03"] == 1
    assert on.loc["2020-10-30"] == 1 and on.loc["2020-11-02"] == 0


def test_summer_strip_is_may_to_august_of_the_same_year():
    idx = pd.bdate_range("2021-01-04", "2021-01-08")
    w, _ = cracks.strip_weights(idx, [cracks.SUMMER])
    held = w.iloc[0]
    assert sorted(held[held > 0].index) == ["NCL0521", "NCL0621", "NCL0721", "NCL0821"]


def test_stock_report_is_used_six_days_after_the_week_ends():
    weeks = pd.date_range("2010-01-01", periods=400, freq="W-FRI")
    stocks = pd.Series(100.0 + np.sin(np.arange(400) / 8.3), index=weeks)
    z = cracks.stock_z(stocks)
    assert (z.index - weeks).unique().tolist() == [pd.Timedelta(days=6)]
    assert (z.index.dayofweek == 3).all()  # Thursday
    # The first five years have no full norm.
    assert z.loc[: "2012-12-31"].isna().all()
