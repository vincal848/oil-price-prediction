"""Metrics behave correctly on this series, and roll dates follow the CME rule."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics
import roll

# --- metrics -------------------------------------------------------------------

def test_perfect_forecast_scores_zero():
    a = np.array([10.0, 20.0, 30.0])
    m = metrics.evaluate(a, a.copy())
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)


def test_rmse_and_mae_are_what_they_claim():
    a = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 5.0])  # errors -1, 0, -2
    m = metrics.evaluate(a, p)
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt(5 / 3))


def test_mape_refuses_to_report_on_near_zero_prices():
    """The original reported MAPE on a window containing -$37.63.

    MAPE divides by the actual value, so on this series it is division by a
    negative number once and by near-zero repeatedly.
    """
    with_negative = np.array([50.0, 20.0, -37.63, 10.0])
    assert np.isnan(metrics.mape(with_negative, with_negative + 1))

    ordinary = np.array([50.0, 60.0, 70.0])
    assert metrics.mape(ordinary, ordinary) == pytest.approx(0.0)
    assert np.isfinite(metrics.mape(ordinary, ordinary + 1))


def test_mase_is_defined_where_mape_is_not():
    actual = np.array([50.0, 20.0, -37.63, 10.0])
    train = np.array([40.0, 42.0, 41.0, 45.0, 44.0])
    value = metrics.mase(actual, actual + 1.0, train)
    assert np.isfinite(value) and value > 0


def test_mase_of_one_means_naive_in_sample_error():
    train = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # naive MAE = 1.0
    actual = np.array([10.0, 11.0, 12.0])
    assert metrics.mase(actual, actual + 1.0, train) == pytest.approx(1.0)


def test_flat_forecast_gives_undefined_direction_not_zero():
    """A persistence forecast makes no directional call.

    Scoring it against the realised sign returns 0.0, which reads as perfectly
    anti-predictive rather than as silent. It must come back nan.
    """
    previous = np.array([10.0, 11.0, 12.0, 13.0])
    actual = np.array([11.0, 10.0, 13.0, 12.0])
    assert np.isnan(metrics.directional_accuracy(actual, previous.copy(), previous))


def test_directional_accuracy_counts_correctly():
    previous = np.array([10.0, 10.0, 10.0, 10.0])
    actual = np.array([11.0, 9.0, 11.0, 9.0])     # up, down, up, down
    predicted = np.array([12.0, 12.0, 12.0, 8.0])  # up, up,   up, down -> 3 of 4
    assert metrics.directional_accuracy(actual, predicted, previous) == pytest.approx(0.75)


def test_skill_score_sign_and_scale():
    better = {"rmse": 5.0}
    benchmark = {"rmse": 10.0}
    assert metrics.skill_score(better, benchmark) == pytest.approx(0.5)
    assert metrics.skill_score(benchmark, benchmark) == pytest.approx(0.0)
    assert metrics.skill_score({"rmse": 20.0}, benchmark) == pytest.approx(-1.0)


# --- roll ----------------------------------------------------------------------

# (delivery year, delivery month, published CME termination date)
PUBLISHED_TERMINATIONS = [
    (2019, 12, "2019-11-20"),   # 25 Nov 2019 was a business day -> 3 back
    (2020, 5, "2020-04-21"),    # 25 Apr 2020 was a Saturday      -> 4 back
    (2020, 6, "2020-05-19"),    # 25 May 2020 was Memorial Day    -> 4 back
    (2021, 1, "2020-12-21"),
    (2022, 3, "2022-02-22"),    # 25 Feb 2022 was a business day  -> 3 back
    (2023, 7, "2023-06-20"),
    (2024, 1, "2023-12-19"),    # 25 Dec 2023 was Christmas       -> 4 back
    (2024, 6, "2024-05-21"),
]


@pytest.mark.parametrize("year,month,expected", PUBLISHED_TERMINATIONS)
def test_termination_dates_match_published_cme_dates(year, month, expected):
    """Against the published CME WTI calendar.

    Two things this pins. The 25th is in the month *preceding* delivery, not the
    delivery month. And whether the 25th is a business day has to account for
    holidays, not just weekends: 25 May 2020 was a Monday and Memorial Day.
    """
    assert roll.termination_date(year, month) == pd.Timestamp(expected)


def test_holiday_aware_business_day_check():
    assert roll.is_business_day("2022-02-25")        # ordinary Friday
    assert not roll.is_business_day("2020-04-25")    # Saturday
    assert not roll.is_business_day("2020-05-25")    # Monday, Memorial Day
    assert not roll.is_business_day("2023-12-25")    # Monday, Christmas


def test_the_negative_settlement_was_not_a_roll_day():
    """2020-04-20 is the day before termination, not the splice.

    Worth pinning: the -$37.63 print is a real collapse in the expiring contract
    under a storage shortage, not an artefact of stitching two contracts together.
    """
    index = pd.bdate_range("2020-04-01", "2020-05-01")
    mask = roll.roll_mask(index)
    assert not mask.loc[pd.Timestamp("2020-04-20")]
    assert not mask.loc[pd.Timestamp("2020-04-21")]
    assert mask.loc[pd.Timestamp("2020-04-22")], "splice lands the day after expiry"


def test_roll_mask_flags_about_one_day_a_month():
    index = pd.bdate_range("2015-01-01", "2019-12-31")
    mask = roll.roll_mask(index)
    months = len(pd.date_range("2015-01-01", "2019-12-31", freq="MS"))
    assert months - 2 <= mask.sum() <= months + 2


def test_roll_diagnostics_shape():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", "2020-12-31")
    prices = pd.Series(60 + np.cumsum(rng.normal(0, 1, len(idx))), index=idx)
    d = roll.roll_diagnostics(prices)
    assert 0 < d["roll_share_of_days"] < 0.15
    assert d["n_roll_days"] < d["n_days"]
    assert np.isfinite(d["volatility_ratio"])


def test_empty_index_is_handled():
    assert len(roll.roll_mask(pd.DatetimeIndex([]))) == 0
