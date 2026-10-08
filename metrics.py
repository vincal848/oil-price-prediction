"""Error metrics, and the reasons for the ones that changed.

The original reported MSE, RMSE, NRMSE and MAPE. Two of those do not survive
contact with this particular series, so the set here is different and the
reasoning is below.
"""

import numpy as np

# Below this price, a percentage error stops meaning anything. WTI trades in the
# tens of dollars, so anything under $5 is effectively a division by zero.
MAPE_FLOOR = 5.0


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, or nan when the series makes it meaningless.

    MAPE divides by the actual value. This validation window contains 2020-04-20 at
    -$37.63 and the days either side in the low teens, so the division is by a
    negative number once and by near-zero repeatedly. The original reported a MAPE
    for this window anyway; the number was not interpretable. Returns nan rather
    than a misleading figure when any actual falls below MAPE_FLOOR.
    """
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)

    # Both conditions matter and they are different failures. A near-zero price
    # blows the ratio up; a negative price makes the ratio change sign, so a
    # forecast that is too high reports as a negative percentage error. Requiring
    # strictly positive prices above the floor rules out both.
    if np.any(actual <= 0) or np.any(actual < MAPE_FLOOR):
        return float("nan")
    return float(np.mean(np.abs((actual - predicted) / actual)))


def mase(actual: np.ndarray, predicted: np.ndarray, train_series: np.ndarray) -> float:
    """Mean absolute scaled error (Hyndman & Koehler 2006).

    Replaces MAPE as the scale-free metric. The denominator is the in-sample mean
    absolute error of a one-step naive forecast, so MASE < 1 means the model beat
    naive persistence on the training data's own terms and MASE > 1 means it did
    not. Unlike MAPE it is defined at and below zero, which this series needs.
    """
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    train_series = np.asarray(train_series, dtype=float).reshape(-1)

    scale = np.mean(np.abs(np.diff(train_series)))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / scale)


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray, previous: np.ndarray) -> float:
    """Share of days the forecast got the direction of the move right.

    Added because level accuracy on a near-random-walk is easy and close to
    useless: a forecast that simply repeats today's price scores well on RMSE while
    containing no information about tomorrow. Direction is what a position would
    depend on. `previous` is the last observed price before each target.
    """
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    previous = np.asarray(previous, dtype=float).reshape(-1)

    actual_move = np.sign(actual - previous)
    predicted_move = np.sign(predicted - previous)

    # A forecast that repeats the last price predicts no move at all, so it never
    # makes a directional call. Scoring it against the realised sign would return
    # 0.0, which reads as perfectly anti-predictive rather than as silent. Days
    # where the model calls no move are excluded, and if it essentially never calls
    # one the metric is undefined for that model.
    called = predicted_move != 0
    if called.mean() < 0.05:
        return float("nan")

    # Days with no actual change are dropped rather than counted as a free hit.
    scored = called & (actual_move != 0)
    if not scored.any():
        return float("nan")
    return float(np.mean(actual_move[scored] == predicted_move[scored]))


def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
    train_series: np.ndarray | None = None,
    previous: np.ndarray | None = None,
) -> dict[str, float]:
    """All metrics for one model, as a dict.

    NRMSE is kept for continuity with the original write-up but is flagged here:
    its denominator is the range of the actual values, and in this window that
    range is set by the -$37.63 print. One outlier in the denominator makes every
    model's NRMSE look better than it is, so RMSE and MASE are the ones to read.
    """
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)

    err = actual - predicted
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    value_range = float(np.max(actual) - np.min(actual))

    out = {
        "mse": mse,
        "rmse": rmse,
        "mae": float(np.mean(np.abs(err))),
        "nrmse": rmse / value_range if value_range else float("nan"),
        "mape": mape(actual, predicted),
    }
    if train_series is not None:
        out["mase"] = mase(actual, predicted, train_series)
    if previous is not None:
        out["directional_accuracy"] = directional_accuracy(actual, predicted, previous)
    return out


def skill_score(
    model_metrics: dict[str, float], benchmark_metrics: dict[str, float], key: str = "rmse"
) -> float:
    """Fractional improvement over the benchmark. Positive means better.

    Reported because an absolute RMSE on a price series is unreadable on its own -
    whether 2.5 is good depends entirely on what persistence would have scored.
    """
    b = benchmark_metrics[key]
    if not np.isfinite(b) or b == 0:
        return float("nan")
    return float(1.0 - model_metrics[key] / b)
