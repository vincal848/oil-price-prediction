"""Splitting, scaling and windowing the WTI front-month series. Pure: no I/O."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def split_and_scale(
    prices: np.ndarray, train_frac: float = 0.75
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, int]:
    """Chronological split, scaler fitted on the training window only.

    Returns (train_scaled, val_scaled, scaler, cut).

    The original fit MinMaxScaler on the full series and split afterward. That
    leaks: the minimum over 2000-2024 is -$37.63 from 2020-04-20, the day the
    expiring May contract settled negative, and that date is in the validation set.
    Fitting on everything meant the normalization of every training row depended on
    a future event. Fit on train, then apply the same transform to validation.
    """
    prices = np.asarray(prices, dtype=float).reshape(-1, 1)
    cut = int(len(prices) * train_frac)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(prices[:cut])
    val = scaler.transform(prices[cut:])
    return train, val, scaler, cut


def make_sequences(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn a 1-column array into (X, y) for one-step-ahead prediction.

    X[i] is the `window` observations ending at i-1, y[i] is observation i. Kept as
    a function rather than inline because both models need the exact same windowing
    to be comparable, and the original built them differently for each.
    """
    series = np.asarray(series, dtype=float).reshape(-1)
    if window >= len(series):
        raise ValueError(f"window {window} needs more than {len(series)} observations")

    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window : i])
        y.append(series[i])
    return np.array(X), np.array(y)


def aligned_targets(
    val_scaled: np.ndarray, window: int, scaler: MinMaxScaler
) -> np.ndarray:
    """Actual prices for the evaluation window, in dollars.

    Every model is scored on exactly these targets. The original evaluated the
    reservoir from val[1:] and the LSTM from val[120:], so the two were scored over
    different periods and the error numbers were never comparable. Both now start
    at the same place: the first point the longest-memory model can predict.
    """
    _, y = make_sequences(val_scaled, window)
    return scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
