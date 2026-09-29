"""Loading, splitting and windowing the WTI front-month series."""

import os

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

TICKER = "CL=F"
START = "2000-08-30"
END = "2024-05-31"
CACHE = os.path.join(os.path.dirname(__file__), "data", "wti.csv")


def load_prices(start=START, end=END, use_cache=True):
    """WTI front-month closes as a pandas Series indexed by date.

    Cached to CSV because every re-run otherwise re-downloads twenty-four years of
    data, and because a cached copy makes the reported results reproducible even if
    Yahoo revises its history.
    """
    if use_cache and os.path.exists(CACHE):
        s = pd.read_csv(CACHE, index_col=0, parse_dates=True).iloc[:, 0]
        return s.loc[start:end]

    df = yf.download(TICKER, start=start, end=end, progress=False, auto_adjust=False)
    close = df["Close"]
    # yfinance returns MultiIndex columns for a single ticker in recent versions,
    # so ['Close'] can come back as a one-column frame rather than a Series.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.name = "close"

    if use_cache:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        close.to_csv(CACHE)
    return close


def split_and_scale(prices, train_frac=0.75):
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


def make_sequences(series, window):
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
        X.append(series[i - window:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def aligned_targets(val_scaled, window, scaler):
    """Actual prices for the evaluation window, in dollars.

    Every model is scored on exactly these targets. The original evaluated the
    reservoir from val[1:] and the LSTM from val[120:], so the two were scored over
    different periods and the error numbers were never comparable. Both now start
    at the same place: the first point the longest-memory model can predict.
    """
    _, y = make_sequences(val_scaled, window)
    return scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
