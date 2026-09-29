"""Splitting, scaling and windowing - including the leak the original had."""

import os
import sys

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import aligned_targets, make_sequences, split_and_scale


def test_scaler_never_sees_validation_data():
    """The regression for the original leak.

    A spike is placed only in the validation half. If the scaler were fitted on the
    whole series, the training half would be squashed by it. Fitted on train only,
    the training half still spans the full 0-1 range.
    """
    prices = np.concatenate([np.linspace(50, 100, 800), np.linspace(50, 100, 200)])
    prices[900] = -37.63  # the 2020-04-20 print, in the validation half

    train, val, scaler, cut = split_and_scale(prices, train_frac=0.8)

    assert cut == 800
    assert scaler.data_min_[0] == pytest.approx(50.0), "scaler saw the validation spike"
    assert train.min() == pytest.approx(0.0)
    assert train.max() == pytest.approx(1.0)
    # The out-of-range validation point maps below zero, which is the correct
    # behaviour: it is outside anything the model was trained on.
    assert val.min() < 0


def test_leaky_and_clean_scalers_actually_differ():
    """Confirms the test above is testing something."""
    prices = np.concatenate([np.linspace(50, 100, 800), np.linspace(50, 100, 200)])
    prices[900] = -37.63

    leaky = MinMaxScaler().fit(prices.reshape(-1, 1))
    _, _, clean, _ = split_and_scale(prices, train_frac=0.8)
    assert leaky.data_min_[0] != clean.data_min_[0]
    assert leaky.data_min_[0] == pytest.approx(-37.63)


def test_split_is_chronological():
    prices = np.arange(100, dtype=float)
    train, val, scaler, cut = split_and_scale(prices, train_frac=0.75)
    assert cut == 75
    assert len(train) == 75 and len(val) == 25
    # Inverting gets the original values back, in order.
    assert scaler.inverse_transform(train)[:, 0] == pytest.approx(prices[:75])


def test_sequences_are_aligned():
    """X[i] must be the window immediately before y[i], with no overlap into it."""
    series = np.arange(20, dtype=float)
    X, y = make_sequences(series, window=5)

    assert X.shape == (15, 5)
    assert y.shape == (15,)
    assert X[0] == pytest.approx([0, 1, 2, 3, 4])
    assert y[0] == pytest.approx(5)
    assert X[-1] == pytest.approx([14, 15, 16, 17, 18])
    assert y[-1] == pytest.approx(19)
    # The target must never appear in its own input window.
    for i in range(len(y)):
        assert y[i] not in X[i]


def test_window_longer_than_series_raises():
    with pytest.raises(ValueError, match="window"):
        make_sequences(np.arange(10, dtype=float), window=10)


def test_aligned_targets_match_sequence_targets():
    """Both models are scored on exactly these values.

    The original evaluated the reservoir from val[1:] and the LSTM from val[120:],
    so their error numbers described different periods and could not be compared.
    """
    prices = np.linspace(40, 90, 500)
    _, val, scaler, _ = split_and_scale(prices, 0.75)
    window = 30

    targets = aligned_targets(val, window, scaler)
    _, y = make_sequences(val, window)

    assert len(targets) == len(y) == len(val) - window
    assert targets == pytest.approx(scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1))
    # Round trip back to real prices.
    assert targets[0] == pytest.approx(prices[375 + window], rel=1e-9)
