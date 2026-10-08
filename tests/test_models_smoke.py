"""fit_esn / fit_lstm run end to end on a tiny series. Skipped where the libs are absent;
CI installs requirements.txt so they do run there."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models
from data import split_and_scale

WINDOW = 5


@pytest.fixture
def tiny():
    prices = 50 + np.cumsum(np.random.default_rng(0).normal(0, 1, 120))
    train, val, scaler, _ = split_and_scale(prices, 0.75)
    return train, val, scaler


def test_fit_esn_returns_dollar_predictions(tiny):
    pytest.importorskip("reservoirpy")
    train, val, scaler = tiny
    pred, _ = models.fit_esn(train, val, WINDOW, scaler, units=10)
    assert pred.shape == (len(val) - WINDOW,) and np.isfinite(pred).all()


def test_fit_lstm_returns_dollar_predictions(tiny):
    pytest.importorskip("tensorflow")
    train, val, scaler = tiny
    pred, _ = models.fit_lstm(train, val, WINDOW, scaler, units=4, epochs=1)
    assert pred.shape == (len(val) - WINDOW,) and np.isfinite(pred).all()
