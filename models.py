"""The three forecasters, all trained on the same data and scored on the same window.

Each `fit_*` returns predictions in dollars over the evaluation window, so they can
be handed straight to metrics.evaluate and compared.
"""

import os
import time

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from data import make_sequences

# Keras and reservoirpy both seed from numpy, so fixing this makes a run repeatable.
SEED = 42


def _quiet_tensorflow():
    # Import is deferred and logging silenced here rather than at module import,
    # because loading TensorFlow costs several seconds and the naive benchmark and
    # the tests for it should not pay that.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    return tf


def fit_naive(val_scaled: np.ndarray, window: int, scaler: MinMaxScaler) -> tuple[np.ndarray, float]:
    """Persistence: tomorrow's price is today's price.

    This is the benchmark the original did not have, and it is the one that
    decides whether either model is worth anything. A one-step-ahead forecast of a
    near-random-walk looks excellent on a chart and on RMSE while carrying no
    information, because repeating the last observation is already most of the way
    to the answer. Every other model is reported against this.
    """
    X, _ = make_sequences(val_scaled, window)
    # The prediction for each target is simply the last value in its input window.
    predicted = X[:, -1]
    return scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(-1), 0.0


def fit_esn(train_scaled: np.ndarray, val_scaled: np.ndarray, window: int, scaler: MinMaxScaler,
            units: int = 20, leak_rate: float = 0.75, spectral_radius: float = 1.025,
            input_scaling: float = 1.0, rc_connectivity: float = 0.15,
            input_connectivity: float = 0.2, ridge: float = 1e-8) -> tuple[np.ndarray, float]:
    """Echo state network: a fixed random reservoir with a trained ridge readout.

    Hyperparameters follow Kumar K. (2023), as in the original. Two changes:

    `fb_connectivity=1.1` is gone. Connectivity arguments are densities and belong
    in [0, 1], and in any case the model was wired `reservoir >> readout` with no
    feedback path, so the setting never did anything.

    The reservoir is driven over the same windowed sequences the LSTM sees, rather
    than over the raw series, so both models are scored on identical targets.

    Written against ReservoirPy 0.4, which the original predates. 0.4 removed
    `reservoirpy.verbosity`, dropped the `bias_scaling` and `fb_connectivity`
    arguments, and moved the RNG onto the node as `seed=`. The original code raises
    on import with a current install; requirements.txt pins >=0.4 rather than
    holding the repository on a 2023 release.
    """
    import reservoirpy
    from reservoirpy.nodes import Reservoir, Ridge
    reservoirpy.set_seed(SEED)

    start = time.perf_counter()

    reservoir = Reservoir(
        units=units,
        sr=spectral_radius,
        input_scaling=input_scaling,
        lr=leak_rate,
        input_connectivity=input_connectivity,
        rc_connectivity=rc_connectivity,
        seed=SEED,
    )
    esn = reservoir >> Ridge(ridge=ridge)

    X_train, y_train = make_sequences(train_scaled, window)
    X_val, _ = make_sequences(val_scaled, window)

    esn.fit(X_train, y_train.reshape(-1, 1), warmup=0)
    predicted = np.asarray(esn.run(X_val)).reshape(-1, 1)

    elapsed = time.perf_counter() - start
    return scaler.inverse_transform(predicted).reshape(-1), elapsed


def fit_lstm(train_scaled: np.ndarray, val_scaled: np.ndarray, window: int, scaler: MinMaxScaler,
             units: int = 50, dropout: float = 0.2, epochs: int = 20, batch_size: int = 32,
             patience: int = 5, verbose: int = 0) -> tuple[np.ndarray, float]:
    """Two stacked LSTM layers with a dense head.

    Three changes from the original:

    The Dense(25) that sat between the two LSTM layers is gone. It applied per
    timestep rather than to a pooled representation, which was not the intent.

    `epochs=6` with `EarlyStopping(patience=10)` could never stop early, since the
    patience exceeded the total number of epochs. Epochs are raised and patience
    lowered so the callback can actually do its job.

    Validation during training is a tail slice of the *training* window, not the
    test set. The original passed the test set as `validation_data` and then
    reported its metrics, which lets the stopping decision see the data being
    scored on.
    """
    tf = _quiet_tensorflow()
    tf.keras.utils.set_random_seed(SEED)

    start = time.perf_counter()

    X_train, y_train = make_sequences(train_scaled, window)
    X_val, _ = make_sequences(val_scaled, window)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, 1)),
        tf.keras.layers.LSTM(units, return_sequences=True),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(units, return_sequences=False),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True)

    model.fit(X_train, y_train,
              batch_size=batch_size, epochs=epochs,
              # Held out from the end of training, never from the scored window.
              validation_split=0.1, shuffle=False,
              callbacks=[early], verbose=verbose)

    predicted = model.predict(X_val, verbose=0)
    elapsed = time.perf_counter() - start
    return scaler.inverse_transform(predicted).reshape(-1), elapsed
