"""Return forecasting and its honest evaluation (see docs/PROTOCOL.md). Pure: no I/O.

Models forecast the h-day log return r; the price forecast is last * exp(r_hat), so
r_hat = 0 is persistence exactly.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

HORIZONS = (1, 5, 20)
LAGS = (1, 5, 20)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
TRAIN_END, VAL_END = "2014-12-31", "2018-12-31"
COST_BP = 2.0  # per side


@dataclass
class Origins:
    """Scored forecast origins for one horizon (non-overlapping, one row per origin)."""

    idx: np.ndarray  # positions of the origin day in the price series
    last: np.ndarray  # price at the origin
    actual: np.ndarray  # price h days later
    y: np.ndarray  # realised h-day log return (nan if either price <= 0)
    r_hat: np.ndarray  # forecast h-day log return


def log_returns(p: np.ndarray) -> np.ndarray:
    """Daily log returns, r[0] and any return touching a non-positive price = nan."""
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = (p[1:] > 0) & (p[:-1] > 0)
    out[1:][ok] = np.log(p[1:][ok] / p[:-1][ok])
    return out


def _features(r: np.ndarray, k: int) -> np.ndarray:
    """Row i = [r_i, r_{i-1}, ..., r_{i-k+1}], missing returns set to 0."""
    X = np.zeros((len(r), k))
    for j in range(k):
        X[j:, j] = np.nan_to_num(r[: len(r) - j])
    return X


def _horizon_target(p: np.ndarray, h: int) -> np.ndarray:
    y = np.full(len(p), np.nan)
    a, b = p[:-h], p[h:]
    ok = (a > 0) & (b > 0)
    y[: len(p) - h][ok] = np.log(b[ok] / a[ok])
    return y


def walk_forward(
    prices: np.ndarray,
    dates: pd.DatetimeIndex,
    h: int,
    start: int,
    end: int,
    k: int = 5,
    alpha: float = 1.0,
    model: str = "ridge",
) -> Origins:
    """Expanding-window forecasts for origins in [start, end - h], refit each quarter.

    A refit at the first origin f of a quarter uses only rows j with j + h <= f, whose
    targets were realised by then. Origins are every h-th day (non-overlapping targets).
    model is "ridge" or "drift" (mean daily log return so far, times h).
    """
    p = np.asarray(prices, dtype=float)
    r = log_returns(p)
    X, y = _features(r, k), _horizon_target(p, h)
    idx = np.arange(start, end - h + 1, h)
    quarter = dates[idx].to_period("Q")
    r_hat = np.zeros(len(idx))
    for q in pd.unique(quarter):
        rows = np.flatnonzero(quarter == q)
        f = idx[rows[0]]
        if model == "drift":
            r_hat[rows] = h * np.nanmean(r[1 : f + 1])
            continue
        j = np.arange(1, f - h + 1)
        j = j[np.isfinite(y[j])]
        fit = Ridge(alpha=alpha, fit_intercept=False).fit(X[j], y[j])
        r_hat[rows] = fit.predict(X[idx[rows]])
    return Origins(idx, p[idx], p[idx + h], y[idx], r_hat)


def newey_west_dm(d: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano statistic and two-sided p for mean(d) = 0, HAC variance.

    d is the loss differential (baseline loss - model loss), so positive = model better.
    Bartlett kernel, bandwidth floor(n^(1/3)).
    """
    n = len(d)
    e = d - d.mean()
    lrv = e @ e / n
    for lag in range(1, int(n ** (1 / 3)) + 1):
        lrv += 2 * (1 - lag / (int(n ** (1 / 3)) + 1)) * (e[lag:] @ e[:-lag]) / n
    if lrv <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(lrv / n)
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def evaluate(o: Origins, r_hat: np.ndarray) -> dict[str, float]:
    """Dollar RMSE/MAE, skill and DM vs persistence, direction hit rate + binomial p."""
    ok = np.isfinite(o.y)
    last, actual, y, rh = o.last[ok], o.actual[ok], o.y[ok], r_hat[ok]
    e = actual - last * np.exp(rh)
    e0 = actual - last
    rmse, rmse0 = np.sqrt(np.mean(e**2)), np.sqrt(np.mean(e0**2))
    dm, dm_p = newey_west_dm(e0**2 - e**2)
    called = (rh != 0) & (y != 0)
    hits = int((np.sign(rh[called]) == np.sign(y[called])).sum())
    n_called = int(called.sum())
    binom_p = stats.binomtest(hits, n_called, 0.5).pvalue if n_called else float("nan")
    return {
        "n": int(ok.sum()),
        "rmse": float(rmse),
        "mae": float(np.mean(np.abs(e))),
        "skill": float(1 - rmse / rmse0),
        "dm": dm,
        "dm_p": dm_p,
        "dir_acc": hits / n_called if n_called else float("nan"),
        "dir_n": n_called,
        "binom_p": float(binom_p),
    }


def fit_weight(r_hat: np.ndarray, y: np.ndarray) -> float:
    """Least-squares shrinkage w in [0, 1] of the model toward persistence."""
    ok = np.isfinite(y)
    denom = r_hat[ok] @ r_hat[ok]
    return float(np.clip(r_hat[ok] @ y[ok] / denom, 0, 1)) if denom > 0 else 0.0


def select(prices: np.ndarray, dates: pd.DatetimeIndex, h: int, val: tuple[int, int]):
    """Best (k, alpha) by validation return MSE, the validation Origins, grid size."""
    best = None
    for k in LAGS:
        for alpha in ALPHAS:
            o = walk_forward(prices, dates, h, val[0], val[1], k, alpha)
            mse = np.nanmean((o.y - o.r_hat) ** 2)
            if best is None or mse < best[0]:
                best = (mse, k, alpha, o)
    return best[1], best[2], best[3], len(LAGS) * len(ALPHAS)


def sign_pnl(o: Origins, roll_day: np.ndarray) -> dict[str, float]:
    """1-day sign strategy, annualised %, gross and net of costs.

    Costs: COST_BP per side on each position change, plus a close-and-reopen (2 sides)
    when the position is held through a roll day (roll_day[i] = target day of origin
    i is a roll). Assumes h = 1 so consecutive origins are consecutive days.
    """
    ok = np.isfinite(o.y)
    pos = np.where(ok, np.sign(o.r_hat), 0.0)
    gross = pos * np.nan_to_num(o.y)
    sides = np.abs(np.diff(pos, prepend=0.0)) + 2 * np.abs(pos) * roll_day
    net = gross - sides * COST_BP * 1e-4
    sharpe = (
        float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else float("nan")
    )
    return {
        "gross_ann_pct": float(gross.mean() * 252 * 100),
        "net_ann_pct": float(net.mean() * 252 * 100),
        "net_sharpe": sharpe,
    }


def protocol(
    prices: np.ndarray, dates: pd.DatetimeIndex, roll_mask: np.ndarray
) -> dict:
    """Run docs/PROTOCOL.md once: select + weight on validation, score the holdout.

    Returns per-horizon holdout rows plus the number of configurations fitted.
    """
    v0 = int(dates.searchsorted(pd.Timestamp(TRAIN_END), side="right"))
    v1 = int(dates.searchsorted(pd.Timestamp(VAL_END), side="right")) - 1
    h0, h1 = v1 + 1, len(dates) - 1
    out: dict = {"horizons": {}, "n_configs": 0}
    for h in HORIZONS:
        k, alpha, val_o, grid = select(prices, dates, h, (v0, v1))
        w = fit_weight(val_o.r_hat, val_o.y)
        ho = walk_forward(prices, dates, h, h0, h1, k, alpha)
        drift = walk_forward(prices, dates, h, h0, h1, model="drift")
        rows = {
            "persistence": evaluate(ho, np.zeros_like(ho.r_hat)),
            "drift": evaluate(ho, drift.r_hat),
            "ridge": evaluate(ho, ho.r_hat),
            "combo": evaluate(ho, w * ho.r_hat),
        }
        out["horizons"][h] = {"k": k, "alpha": alpha, "w": w, "rows": rows}
        out["n_configs"] += grid + 1  # grid on validation, drift baseline
        if h == 1:
            out["horizons"][h]["sign_pnl"] = sign_pnl(ho, roll_mask[ho.idx + 1])
    return out
