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
    extra: np.ndarray | None = None,
) -> Origins:
    """Expanding-window forecasts for origins in [start, end - h], refit each quarter.

    A refit at the first origin f of a quarter uses only rows j with j + h <= f, whose
    targets were realised by then. Origins are every h-th day (non-overlapping targets).
    model is "ridge" or "drift" (mean daily log return so far, times h).
    extra: optional (n, m) exogenous features, already lagged by the caller, appended
    to the k return lags (k may be 0).
    """
    p = np.asarray(prices, dtype=float)
    r = log_returns(p)
    X, y = _features(r, k), _horizon_target(p, h)
    if extra is not None:
        X = np.hstack([X, extra])
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


EXOG_FIT_END = "2018-12-31"
NEW_START = "2024-06-01"
CANDIDATES = {
    "C1": ("spread",),
    "C2": ("basis",),
    "C3": ("usd",),
    "C4": ("spread", "basis"),
    "C5": ("spread", "basis", "usd"),
    "C6": ("spread", "basis", "usd", "ret1"),
}
ALPHA2 = 1000.0
BONFERRONI = 0.05 / len(CANDIDATES)


def exog_features(
    front: pd.Series, brent: pd.Series, wti: pd.Series, usd: pd.Series
) -> pd.DataFrame:
    """PROTOCOL-2 features on the front-month index, each lagged one day, standardised.

    brent/wti/usd are reindexed onto front's dates (forward-filled at most 5 days).
    Standardised with the std over the fit segment; missing values are 0.
    """
    ix = front.index
    brent, wti, usd = (
        s.reindex(ix, method="ffill", limit=5) for s in (brent, wti, usd)
    )

    def logratio(a: pd.Series, b: pd.Series) -> pd.Series:
        return np.log(a.where(a > 0) / b.where(b > 0))

    spread = logratio(brent, wti)
    raw = pd.DataFrame(
        {
            "spread": spread - spread.rolling(252, min_periods=60).mean(),
            "basis": logratio(front, wti),
            "usd": np.log(usd).diff(5),
            "ret1": pd.Series(log_returns(front.to_numpy(float)), index=ix),
        }
    ).shift(1)
    return (raw / raw.loc[:EXOG_FIT_END].std()).fillna(0.0)


def protocol2(front: pd.Series, feats: pd.DataFrame, roll_mask: np.ndarray) -> dict:
    """Run docs/PROTOCOL-2.md once: weight on validation, score the new holdout."""
    p, dates = front.to_numpy(float), front.index
    v0 = int(dates.searchsorted(pd.Timestamp(EXOG_FIT_END), side="right"))
    v1 = int(dates.searchsorted(pd.Timestamp("2024-05-31"), side="right")) - 1
    h0, h1 = v1 + 1, len(dates) - 1
    assert dates[h0] >= pd.Timestamp(NEW_START)
    out: dict = {
        "holdout": [str(dates[h0].date()), str(dates[h1].date())],
        "n_configs": len(CANDIDATES),
    }
    base = walk_forward(p, dates, 1, h0, h1)
    drift = walk_forward(p, dates, 1, h0, h1, model="drift")
    out["persistence"] = evaluate(base, np.zeros_like(base.r_hat))
    out["drift"] = evaluate(base, drift.r_hat)
    out["candidates"] = {}
    for name, cols in CANDIDATES.items():
        extra = feats[list(cols)].to_numpy(float)
        val = walk_forward(p, dates, 1, v0, v1, k=0, alpha=ALPHA2, extra=extra)
        w = fit_weight(val.r_hat, val.y)
        ho = walk_forward(p, dates, 1, h0, h1, k=0, alpha=ALPHA2, extra=extra)
        row = evaluate(ho, w * ho.r_hat)
        row["w"] = w
        row["significant"] = bool(row["skill"] > 0 and row["dm_p"] < BONFERRONI)
        row["sign_pnl"] = sign_pnl(
            Origins(ho.idx, ho.last, ho.actual, ho.y, w * ho.r_hat),
            roll_mask[ho.idx + 1],
        )
        out["candidates"][name] = row
    return out
