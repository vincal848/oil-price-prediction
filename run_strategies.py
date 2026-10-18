"""Trend, carry and volatility targeting on WTI, Brent, RBOB and heating oil.

python run_strategies.py      # tables to stdout, results/strategies.json, figures

Reads the CSVs in data/futures/ (Yahoo continuous front-month closes and FRED spot
prices). Exploratory: the point is to see how each rule behaves and why, across
regimes, not to certify an edge.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from roll import brent_roll_mask, month_start_roll_mask, roll_mask
from strategies import (
    build_positions,
    daily_returns,
    lookback_grid,
    pnl,
    splice_spread,
    summary,
    turnover,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "futures")
IMG = os.path.join(HERE, "docs", "img")
END = "2026-10-14"  # last complete session in the pull

CONTRACTS = {
    "WTI": ("CL", "WTISPOT"),
    "Brent": ("BZ", "BRENTSPOT"),
    "RBOB": ("RB", "GASSPOT"),
    "Heating oil": ("HO", "HOSPOT"),
}

# Splice calendars. WTI, RBOB and heating oil are checked against the data
# (basis jumps 2-4x larger on rule days, see roll_check below). Brent's is weaker
# (1.2-1.5x): FRED's Brent spot is Dated Brent, not the ICE future's underlying.
ROLL_RULES = {
    "CL": roll_mask,
    "BZ": brent_roll_mask,
    "RB": month_start_roll_mask,
    "HO": month_start_roll_mask,
}

PERIODS = [
    ("2001-2007 boom", "2001-01-01", "2007-12-31"),
    ("2008-2009 crisis", "2008-01-01", "2009-12-31"),
    ("2010-2013 range", "2010-01-01", "2013-12-31"),
    ("2014-2016 crash", "2014-01-01", "2016-12-31"),
    ("2017-2019", "2017-01-01", "2019-12-31"),
    ("2020 Covid", "2020-01-01", "2020-12-31"),
    ("2021-2023", "2021-01-01", "2023-12-31"),
    ("2024-2026", "2024-01-01", END),
]


YAHOO = {"CL": "CL=F", "BZ": "BZ=F", "RB": "RB=F", "HO": "HO=F"}
FRED_IDS = {
    "WTISPOT": "DCOILWTICO",
    "BRENTSPOT": "DCOILBRENTEU",
    "GASSPOT": "DGASNYH",
    "HOSPOT": "DHOILNYH",
}


def fetch(name: str) -> None:
    """Download one series into data/futures/ (same sources as run.py)."""
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"{name}.csv")
    if name in YAHOO:
        import yfinance as yf

        df = yf.download(
            YAHOO[name], start="2000-08-01", progress=False, auto_adjust=False
        )
        close = df["Close"]
        close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
        close.dropna().rename("close").to_csv(path, index_label="Date")
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_IDS[name]}"
        df = pd.read_csv(url, na_values=".", index_col=0, parse_dates=True)
        df.loc["2000-01-01":].iloc[:, 0].dropna().rename("spot").to_csv(
            path, index_label="Date"
        )


def load(name: str) -> pd.Series:
    if not os.path.exists(os.path.join(DATA, f"{name}.csv")):
        fetch(name)
    s = pd.read_csv(
        os.path.join(DATA, f"{name}.csv"), index_col=0, parse_dates=True
    ).iloc[:, 0]
    return s[~s.index.duplicated()].sort_index().loc[:END].dropna()


def run_contract(label: str, fut_name: str, spot_name: str) -> dict:
    fut, spot = load(fut_name), load(spot_name)
    roll = ROLL_RULES[fut_name](fut.index)
    r = daily_returns(fut, roll)
    first = r.index[0] + pd.Timedelta(days=400)  # warm-up for the 12-month lookback
    positions = build_positions(fut, r, spot, roll_days=roll)
    rets = {k: pnl(v, r, roll_days=roll).loc[first:] for k, v in positions.items()}
    table = {
        k: {**summary(v), "turnover": turnover(positions[k].loc[first:])}
        for k, v in rets.items()
    }
    periods = {
        k: {p: summary(v.loc[a:b])["sharpe"] for p, a, b in PERIODS}
        for k, v in rets.items()
    }
    grid = lookback_grid(r.loc[first - pd.Timedelta(days=400) :], roll_days=roll)
    corr = pd.DataFrame(
        {
            k: rets[k]
            for k in ["buy and hold", "trend 12m, vol-targeted", "carry, vol-targeted"]
        }
    ).corr()
    return {
        "r": r,
        "rets": rets,
        "table": table,
        "periods": periods,
        "grid": grid,
        "corr": corr,
        "fut": fut,
        "spot": spot,
        "roll": roll,
        "roll_check": roll_check(fut, spot, roll),
    }


def roll_check(fut: pd.Series, spot: pd.Series, roll: pd.Series) -> float:
    """Mean |daily change in futures-minus-spot basis| on splice days / other days.

    Well above 1 means the calendar finds the splices; near 1 means it does not.
    """
    s = spot.reindex(fut.index).ffill(limit=5)
    ok = (fut > 0) & (s > 0)
    db = np.log(fut.where(ok) / s.where(ok)).diff().abs()
    m = roll.reindex(db.index, fill_value=False).astype(bool)
    return float(db[m].mean() / db[~m].mean())


def vol_clustering(r: pd.Series, lags=(1, 5, 21)) -> dict:
    """Autocorrelation of returns vs absolute returns: direction vs size."""
    return {
        "returns": {l: float(r.autocorr(l)) for l in lags},
        "abs_returns": {l: float(r.abs().autocorr(l)) for l in lags},
    }


def fmt_table(table: dict) -> str:
    df = pd.DataFrame(table).T[
        ["ann_ret", "ann_vol", "sharpe", "max_dd", "hit", "turnover"]
    ]
    df[["ann_ret", "ann_vol", "max_dd", "hit"]] *= 100
    return df.round(2).to_string()


def figures(res: dict) -> None:
    os.makedirs(IMG, exist_ok=True)
    wti = res["WTI"]

    # 1. Equity curves, WTI, log scale.
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in [
        "buy and hold",
        "buy and hold, vol-targeted",
        "trend 12m, vol-targeted",
        "carry, vol-targeted",
        "trend 12m + carry, vol-targeted",
    ]:
        ax.plot((1 + wti["rets"][k]).cumprod(), label=k, lw=1.3 if "+" in k else 1)
    ax.set_yscale("log")
    ax.set_title("WTI front month: growth of $1, net of costs")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "strat_equity.png"), dpi=130)
    plt.close(fig)

    # 2. Lookback x rebalance heatmaps, all four contracts.
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for ax, (label, d) in zip(axes, res.items()):
        g = d["grid"]
        im = ax.imshow(g.values, cmap="RdBu", vmin=-0.6, vmax=0.6, aspect="auto")
        ax.set_xticks(range(g.shape[1]), [str(c) for c in g.columns], fontsize=8)
        ax.set_yticks(range(g.shape[0]), ["daily", "weekly", "monthly"], fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("lookback (days)", fontsize=8)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ax.text(
                    j, i, f"{g.values[i, j]:.2f}", ha="center", va="center", fontsize=7
                )
    fig.colorbar(im, ax=axes, shrink=0.8, label="net Sharpe")
    fig.suptitle(
        "Vol-targeted trend: net Sharpe by lookback and rebalance frequency",
        fontsize=11,
        y=1.06,
    )
    fig.savefig(os.path.join(IMG, "strat_lookbacks.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 3. Basis vs carry P&L: when the term structure is informative.
    spread = splice_spread(wti["fut"], wti["spot"], wti["roll"]).clip(-0.15, 0.15)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    a1.fill_between(
        spread.index,
        spread.clip(lower=0) * 100,
        color="tab:red",
        alpha=0.5,
        lw=0,
        step="post",
        label="contango (short)",
    )
    a1.fill_between(
        spread.index,
        spread.clip(upper=0) * 100,
        color="tab:blue",
        alpha=0.5,
        lw=0,
        step="post",
        label="backwardation (long)",
    )
    a1.set_ylabel("next vs front, %")
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    a1.set_title(
        "WTI front-to-next spread, read at each contract roll, and the carry rule"
    )
    a2.plot(
        (1 + wti["rets"]["carry, vol-targeted"]).cumprod(),
        color="k",
        lw=1,
        label="carry, next vs front",
    )
    a2.plot(
        (1 + wti["rets"]["carry, vs spot, vol-targeted"]).cumprod(),
        color="grey",
        lw=1,
        ls="--",
        label="carry, front vs spot",
    )
    a2.set_yscale("log")
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    a2.set_ylabel("growth of $1")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "strat_carry.png"), dpi=130)
    plt.close(fig)


def main() -> None:
    res = {label: run_contract(label, f, s) for label, (f, s) in CONTRACTS.items()}
    out = {}
    for label, d in res.items():
        print(
            f"\n=== {label}: {d['r'].index[0].date()} to {d['r'].index[-1].date()} "
            f"(splice check {d['roll_check']:.1f}x) ==="
        )
        print(fmt_table(d["table"]))
        print("\nSharpe by period")
        print(pd.DataFrame(d["periods"]).T.round(2).to_string())
        print(
            "\nVol-targeted trend, net Sharpe by lookback (cols) and rebalance (rows)"
        )
        print(d["grid"].round(2).to_string())
        print("\nDaily return correlations")
        print(d["corr"].round(2).to_string())
        out[label] = {
            "table": d["table"],
            "periods": d["periods"],
            "lookback_grid": {
                f: {str(k): v for k, v in row.items()}
                for f, row in d["grid"].iterrows()
            },
            "correlations": d["corr"].round(4).to_dict(),
            "vol_clustering": vol_clustering(d["r"]),
            "splice_check": d["roll_check"],
        }
        vc = out[label]["vol_clustering"]
        print(
            f"\nAutocorrelation, returns {vc['returns']}  |returns| {vc['abs_returns']}"
        )
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "strategies.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    figures(res)


if __name__ == "__main__":
    main()
