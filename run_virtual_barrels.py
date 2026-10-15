"""Run the Virtual Barrels strategy families on WTI and combine them into a book.

python run_virtual_barrels.py   # tables to stdout, results/virtual_barrels.json, figures

Data in data/vb/ (pulled from EIA, CFTC and FRED; see README):
  F1-F4.csv        EIA Cushing WTI futures contracts 1-4, daily, 2000 to 2024-04-05
  CUSHING.csv      EIA weekly Cushing crude stocks (kbbl), 2004-
  USCRUDE.csv      EIA weekly US commercial crude stocks ex-SPR (kbbl)
  COT_WTI.csv      CFTC disaggregated futures-only positions, NYMEX WTI, 2006-
  OVX.csv          CBOE crude oil ETF volatility index (FRED OVXCLS), 2007-
  CL.csv, BZ.csv   Yahoo continuous front-month WTI and Brent (for the WTI-Brent leg)
  WTISPOT.csv, BRENTSPOT.csv   FRED spot prices (WTI-Brent signal)
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import virtual_barrels as vb
from roll import brent_roll_mask, roll_mask
from strategies import daily_returns, ewma_vol, pnl, summary, turnover, vol_target

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "vb")
IMG = os.path.join(HERE, "docs", "img")
TARGET = 0.15
START, END = "2008-01-01", "2024-04-05"  # all data sources overlap; EIA curve ends here

FTF_GRID = [(m, n) for m in (1, 2, 3, 4) for n in (8, 13, 18, 26)]
FADE_GRID = [(n, x) for n in (150, 175, 200, 225, 250) for x in (5, 10, 15, 20)]
CARRYMOM_GRID = [(1, n) for n in (10, 21, 42, 63)] + [(5, n) for n in (21, 42, 63)]
MOM_GRID = [(1, n) for n in (10, 20, 40, 60, 120, 250)]


def load(name: str) -> pd.Series:
    s = pd.read_csv(
        os.path.join(DATA, f"{name}.csv"), index_col=0, parse_dates=True
    ).iloc[:, 0]
    return s[~s.index.duplicated()].sort_index().dropna()


def load_inputs() -> dict:
    curve = pd.concat(
        {f"C{k}": load(f"F{k}") for k in range(1, 5)}, axis=1, sort=True
    ).dropna()
    # EIA repeats the previous settlement on some exchange holidays (2018-19); drop
    # those rows so they do not enter as zero-return days.
    curve = curve[curve.diff().abs().sum(axis=1) != 0]
    cot = pd.read_csv(
        os.path.join(DATA, "COT_WTI.csv"), index_col=0, parse_dates=True
    ).sort_index()
    return {
        "curve": curve,
        "roll": roll_mask(curve.index),
        "cot": cot,
        "cushing": load("CUSHING"),
        "uscrude": load("USCRUDE"),
        "ovx": load("OVX"),
        "cl": load("CL"),
        "bz": load("BZ"),
        "wti_spot": load("WTISPOT"),
        "brent_spot": load("BRENTSPOT"),
    }


def outright(position: pd.Series, r: pd.Series, roll: pd.Series) -> pd.Series:
    """Vol-target a position in the held contract and return its net P&L."""
    return pnl(vol_target(position, ewma_vol(r), TARGET), r, roll_days=roll)


def build(d: dict) -> tuple[dict, dict]:
    """Every strategy's daily net return, plus the positions behind them."""
    curve, roll = d["curve"], d["roll"]
    idx = curve.index
    r2 = vb.held_returns(curve, 2, roll)
    tri = vb.total_return_index(r2)
    c = vb.carry(curve)
    net = vb.mm_net(d["cot"])
    cot_dates = vb.cot_usable_dates(d["cot"].index)
    weekly = lambda s: vb.on_release(s, idx, cot_dates)

    pos = {
        "buy and hold": pd.Series(1.0, index=idx),
        "carry": vb.carry_signal(c),
        "carry-momentum": vb.carry_momentum_signal(c, 1, 21),
        "momentum M(1,20)": vb.ma_signal(tri, 1, 20),
        "momentum, multi-horizon": vb.multi_horizon_momentum(tri),
        "momentum, reaction-sized": vb.reaction(vb.momentum_strength(tri)).fillna(0.0),
        "follow the flow": weekly(
            vb.ensemble(vb.follow_the_flow, net, FTF_GRID)
        ).fillna(0.0),
        "fade the crowd": weekly(vb.ensemble(vb.fade_the_crowd, net, FADE_GRID)).fillna(
            0.0
        ),
        "hedging pressure": weekly(
            vb.hedging_pressure_signal(vb.hedging_pressure(d["cot"]))
        ).fillna(0.0),
        "inventory cycle": vb.on_release(
            vb.inventory_signal(vb.seasonal_deviation(d["uscrude"])),
            idx,
            vb.EIA_RELEASE_LAG,
        ).fillna(0.0),
    }
    rets = {k: outright(v, r2, roll) for k, v in pos.items()}

    # Calendar spread C2-C3 from the squeeze model.
    sr = vb.spread_returns(curve, 2, 3, roll)
    sq = vb.on_release(
        vb.squeeze_spread_signal(vb.cushing_utilisation(d["cushing"])),
        idx,
        vb.EIA_RELEASE_LAG,
    ).fillna(0.0)
    pos["squeeze spread (C2-C3)"] = sq
    rets["squeeze spread (C2-C3)"] = pnl(
        vol_target(sq, ewma_vol(sr), TARGET), sr, cost_bp=4.0, roll_days=roll
    )

    # WTI-Brent convergence: signal from spot, P&L from the two front-month futures,
    # run on the days both futures trade and then moved onto the curve's calendar.
    both = d["cl"].index.intersection(d["bz"].index)
    cl, bz = d["cl"].reindex(both), d["bz"].reindex(both)
    cl_roll, bz_roll = roll_mask(both), brent_roll_mask(both)
    rwb = daily_returns(cl, cl_roll) - daily_returns(bz, bz_roll)
    conv = vb.convergence_signal(
        d["wti_spot"].reindex(both).ffill(limit=5).shift(1),
        d["brent_spot"].reindex(both).ffill(limit=5).shift(1),
    )
    pos["WTI-Brent convergence"] = conv.reindex(idx).ffill().fillna(0.0)
    wb = pnl(
        vol_target(conv, ewma_vol(rwb), TARGET),
        rwb,
        cost_bp=4.0,
        roll_days=cl_roll | bz_roll,
    )
    rets["WTI-Brent convergence"] = vb.to_calendar(wb, idx)

    # Volatility risk premium: short one-month variance swaps struck at OVX, marked
    # to market daily, realised leg on front-month WTI (what USO, and so OVX,
    # tracks), sized when sold, 1 vol point of bid-ask per sale.
    cl_r = daily_returns(d["cl"], roll_mask(d["cl"].index))
    vrp = vb.short_variance_swaps(d["ovx"], cl_r, target=TARGET)
    rets["short variance (VRP)"] = vb.to_calendar(vrp.scaled, idx)

    return {k: v.loc[START:END] for k, v in rets.items()}, pos


def book(rets: dict, members: list[str]) -> pd.Series:
    """Equal-risk combination of member strategies, rescaled to the target vol.

    The book's leverage is reset monthly and each change is charged 2bp, so
    re-sizing the whole book is not free.
    """
    m = pd.DataFrame({k: rets[k] for k in members}).fillna(0.0).mean(axis=1)
    return vb.scale_to_vol(m, TARGET, rebalance="M", cost_bp=2.0)


def robustness(d: dict) -> dict:
    """Sharpe of each parameter set on its own: median and 10th-90th percentile.

    Bouchouev's point is that one optimised lookback is a fragile thing to trade;
    the spread across parameters says how much of a rule's result is the rule.
    """
    curve, roll = d["curve"], d["roll"]
    idx = curve.index
    r2 = vb.held_returns(curve, 2, roll)
    tri = vb.total_return_index(r2)
    c = vb.carry(curve)
    net = vb.mm_net(d["cot"])
    cot_dates = vb.cot_usable_dates(d["cot"].index)
    families = {
        "momentum": [vb.ma_signal(tri, a, b) for a, b in MOM_GRID],
        "carry-momentum": [vb.carry_momentum_signal(c, a, b) for a, b in CARRYMOM_GRID],
        "follow the flow": [
            vb.on_release(vb.follow_the_flow(net, a, b), idx, cot_dates).fillna(0)
            for a, b in FTF_GRID
        ],
        "fade the crowd": [
            vb.on_release(vb.fade_the_crowd(net, a, b), idx, cot_dates).fillna(0)
            for a, b in FADE_GRID
        ],
    }
    out = {}
    for name, sigs in families.items():
        sh = np.array(
            [summary(outright(s, r2, roll).loc[START:END])["sharpe"] for s in sigs]
        )
        out[name] = {
            "n": len(sh),
            "median": float(np.nanmedian(sh)),
            "p10": float(np.nanpercentile(sh, 10)),
            "p90": float(np.nanpercentile(sh, 90)),
            "share_positive": float(np.mean(sh > 0)),
        }
    return out


PERIODS = [
    ("2008-09 crisis", "2008-01-01", "2009-12-31"),
    ("2010-13 range", "2010-01-01", "2013-12-31"),
    ("2014-16 glut", "2014-01-01", "2016-12-31"),
    ("2017-19", "2017-01-01", "2019-12-31"),
    ("2020 Covid", "2020-01-01", "2020-12-31"),
    ("2021-24", "2021-01-01", END),
]

CURVE = ["carry", "carry-momentum"]
PRICE = ["momentum M(1,20)", "momentum, multi-horizon", "momentum, reaction-sized"]
FLOWS = ["follow the flow", "fade the crowd", "hedging pressure"]
FUND = ["inventory cycle", "squeeze spread (C2-C3)", "WTI-Brent convergence"]
VOL = ["short variance (VRP)"]
CORE = ["carry", "carry-momentum", "inventory cycle", "follow the flow"]


def figures(rets: dict, books: dict, corr: pd.DataFrame, robust: dict) -> None:
    os.makedirs(IMG, exist_ok=True)
    groups = [
        ("Curve", CURVE),
        ("Price momentum", PRICE),
        ("Positioning", FLOWS),
        ("Fundamentals", FUND),
        ("Volatility", VOL),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, (title, members) in zip(axes.flat, groups):
        for k in members:
            ax.plot(
                (1 + rets[k]).cumprod(),
                lw=1,
                label=f"{k} ({summary(rets[k])['sharpe']:.2f})",
            )
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.3)
    ax = axes.flat[-1]
    for k, v in books.items():
        ax.plot(
            (1 + v).cumprod(),
            lw=1.4 if "Virtual" in k else 1,
            label=f"{k} ({summary(v)['sharpe']:.2f})",
        )
    ax.set_yscale("log")
    ax.set_title("Books", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.3)
    fig.suptitle(
        f"WTI strategies, {START[:4]}-{END[:7]}, 15% vol target, net of costs (Sharpe in brackets)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "vb_families.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(
                j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation of daily strategy returns")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "vb_correlation.png"), dpi=130)
    plt.close(fig)


def main() -> None:
    d = load_inputs()
    rets, pos = build(d)
    everything = CURVE + PRICE + FLOWS + FUND + VOL
    books = {
        "buy and hold": rets["buy and hold"],
        "simple: M(1,20) + carry": book(rets, ["momentum M(1,20)", "carry"]),
        "Virtual Barrels book": book(rets, everything),
        "VB book ex-VRP": book(rets, [k for k in everything if k not in VOL]),
        # Chosen after seeing the table: the four families that are positive across
        # parameters and most regimes. In-sample by construction.
        "core four (chosen in-sample)": book(rets, CORE),
    }
    table = {k: summary(v) for k, v in {**rets, **books}.items()}
    df = pd.DataFrame(table).T[["ann_ret", "ann_vol", "sharpe", "max_dd", "hit"]]
    df[["ann_ret", "ann_vol", "max_dd", "hit"]] *= 100
    print(
        f"WTI, {START} to {END}, each strategy vol-targeted to {TARGET:.0%}, net of costs\n"
    )
    print(df.round(2).to_string())

    periods = pd.DataFrame(
        {
            k: {p: summary(v.loc[a:b])["sharpe"] for p, a, b in PERIODS}
            for k, v in {**rets, **books}.items()
        }
    ).T
    print("\nSharpe by period")
    print(periods.round(2).to_string())

    corr = pd.DataFrame({k: rets[k] for k in everything}).corr()
    print(
        f"\nAverage pairwise correlation across strategies: {corr.values[np.triu_indices(len(corr), 1)].mean():.2f}"
    )

    robust = robustness(d)
    print("\nParameter robustness (Sharpe of each parameter set on its own)")
    print(pd.DataFrame(robust).T.round(2).to_string())

    turn = {k: turnover(v.loc[START:END]) for k, v in pos.items()}
    out = {
        "window": [START, END],
        "target_vol": TARGET,
        "table": table,
        "periods": periods.to_dict(orient="index"),
        "correlation": corr.round(4).to_dict(),
        "robustness": robust,
        "turnover": turn,
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "virtual_barrels.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    figures(rets, books, corr, robust)


if __name__ == "__main__":
    main()
