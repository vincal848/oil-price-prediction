"""WTI curve strategy book, 2003 onward: four sleeves and a combined book.

python book.py   -> results/book/book.json, results/book/*.png

Built on the study's engine (whole contracts, next-settlement execution, protocol
costs, rolls charged as spreads). What changes from the study is how the
strategies are implemented, following how desks run them:

- signals are slower and carry a no-trade band (flip only on a decisive move);
- positions rebalance only when the target drifts more than 25%;
- sleeves are combined into one risk-balanced book.

Every sleeve is logged to the trial ledger before its result is computed.
"""

from __future__ import annotations

import json
import os

import curve
import engine
import numpy as np
import pandas as pd
from study import RESULTS, build_inputs, log_trial

START = "2003-01-02"
OUT = os.path.join(RESULTS, "book")
BUFFER = 0.25
IN_BOOK = ["Carry-momentum (outright)", "Curve carry (outright)", "Trend (outright)"]
# Fixes tried for the 2022-25 grind (BOOK.md "When oil falls"); neither went in.
FIXES = ["Curve carry gated by slow trend (outright)", "Slow trend 3/6/12m (outright)"]

CRISES = {
    "2008 crash (Jul-Dec 2008)": ("2008-07-01", "2008-12-31"),
    "2014-16 collapse (Jul 2014-Feb 2016)": ("2014-07-01", "2016-02-29"),
    "COVID / negative WTI (Feb-May 2020)": ("2020-02-01", "2020-05-31"),
    "Ukraine spike (Feb-Jun 2022)": ("2022-02-01", "2022-06-30"),
    "2026 supply shock (Mar 2026)": ("2026-03-01", "2026-04-02"),
}

# Oil price declines since 2003, peak to trough of the rolled 2nd contract.
DROPS = {
    "2008 crash": ("2008-07-03", "2008-12-31"),
    "2014-16 collapse": ("2014-06-20", "2016-02-11"),
    "Q4 2018": ("2018-10-03", "2018-12-24"),
    "2020 COVID": ("2020-01-06", "2020-04-28"),
    "2022-23 slide": ("2022-06-08", "2023-03-17"),
    "Late 2023": ("2023-09-27", "2023-12-12"),
    "2024-25 slide": ("2024-04-05", "2025-05-05"),
}


# ---------------------------------------------------------------------------
# Sleeves
# ---------------------------------------------------------------------------


def sleeves(d: dict) -> dict[str, tuple[str, dict, engine.Book]]:
    cm, px, panel = d["cm"], d["px"], d["panel"]
    w2, w7 = d["w2"], d["w7"]
    out_leg = {"leg": (1.0, w2)}
    spread_leg = {"near": (1.0, w2), "far": (-1.0, w7)}
    c13 = engine.carry(cm, "1m", "13m")
    c7 = engine.carry(cm, "2m", "7m")
    near = curve.held_excess_return(panel, curve.gsci_targets(panel))
    logp = np.log1p(near.fillna(0.0)).cumsum()
    trend = sum(np.sign(logp - logp.shift(n)) for n in (21, 63, 252)) / 3
    slow_trend = sum(np.sign(logp - logp.shift(n)) for n in (63, 126, 252)) / 3
    carry_sig = engine.hysteresis(c13 / c13.rolling(252, min_periods=63).std(), 0.5)
    gated = carry_sig.where(np.sign(carry_sig) == np.sign(slow_trend), 0.0)

    specs = {
        "Carry-momentum (outright)": (
            (
                "Long/short the 2nd contract on the direction of carry: today's "
                "1m-13m carry vs its 21-day average (the study's H2 signal)"
            ),
            {"signal": "cm", "short": 1, "long": 21},
            engine.carry_momentum(c13, 1, 21),
            out_leg,
            False,
        ),
        "Curve carry (outright)": (
            (
                "Long the 2nd contract in backwardation, short in contango (1m-13m "
                "carry, with a 0.5-sigma band around zero so it flips rarely)"
            ),
            {"signal": "carry_band", "band": 0.5},
            carry_sig,
            out_leg,
            False,
        ),
        "Trend (outright)": (
            "Average of 1, 3 and 12-month trend signs on the rolled front contract",
            {"signal": "trend_1_3_12"},
            trend,
            out_leg,
            False,
        ),
        "Carry-momentum (2nd-7th spread)": (
            (
                "Carry-momentum with a 5/63-day signal and 0.5-sigma band, traded "
                "as the 2nd-vs-7th calendar spread on 2m-7m carry"
            ),
            {
                "signal": "cm_band",
                "short": 5,
                "long": 63,
                "band": 0.5,
                "tenors": "2m-7m",
            },
            engine.carry_momentum_band(c7, 5, 63, 0.5),
            spread_leg,
            True,
        ),
        FIXES[0]: (
            "Curve carry, but flat whenever the 3/6/12-month trend disagrees",
            {"signal": "carry_band if sign==sign(trend 3/6/12m) else flat"},
            gated,
            out_leg,
            False,
        ),
        FIXES[1]: (
            "Trend without the 1-month component: 3, 6 and 12-month signs",
            {"signal": "trend_3_6_12"},
            slow_trend,
            out_leg,
            False,
        ),
    }
    books = {}
    for name, (desc, params, sig, legs, is_spread) in specs.items():
        log_trial(
            f"book {name}",
            {**params, "buffer": BUFFER, "start": START, "vol_target": 0.15},
        )
        b = engine.run(sig, legs, px, spread_trade=is_spread, buffer=BUFFER)
        books[name] = (desc, params, b)
    return books


def combine(rets: pd.DataFrame, target_vol: float = 0.15) -> pd.Series:
    """Equal-weight the sleeves, then scale to target vol on trailing 1y vol (lagged)."""
    eq = rets.mean(axis=1)
    vol = eq.rolling(252, min_periods=63).std().shift(1) * np.sqrt(252)
    lev = (target_vol / vol).clip(upper=3.0).fillna(1.0)
    return eq * lev


# ---------------------------------------------------------------------------
# Tearsheet
# ---------------------------------------------------------------------------


def drawdown(r: pd.Series) -> pd.Series:
    eq = (1 + r).cumprod()
    return eq / eq.cummax() - 1


def tearsheet(r: pd.Series, oil: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / 252
    dd = drawdown(r)
    under = dd < 0
    spells = (under != under.shift()).cumsum()[under]
    longest = int(spells.value_counts().max()) if len(spells) else 0
    m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    y = (1 + r).groupby(r.index.year).prod() - 1
    down = r[r < 0]
    return {
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1),
        "ann_vol": float(r.std() * np.sqrt(252)),
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
        "sortino": float(r.mean() * 252 / (down.std() * np.sqrt(252))),
        "max_drawdown": float(dd.min()),
        "calmar": float((eq.iloc[-1] ** (1 / years) - 1) / -dd.min()),
        "longest_drawdown_days": longest,
        "hit_rate_months": float((m > 0).mean()),
        "positive_years": f"{int((y > 0).sum())}/{len(y)}",
        "best_month": float(m.max()),
        "worst_month": float(m.min()),
        "profit_factor": float(r[r > 0].sum() / -r[r < 0].sum()),
        "corr_to_wti": float(r.corr(oil.reindex(r.index))),
        "total_return_multiple": float(eq.iloc[-1]),
        "annual": {int(k): float(v) for k, v in y.items()},
        "crises": {
            k: float((1 + r.loc[a:b]).prod() - 1) for k, (a, b) in CRISES.items()
        },
    }


def trading(book: engine.Book, d: dict) -> dict:
    h = book.holdings.loc[START:]
    u = book.units.loc[START:]
    traded = h.diff().abs().sum(axis=1)
    cost = book.cost.loc[START:]

    def cap(since: str, q: float | None) -> float:
        b = engine.Book(
            book.returns.loc[since:],
            book.gross.loc[since:],
            h.loc[since:],
            u.loc[since:],
            cost.loc[since:],
        )
        return engine.capacity(b, d["vol"], d["oi"], quantile=q)["capacity"]

    caps = {q: cap(START, q) for q in (None, 0.95)}
    sign = np.sign(u)
    return {
        "gross_ann_return": float(book.gross.loc[START:].mean() * 252),
        "cost_ann": float(cost.mean() * 252 / 10e6),
        "contracts_per_year": float(traded.sum() / (len(h) / 252)),
        "position_flips_per_year": float(
            ((sign != sign.shift()) & (sign != 0)).sum() / (len(h) / 252)
        ),
        "time_long": float((sign > 0).mean()),
        "time_short": float((sign < 0).mean()),
        "avg_contracts_held": float(u.abs().mean()),
        "capacity_worst_day": caps[None],
        "capacity_q95": caps[0.95],
        # Electronic trading made every market here deeper after the mid-2000s.
        "capacity_worst_day_since_2010": cap("2010-01-01", None),
        "capacity_q95_since_2010": cap("2010-01-01", 0.95),
    }


def oil_drops(rets: pd.DataFrame, oil: pd.Series) -> dict:
    """Return of each sleeve over each oil decline, and by size of the monthly oil move."""
    eq = (1 + oil.fillna(0.0)).cumprod()
    episodes = {
        name: {
            "oil": float(eq.loc[b] / eq.loc[a] - 1),
            **{k: float((1 + rets[k].loc[a:b]).prod() - 1) for k in rets},
        }
        for name, (a, b) in DROPS.items()
    }
    m = rets.apply(lambda r: (1 + r).groupby(r.index.to_period("M")).prod() - 1)
    mo = (1 + oil).groupby(oil.index.to_period("M")).prod().reindex(m.index) - 1
    buckets = pd.cut(
        mo,
        [-1, -0.10, -0.03, 0.03, 0.10, 5],
        labels=["oil < -10%", "-10% to -3%", "-3% to +3%", "+3% to +10%", "> +10%"],
    )
    by_move = m.groupby(buckets, observed=True).mean()
    by_move["months"] = mo.groupby(buckets, observed=True).size()
    return {"episodes": episodes, "avg_month_by_oil_move": by_move.to_dict("index")}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figures(rets: pd.DataFrame, oil_px: pd.Series, pos: pd.Series) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = ["#1f4e79", "#c55a11", "#548235", "#000000"]

    # 1. Equity curves + drawdown of the book
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1], sharex=True)
    for (k, r), c in zip(rets.items(), colors):
        lw = 2.4 if k == "Combined book" else 1.2
        a.plot((1 + r).cumprod(), label=k, color=c, lw=lw)
    a.set_yscale("log")
    a.set_ylabel("Growth of $1 (log), net of costs")
    a.legend(frameon=False, loc="upper left")
    a.set_title("WTI curve strategy book, 2003-2026, net of costs")
    dd = drawdown(rets["Combined book"])
    b.fill_between(dd.index, dd * 100, 0, color="#c00000", alpha=0.5)
    b.set_ylabel("Book drawdown %")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "equity.png"), dpi=150)
    plt.close(fig)

    # 2. Annual returns heatmap
    ann = rets.apply(lambda r: (1 + r).groupby(r.index.year).prod() - 1) * 100
    fig, ax = plt.subplots(figsize=(11, 3.6))
    v = np.nanmax(np.abs(ann.values))
    im = ax.imshow(ann.T.values, cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
    labels = [str(y) if y < ann.index.max() else f"{y} YTD" for y in ann.index]
    ax.set_xticks(range(len(ann.index)), labels, rotation=90)
    ax.set_yticks(range(len(ann.columns)), ann.columns)
    for i in range(ann.shape[1]):
        for j in range(ann.shape[0]):
            ax.text(j, i, f"{ann.iat[j, i]:.0f}", ha="center", va="center", fontsize=7)
    ax.set_title("Calendar-year returns, % net")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "annual.png"), dpi=150)
    plt.close(fig)

    # 3. Positioning on price (carry-momentum outright)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    p = oil_px.loc[START:]
    ax.plot(p, color="#333333", lw=0.8)
    s = pos.reindex(p.index).fillna(0)
    ax.fill_between(
        p.index,
        0,
        p.max() * 1.05,
        where=s > 0,
        color="#548235",
        alpha=0.18,
        label="long",
        step="post",
    )
    ax.fill_between(
        p.index,
        0,
        p.max() * 1.05,
        where=s < 0,
        color="#c00000",
        alpha=0.18,
        label="short",
        step="post",
    )
    ax.set_ylabel("WTI 2nd contract $/bbl")
    ax.set_title("Curve carry sleeve: long in backwardation, short in contango")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "positioning.png"), dpi=150)
    plt.close(fig)

    # 4. Rolling 2y Sharpe
    fig, ax = plt.subplots(figsize=(11, 3.4))
    rs = rets.rolling(504).mean() / rets.rolling(504).std() * np.sqrt(252)
    for (k, r), c in zip(rs.items(), colors):
        ax.plot(r, label=k, color=c, lw=2.2 if k == "Combined book" else 0.9)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("Rolling 2-year Sharpe")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rolling_sharpe.png"), dpi=150)
    plt.close(fig)


def drops_figure(episodes: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(episodes)
    oil = np.array([episodes[n]["oil"] for n in names]) * 100
    bk = np.array([episodes[n]["Combined book"] for n in names]) * 100
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(x - 0.2, oil, 0.4, color="#999999", label="WTI")
    ax.bar(x + 0.2, bk, 0.4, color=np.where(bk > 0, "#548235", "#c00000"), label="Book")
    for xi, v in zip(x, bk):
        ax.text(
            xi + 0.2, v + (2 if v > 0 else -6), f"{v:+.0f}%", ha="center", fontsize=9
        )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, names)
    ax.set_ylabel("%")
    ax.set_title("Major oil declines since 2003: WTI vs the book")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "oil_drops.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    d = build_inputs()
    books = sleeves(d)
    log_trial(
        "book Combined book",
        {"weights": "equal", "sleeves": IN_BOOK, "vol_target": 0.15, "start": START},
    )
    rets = pd.DataFrame({k: b.returns for k, (_, _, b) in books.items()}).loc[START:]
    rets["Combined book"] = combine(rets[IN_BOOK])
    oil = curve.held_excess_return(d["panel"], curve.gsci_targets(d["panel"]))
    res = {"start": START, "end": str(rets.index.max().date()), "sleeves": {}}
    for k, (desc, params, b) in books.items():
        res["sleeves"][k] = {
            "description": desc,
            "params": params,
            "performance": tearsheet(rets[k], oil),
            "trading": trading(b, d),
        }
    res["combined"] = {
        "performance": tearsheet(rets["Combined book"], oil),
        "sleeves": IN_BOOK,
        "sleeve_correlations": rets.drop(columns="Combined book")
        .corr()
        .round(2)
        .to_dict(),
    }
    log_trial(
        "book Combined book v2",
        {
            "sleeves": ["cm", "gated carry", "slow trend"],
            "weights": "equal",
            "vol_target": 0.15,
            "start": START,
        },
    )
    v2 = combine(rets[[IN_BOOK[0], *FIXES]])
    res["combined_with_fixes"] = {
        "sleeves": [IN_BOOK[0], *FIXES],
        "performance": tearsheet(v2, oil),
    }
    res["oil_drops"] = oil_drops(rets[IN_BOOK + ["Combined book"]], oil.loc[START:])
    drops_figure(res["oil_drops"]["episodes"])
    p10 = tearsheet(combine(rets[IN_BOOK], 0.10), oil)
    res["combined"]["at_10pct_vol"] = {
        k: v for k, v in p10.items() if k not in ("annual", "crises")
    }
    with open(os.path.join(OUT, "book.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    px2 = (d["w2"].reindex(d["px"].index).fillna(0) * d["px"].fillna(0)).sum(axis=1)
    figures(
        rets[IN_BOOK + ["Combined book"]],
        px2.where(px2 != 0),
        books["Curve carry (outright)"][2].units,
    )
    for k in rets:
        p = res["sleeves"].get(k, res.get("combined"))["performance"]
        print(
            f"{k:44s} CAGR {p['cagr']:6.1%} vol {p['ann_vol']:5.1%} "
            f"SR {p['sharpe']:5.2f} MDD {p['max_drawdown']:6.1%} "
            f"yrs+ {p['positive_years']}"
        )


if __name__ == "__main__":
    main()
