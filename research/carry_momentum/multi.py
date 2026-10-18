"""Multi-market energy curve book: the WTI book's three sleeves on every market.

python multi.py   -> results/multi/multi.json, results/multi/*.png

Same signals, sizing, buffer, execution and cost model as book.py, applied
contract by contract to each market in markets.MARKETS that has data. Each
market gets its own three-sleeve book; the full book weights every
(market, sleeve) pair equally and scales the blend to 15% vol on trailing
one-year vol. Every sleeve is logged to the trial ledger before it runs.
"""

from __future__ import annotations

import json
import os

import book
import curve
import engine
import markets
import numpy as np
import pandas as pd
from study import RESULTS, log_trial

START = book.START
OUT = os.path.join(RESULTS, "multi")
SLEEVES = ["Carry-momentum", "Curve carry", "Trend"]
GRIND = ("2022-06-15", "2025-12-31")


def signals(d: dict) -> dict[str, pd.Series]:
    """The three book signals for one market, defined exactly as in book.py."""
    c13 = engine.carry(d["cm"], "1m", "13m")
    near = curve.held_excess_return(d["panel"], curve.gsci_targets(d["panel"]))
    logp = np.log1p(near.fillna(0.0)).cumsum()
    return {
        "Carry-momentum": engine.carry_momentum(c13, 1, 21),
        "Curve carry": engine.hysteresis(
            c13 / c13.rolling(252, min_periods=63).std(), 0.5
        ),
        "Trend": sum(np.sign(logp - logp.shift(n)) for n in (21, 63, 252)) / 3,
    }


def run_market(d: dict) -> dict[str, engine.Book]:
    m = d["market"]
    out = {}
    for name, sig in signals(d).items():
        log_trial(
            f"multi {m.name} {name}",
            {
                "market": m.name,
                "sleeve": name,
                "buffer": book.BUFFER,
                "start": START,
                "vol_target": 0.15,
            },
        )
        out[name] = engine.run(
            sig,
            {"leg": (1.0, d["w2"])},
            d["px"],
            spread_trade=False,
            buffer=book.BUFFER,
            costs=m.costs,
            multiplier=m.multiplier,
        )
    return out


def available() -> list[markets.Market]:
    return [m for m in markets.MARKETS.values() if markets.raw_files(m)]


def held_return(d: dict) -> pd.Series:
    return curve.held_excess_return(d["panel"], curve.gsci_targets(d["panel"]))


def window_return(r: pd.Series, a: str, b: str) -> float:
    return float((1 + r.loc[a:b]).prod() - 1)


def figures(
    books: pd.DataFrame,
    grid: pd.DataFrame,
    corr: pd.DataFrame,
    grind: pd.DataFrame,
    drops: dict,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    pal = ["#1f4e79", "#c55a11", "#548235", "#7f6000", "#7030a0", "#2e75b6", "#a5a5a5"]

    # 1. Equity: each market's book, WTI-only book, full book
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1], sharex=True)
    for (k, r), c in zip(books.drop(columns=["Core book"]).items(), pal):
        a.plot((1 + r).cumprod(), label=k, color=c, lw=1.0, alpha=0.85)
    a.plot((1 + books["Core book"]).cumprod(), label="Core book", color="black", lw=2.6)
    a.set_yscale("log")
    a.set_ylabel("Growth of $1 (log), net of costs")
    a.set_title("Energy curve book: three sleeves on every market, 2003-2026")
    a.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
    for k, c in (("Core book", "black"), ("WTI", pal[0])):
        dd = book.drawdown(books[k]) * 100
        b.plot(
            dd,
            color=c,
            lw=1.4 if k == "Core book" else 0.9,
            label=k + (" only" if k == "WTI" else ""),
        )
    b.set_ylabel("Drawdown %")
    b.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "equity.png"), dpi=150)
    plt.close(fig)

    # 2. Sharpe grid, market x sleeve
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(grid) + 1.4))
    v = 0.8
    im = ax.imshow(grid.values, cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(grid.shape[1]), grid.columns)
    ax.set_yticks(range(grid.shape[0]), grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.iat[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Sharpe ratio, net of costs, 2003-2026")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sharpe_grid.png"), dpi=150)
    plt.close(fig)

    # 3. Market correlations
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(
                j,
                i,
                f"{corr.iat[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if corr.iat[i, j] > 0.6 else "black",
                fontsize=9,
            )
    ax.set_title("Correlation of each market's book (daily)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "correlation.png"), dpi=150)
    plt.close(fig)

    # 4. Oil declines: WTI-only book vs full book
    names = list(drops)
    x = np.arange(len(names))
    w = np.array([drops[n]["wti_book"] for n in names]) * 100
    f = np.array([drops[n]["core_book"] for n in names]) * 100
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(x - 0.2, w, 0.4, color="#9dc3e6", label="WTI-only book")
    ax.bar(x + 0.2, f, 0.4, color="black", label="Core book (WTI, Brent, gas)")
    for xi, v1, v2 in zip(x, w, f):
        ax.text(
            xi - 0.2, v1 + (2 if v1 > 0 else -6), f"{v1:+.0f}", ha="center", fontsize=8
        )
        ax.text(
            xi + 0.2,
            v2 + (2 if v2 > 0 else -6),
            f"{v2:+.0f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, names)
    ax.set_ylabel("% over the decline")
    ax.set_title("Major WTI declines since 2003: WTI-only book vs full energy book")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "oil_drops.png"), dpi=150)
    plt.close(fig)

    # 5. 2022-25 by market and sleeve
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(grind) + 1.4))
    v = np.nanmax(np.abs(grind.values))
    im = ax.imshow(
        grind.values * 100, cmap="RdYlGn", vmin=-v * 100, vmax=v * 100, aspect="auto"
    )
    ax.set_xticks(range(grind.shape[1]), grind.columns)
    ax.set_yticks(range(grind.shape[0]), grind.index)
    for i in range(grind.shape[0]):
        for j in range(grind.shape[1]):
            ax.text(
                j,
                i,
                f"{grind.iat[i, j] * 100:+.0f}%",
                ha="center",
                va="center",
                fontsize=9,
            )
    ax.set_title("Return over the 2022-25 grind (Jun 2022 - Dec 2025)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "grind.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    mkts = available()
    data = {m.name: markets.inputs(m) for m in mkts}
    runs = {name: run_market(d) for name, d in data.items()}

    sleeve_rets = pd.DataFrame(
        {(mk, s): b.returns for mk, bs in runs.items() for s, b in bs.items()}
    ).loc[START:]
    books = pd.DataFrame({mk: book.combine(sleeve_rets[mk]) for mk in runs})
    tier = {mk: data[mk]["market"].tier for mk in runs}
    core = [mk for mk in runs if tier[mk] == "core"]
    down = [mk for mk in runs if tier[mk] == "downstream"]
    log_trial(
        "multi Core book",
        {
            "markets": core,
            "sleeves": SLEEVES,
            "weights": "equal",
            "vol_target": 0.15,
            "start": START,
        },
    )
    core_rets = sleeve_rets[core]
    books["Core book"] = book.combine(core_rets)
    layers = {}
    if down:
        for name, mks in (("Downstream book", down), ("All markets", list(runs))):
            log_trial(
                f"multi {name}",
                {
                    "markets": mks,
                    "weights": "equal",
                    "vol_target": 0.15,
                    "start": START,
                },
            )
            layers[name] = book.combine(sleeve_rets[mks])

    wti_oil = held_return(data["WTI"]).loc[START:]
    res = {
        "start": START,
        "end": str(books.index.max().date()),
        "markets": {},
        "core_book": {},
    }
    for mk, bs in runs.items():
        m = data[mk]["market"]
        own = held_return(data[mk]).loc[START:]
        res["markets"][mk] = {
            "first_date": str(data[mk]["panel"].date.min().date()),
            "expiry_check": markets.check_expiries(data[mk]["panel"]),
            "multiplier": m.multiplier,
            "tick": m.tick,
            "book": book.tearsheet(books[mk], wti_oil),
            "corr_to_own_market": float(books[mk].corr(own.reindex(books.index))),
            "sleeves": {
                s: {
                    "performance": {
                        k: v
                        for k, v in book.tearsheet(sleeve_rets[(mk, s)], own).items()
                        if k not in ("annual", "crises")
                    },
                    "trading": book.trading(b, data[mk]),
                    "grind": window_return(sleeve_rets[(mk, s)], *GRIND),
                }
                for s, b in bs.items()
            },
        }
    res["core_book"] = {
        "performance": book.tearsheet(books["Core book"], wti_oil),
        "at_10pct_vol": {
            k: v
            for k, v in book.tearsheet(book.combine(core_rets, 0.10), wti_oil).items()
            if k not in ("annual", "crises")
        },
        "markets": core,
        "grind": window_return(books["Core book"], *GRIND),
        "sleeve_type_across_markets": {
            s: {
                k: v
                for k, v in book.tearsheet(
                    book.combine(core_rets.xs(s, axis=1, level=1)), wti_oil
                ).items()
                if k not in ("annual", "crises")
            }
            for s in SLEEVES
        },
    }
    res["layers"] = {
        name: {
            "markets": down if name == "Downstream book" else list(runs),
            "performance": book.tearsheet(r, wti_oil),
            "grind": window_return(r, *GRIND),
            "corr_to_core_book": float(r.corr(books["Core book"])),
        }
        for name, r in layers.items()
    }
    corr = books.drop(columns="Core book").corr().round(2)
    res["market_book_correlations"] = corr.to_dict()

    eq = (1 + wti_oil.fillna(0.0)).cumprod()
    drops = {
        n: {
            "wti": float(eq.loc[b] / eq.loc[a] - 1),
            "wti_book": window_return(books["WTI"], a, b),
            "core_book": window_return(books["Core book"], a, b),
        }
        for n, (a, b) in book.DROPS.items()
    }
    res["oil_drops"] = drops
    m_full = (1 + books["Core book"]).groupby(books.index.to_period("M")).prod() - 1
    m_wti = (1 + books["WTI"]).groupby(books.index.to_period("M")).prod() - 1
    mo = (1 + wti_oil).groupby(wti_oil.index.to_period("M")).prod() - 1
    bucket = pd.cut(
        mo.reindex(m_full.index),
        [-1, -0.10, -0.03, 0.03, 0.10, 5],
        labels=["oil < -10%", "-10% to -3%", "-3% to +3%", "+3% to +10%", "> +10%"],
    )
    res["avg_month_by_wti_move"] = pd.DataFrame(
        {
            "core_book": m_full.groupby(bucket, observed=True).mean(),
            "wti_book": m_wti.groupby(bucket, observed=True).mean(),
            "months": mo.reindex(m_full.index).groupby(bucket, observed=True).size(),
        }
    ).to_dict("index")

    grid = pd.DataFrame(
        {
            s: {
                mk: res["markets"][mk]["sleeves"][s]["performance"]["sharpe"]
                for mk in runs
            }
            for s in SLEEVES
        }
    )
    grid["Market book"] = {mk: res["markets"][mk]["book"]["sharpe"] for mk in runs}
    grind = pd.DataFrame(
        {
            s: {mk: res["markets"][mk]["sleeves"][s]["grind"] for mk in runs}
            for s in SLEEVES
        }
    )
    grind["Market book"] = {mk: window_return(books[mk], *GRIND) for mk in runs}
    grind.loc["Core book"] = [np.nan] * len(SLEEVES) + [res["core_book"]["grind"]]

    with open(os.path.join(OUT, "multi.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    figures(books, grid, corr, grind, drops)

    print(grid.round(2).to_string())
    print(grind.round(2).to_string())
    for k in books:
        p = book.tearsheet(books[k], wti_oil)
        print(
            f"{k:16s} CAGR {p['cagr']:6.1%} SR {p['sharpe']:5.2f} "
            f"MDD {p['max_drawdown']:6.1%} yrs+ {p['positive_years']}"
        )
    for k, v in res["layers"].items():
        p = v["performance"]
        print(
            f"{k:16s} CAGR {p['cagr']:6.1%} SR {p['sharpe']:5.2f} "
            f"MDD {p['max_drawdown']:6.1%} 2022-25 {v['grind']:+.0%} "
            f"corr to core {v['corr_to_core_book']:.2f}"
        )


if __name__ == "__main__":
    main()
