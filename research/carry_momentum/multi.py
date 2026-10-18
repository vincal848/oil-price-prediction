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
# A market joins the tradeable books only if every sleeve could run a $10M book
# on a typical busy day since 2010 (95th percentile day at 5% of volume, 2% of OI).
LIQUIDITY_MIN = 10e6


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
    layers: dict,
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

    # 1. Equity: the WTI-only book against the multi-market books
    lines = {
        "WTI only": books["WTI"],
        "Core book (WTI, Brent, gas)": books["Core book"],
    }
    for k in ("Downstream book", "All markets"):
        if k in layers:
            lines[k] = layers[k]
    style = {
        "WTI only": ("#9dc3e6", 1.3),
        "Core book (WTI, Brent, gas)": ("#1f4e79", 1.8),
        "Downstream book": ("#c55a11", 1.4),
        "All markets": ("black", 2.6),
    }
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1], sharex=True)
    for k, r in lines.items():
        c, lw = style[k]
        a.plot((1 + r.dropna()).cumprod(), label=k, color=c, lw=lw)
    a.set_yscale("log")
    a.set_ylabel("Growth of $1 (log), net of costs")
    a.set_title("Energy curve books, 2003-2026, 15% vol target, net of costs")
    a.legend(frameon=False, loc="upper left", fontsize=9)
    for k in ("WTI only", "All markets"):
        if k in lines:
            c, lw = style[k]
            b.plot(
                book.drawdown(lines[k].dropna()) * 100, color=c, lw=lw * 0.6, label=k
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

    # 4. Oil declines: WTI-only book against the multi-market books
    names = list(drops)
    x = np.arange(len(names))
    series = [
        ("wti_book", "WTI-only book", "#9dc3e6"),
        ("core_book", "Core book (WTI, Brent, gas)", "#1f4e79"),
        ("all_markets", "All markets", "black"),
    ]
    series = [s for s in series if s[0] in drops[names[0]]]
    wdt = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for j, (key, label, color) in enumerate(series):
        v = np.array([drops[n][key] for n in names]) * 100
        xs = x + (j - (len(series) - 1) / 2) * wdt
        ax.bar(xs, v, wdt, color=color, label=label)
        for xi, vi in zip(xs, v):
            ax.text(
                xi, vi + (2 if vi > 0 else -7), f"{vi:+.0f}", ha="center", fontsize=7
            )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, names)
    ax.set_ylabel("% over the decline")
    ax.set_title("Major WTI declines since 2003: WTI-only book vs multi-market books")
    ax.legend(frameon=False, fontsize=8)
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
            if np.isnan(grind.iat[i, j]):
                continue
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

    # Every market stops where WTI does: Datastream's NYMEX rows end before ICE's.
    end = data["WTI"]["panel"].date.max()
    own_cal = pd.DataFrame(
        {(mk, s): b.returns for mk, bs in runs.items() for s, b in bs.items()}
    ).loc[START:end]
    # Across markets, a day one exchange is shut (a US holiday at ICE, say) is a
    # zero-P&L day for that market, not a day it drops out of the average.
    started = own_cal.notna().cummax()
    sleeve_rets = own_cal.fillna(0.0).where(started)
    books = pd.DataFrame(
        {mk: book.combine(own_cal[mk].dropna(how="all")) for mk in runs}
    )
    tier = {mk: data[mk]["market"].tier for mk in runs}
    trading = {
        mk: {s: book.trading(b, data[mk]) for s, b in bs.items()}
        for mk, bs in runs.items()
    }
    capacity = {
        mk: min(t["capacity_q95_since_2010"] for t in trading[mk].values())
        for mk in runs
    }
    liquid = {mk: capacity[mk] >= LIQUIDITY_MIN for mk in runs}
    core = [mk for mk in runs if tier[mk] == "core" and liquid[mk]]
    down = [mk for mk in runs if tier[mk] == "downstream" and liquid[mk]]
    thin = [mk for mk in runs if not liquid[mk]]
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
        members = {"Downstream book": down, "All markets": core + down}
        if thin:
            members["All markets incl. thin swaps"] = list(runs)
        for name, mks in members.items():
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

    wti_oil = held_return(data["WTI"]).loc[START:end]
    res = {
        "start": START,
        "end": str(books.index.max().date()),
        "markets": {},
        "core_book": {},
        "liquidity_min_q95_since_2010": LIQUIDITY_MIN,
        "thin_markets": thin,
    }
    for mk, bs in runs.items():
        m = data[mk]["market"]
        own = held_return(data[mk]).loc[START:end]
        res["markets"][mk] = {
            "first_date": str(data[mk]["panel"].date.min().date()),
            "expiry_check": markets.check_expiries(data[mk]["panel"]),
            "multiplier": m.multiplier,
            "tier": m.tier,
            "capacity_q95_since_2010": capacity[mk],
            "tradeable": liquid[mk],
            "tick": m.tick,
            "book": book.tearsheet(books[mk], wti_oil),
            "corr_to_own_market": float(books[mk].corr(own.reindex(books.index))),
            "sleeves": {
                s: {
                    "performance": {
                        k: v
                        for k, v in book.tearsheet(own_cal[(mk, s)], own).items()
                        if k not in ("annual", "crises")
                    },
                    "trading": trading[mk][s],
                    "grind": window_return(own_cal[(mk, s)].dropna(), *GRIND),
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
            "markets": members[name],
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
            **(
                {"all_markets": window_return(layers["All markets"], a, b)}
                if "All markets" in layers
                else {}
            ),
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
    if "All markets" in res["layers"]:
        grind.loc["All markets"] = [np.nan] * len(SLEEVES) + [
            res["layers"]["All markets"]["grind"]
        ]
    label = {mk: f"{mk} (thin)" for mk in thin}
    grid = grid.rename(index=label)
    grind = grind.rename(index=label)

    with open(os.path.join(OUT, "multi.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    figures(layers, books, grid, corr, grind, drops)

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
