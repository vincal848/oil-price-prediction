"""Downstream spread book: refining margins, the Brent-WTI arb and the diesel arb.

python spreads.py   -> results/spreads/spreads.json, results/spreads/*.png

Each spread holds matched delivery months in every leg, in whole contracts at
exchange ratios, and is traded with four strategies:

- curve carry: long the spread when its own curve is backwardated (near margin
  above the deferred margin), with the book's 0.5-sigma band;
- carry-momentum: the direction of that spread carry against its 21-day mean;
- trend: 1, 3 and 12-month direction of the spread's own P&L;
- seasonal mean reversion: fade the margin when it is far from its usual level
  for that calendar month over the previous five years.

Sizing, buffer, execution and costs follow book.py and multi.py. Every
configuration is logged to the trial ledger before it runs.
"""

from __future__ import annotations

import json
import os

import book
import engine
import markets
import multi
import numpy as np
import pandas as pd
from study import RESULTS, log_trial

START = book.START
OUT = os.path.join(RESULTS, "spreads")
GRIND = multi.GRIND
SLEEVES = ["Curve carry", "Carry-momentum", "Trend", "Seasonal reversion"]
BBL = {  # multiply a settlement by this to get $ per barrel
    "wti": 1.0,
    "brent": 1.0,
    "heating_oil": 42.0,
    "gasoline": 42.0,
    "gasoil": 1 / 7.45,  # barrels per tonne of gasoil
}


def gasoline_prefix(d: pd.Timestamp) -> str:
    return "NRB" if d >= pd.Timestamp("2006-01-01") else "NHU"


# name: (reference market for delivery months, [(market, qty, prefix)], crude
# or base-leg lots per unit, used to quote the spread per barrel)
SPREADS = {
    "Heating oil crack": ("wti", [("heating_oil", 1, "NHO"), ("wti", -1, None)], 1),
    "Gasoline crack": ("wti", [("gasoline", 1, gasoline_prefix), ("wti", -1, None)], 1),
    "3-2-1 crack": (
        "wti",
        [
            ("gasoline", 2, gasoline_prefix),
            ("heating_oil", 1, "NHO"),
            ("wti", -3, None),
        ],
        3,
    ),
    "Gasoil crack": ("brent", [("gasoil", 4, "LLE"), ("brent", -3, None)], 3),
    "Brent-WTI": ("wti", [("brent", 1, "LLC"), ("wti", -1, None)], 1),
    "Diesel arb (gasoil vs heating oil)": (
        "heating_oil",
        [("gasoil", 4, "LLE"), ("heating_oil", -3, None)],
        3,
    ),
}


def listed_cost(name: str) -> float:
    """$ per unit when the spread trades as one exchange-listed instrument: one
    $0.01/bbl tick on the crude-equivalent barrels plus exchange fees per lot."""
    _, legs, crude_lots = SPREADS[name]
    lots = sum(abs(q) for _, q, _ in legs)
    return crude_lots * 1000 * 0.01 + lots * engine.FEES


def load() -> dict:
    keys = sorted({m for _, legs, _ in SPREADS.values() for m, _, _ in legs})
    return {k: markets.inputs(markets.MARKETS[k]) for k in keys}


def build_legs(name: str, data: dict) -> list[engine.Leg]:
    ref, legs, _ = SPREADS[name]
    w_ref = data[ref]["w2"]
    out = []
    for mk, qty, prefix in legs:
        m = data[mk]["market"]
        w = (
            w_ref
            if prefix is None and mk == ref
            else engine.map_contracts(w_ref, prefix)
        )
        out.append(engine.Leg(w, data[mk]["px"], qty, m.multiplier, m.costs))
    return out


CONTRACT_BBL = {  # barrels per contract
    "wti": 1000,
    "brent": 1000,
    "heating_oil": 1000,  # 42,000 gallons
    "gasoline": 1000,
    "gasoil": 745,  # 100 tonnes
}


def spread_curve(name: str, data: dict, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """The spread's value per barrel of crude at each constant-maturity tenor:
    the $ value of one spread unit divided by its crude barrels."""
    _, legs, crude_lots = SPREADS[name]
    total = None
    for mk, qty, _ in legs:
        cm = data[mk]["cm"].reindex(idx).ffill(limit=5) * BBL[mk]
        term = cm * qty * CONTRACT_BBL[mk]
        total = term if total is None else total + term
    return total / (crude_lots * 1000)


def seasonal_z(level: pd.Series) -> pd.Series:
    """Distance of the margin from its average in the same calendar month over the
    previous five years, in units of its trailing one-year deviation."""
    m = level.resample("ME").mean()
    norm = m.groupby(m.index.month).transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    daily_norm = norm.reindex(level.index, method="bfill")
    # The norm for month t uses only months t-12, t-24, ... so it is known on day 1.
    dev = level - daily_norm
    return dev / dev.rolling(252, min_periods=126).std()


def signals(name: str, data: dict, legs: list[engine.Leg], idx) -> dict:
    curve_ = spread_curve(name, data, idx)
    carry = curve_["1m"] - curve_["13m"]
    scale = carry.rolling(252, min_periods=63).std()
    pnl = engine.legs_unit_pnl(legs, idx).cumsum()
    z = seasonal_z(curve_["2m"])
    return {
        "Curve carry": engine.hysteresis(carry / scale, 0.5),
        "Carry-momentum": engine.carry_momentum(carry, 1, 21),
        "Trend": sum(np.sign(pnl - pnl.shift(n)) for n in (21, 63, 252)) / 3,
        "Seasonal reversion": (-np.sign(z) * (z.abs() > 0.5)).fillna(0.0),
    }


def capacity(b: engine.Book, data: dict, since: str, q) -> float:
    vol = pd.concat([d["vol"] for d in data.values()], axis=1)
    oi = pd.concat([d["oi"] for d in data.values()], axis=1)
    vol = vol.T.groupby(level=0).sum().T
    oi = oi.T.groupby(level=0).sum().T
    s = engine.Book(
        b.returns.loc[since:],
        b.gross.loc[since:],
        b.holdings.loc[since:],
        b.units.loc[since:],
        b.cost.loc[since:],
    )
    return engine.capacity(s, vol, oi, quantile=q)["capacity"]


def figures(lines: dict, grid: pd.DataFrame, grind: pd.DataFrame, drops: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    style = {
        "WTI only": ("#9dc3e6", 1.3),
        "Core outright book": ("#1f4e79", 1.6),
        "Spread book": ("#c55a11", 1.8),
        "Core + spreads": ("black", 2.6),
    }
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1], sharex=True)
    for k, r in lines.items():
        c, lw = style[k]
        a.plot((1 + r.dropna()).cumprod(), label=k, color=c, lw=lw)
    a.set_yscale("log")
    a.set_ylabel("Growth of $1 (log), net of costs")
    a.set_title(
        "Outright core book and downstream spread book, 2003-2026, net of costs"
    )
    a.legend(frameon=False, loc="upper left", fontsize=9)
    for k in ("WTI only", "Core + spreads"):
        c, lw = style[k]
        b.plot(book.drawdown(lines[k].dropna()) * 100, color=c, lw=lw * 0.6, label=k)
    b.set_ylabel("Drawdown %")
    b.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "equity.png"), dpi=150)
    plt.close(fig)

    for frame, fname, title, fmt, scale in (
        (
            grid,
            "sharpe_grid.png",
            "Sharpe ratio, net of legged costs, 2003-2026",
            "{:.2f}",
            1,
        ),
        (
            grind,
            "grind.png",
            "Return over the 2022-25 grind (Jun 2022 - Dec 2025)",
            "{:+.0f}%",
            100,
        ),
    ):
        fig, ax = plt.subplots(figsize=(10, 0.55 * len(frame) + 1.5))
        vals = frame.values * scale
        v = np.nanmax(np.abs(vals))
        ax.imshow(vals, cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(frame.shape[1]), frame.columns)
        ax.set_yticks(range(frame.shape[0]), frame.index)
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if not np.isnan(vals[i, j]):
                    ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150)
        plt.close(fig)

    names = list(drops)
    x = np.arange(len(names))
    series = [
        ("wti_book", "WTI-only book", "#9dc3e6"),
        ("spread_book", "Spread book", "#c55a11"),
        ("core_plus_spreads", "Core + spreads", "black"),
    ]
    wdt = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for j, (key, label, color) in enumerate(series):
        v = np.array([drops[n][key] for n in names]) * 100
        xs = x + (j - 1) * wdt
        ax.bar(xs, v, wdt, color=color, label=label)
        for xi, vi in zip(xs, v):
            ax.text(
                xi, vi + (2 if vi > 0 else -7), f"{vi:+.0f}", ha="center", fontsize=7
            )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, names)
    ax.set_ylabel("% over the decline")
    ax.set_title("Major WTI declines since 2003")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "oil_drops.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    data = load()
    end = data["wti"]["panel"].date.max()
    idx = data["wti"]["px"].index.union(data["brent"]["px"].index)
    idx = idx[(idx >= "2000-01-01") & (idx <= end)]

    runs, trade, listed = {}, {}, {}
    for name in SPREADS:
        legs = build_legs(name, data)
        for sleeve, sig in signals(name, data, legs, idx).items():
            log_trial(
                f"spread {name} {sleeve}",
                {
                    "spread": name,
                    "sleeve": sleeve,
                    "buffer": book.BUFFER,
                    "start": START,
                    "vol_target": 0.15,
                },
            )
            b = engine.run_legs(sig, legs, idx, buffer=book.BUFFER)
            runs[(name, sleeve)] = b
            log_trial(
                f"spread {name} {sleeve} listed costs",
                {
                    "spread": name,
                    "sleeve": sleeve,
                    "costs": "listed spread",
                    "unit_cost": listed_cost(name),
                    "start": START,
                },
            )
            listed[(name, sleeve)] = engine.run_legs(
                sig, legs, idx, buffer=book.BUFFER, unit_cost=listed_cost(name)
            ).returns
            trade[(name, sleeve)] = {
                "cost_ann": float(b.cost.loc[START:].mean() * 252 / 10e6),
                "gross_ann": float(b.gross.loc[START:].mean() * 252),
                "avg_units": float(b.units.loc[START:].abs().mean()),
                "capacity_q95_since_2010": capacity(b, data, "2010-01-01", 0.95),
                "capacity_worst_day_since_2010": capacity(b, data, "2010-01-01", None),
            }

    rets = pd.DataFrame({k: b.returns for k, b in runs.items()}).loc[START:end]
    started = (rets != 0).cummax()
    rets = rets.where(started)

    # The outright core book from multi.py, rebuilt here (same trial names).
    core_runs = {
        m: multi.run_market(markets.inputs(markets.MARKETS[m]))
        for m in ("wti", "brent", "natural_gas")
    }
    core_own = pd.DataFrame(
        {(m, s): b.returns for m, bs in core_runs.items() for s, b in bs.items()}
    ).loc[START:end]
    core_rets = core_own.fillna(0.0).where(core_own.notna().cummax())
    core = book.combine(core_rets)
    wti_only = book.combine(core_own["wti"].dropna(how="all"))

    log_trial(
        "spread Spread book",
        {
            "spreads": list(SPREADS),
            "sleeves": SLEEVES,
            "weights": "equal",
            "start": START,
        },
    )
    spread_book = book.combine(rets)
    log_trial(
        "spread Core + spreads",
        {
            "halves": ["core outright", "spread book"],
            "weights": "equal",
            "start": START,
        },
    )
    both = book.combine(pd.concat([core, spread_book], axis=1).dropna(how="all"))

    # Sensitivity: the same spreads traded as exchange-listed instruments.
    lr = pd.DataFrame(listed).loc[START:end]
    lr = lr.where((lr != 0).cummax())
    log_trial("spread Spread book listed costs", {"costs": "listed", "start": START})
    spread_listed = book.combine(lr)
    log_trial("spread Core + spreads listed costs", {"costs": "listed", "start": START})
    both_listed = book.combine(
        pd.concat([core, spread_listed], axis=1).dropna(how="all")
    )

    oil = multi.held_return(data["wti"]).loc[START:end]
    lines = {
        "WTI only": wti_only,
        "Core outright book": core,
        "Spread book": spread_book,
        "Core + spreads": both,
    }
    res = {"start": START, "end": str(end.date()), "spreads": {}, "books": {}}
    for name in SPREADS:
        sb = book.combine(rets[name])
        res["spreads"][name] = {
            "legs": [(mk, q) for mk, q, _ in SPREADS[name][1]],
            "book": book.tearsheet(sb, oil),
            "grind": multi.window_return(sb.dropna(), *GRIND),
            "corr_to_wti": float(sb.corr(oil)),
            "sleeves": {
                s: {
                    "performance": {
                        k: v
                        for k, v in book.tearsheet(rets[(name, s)], oil).items()
                        if k not in ("annual", "crises")
                    },
                    "trading": trade[(name, s)],
                    "grind": multi.window_return(rets[(name, s)].dropna(), *GRIND),
                }
                for s in SLEEVES
            },
        }
    extra = {
        "Spread book, listed costs": spread_listed,
        "Core + spreads, listed costs": both_listed,
    }
    for k, r in {**lines, **extra}.items():
        res["books"][k] = {
            "performance": book.tearsheet(r, oil),
            "grind": multi.window_return(r.dropna(), *GRIND),
        }
    res["listed_cost_per_unit"] = {n: listed_cost(n) for n in SPREADS}
    res["listed_sleeves"] = {
        f"{n} | {s}": book.tearsheet(lr[(n, s)], oil)["sharpe"] for n, s in lr.columns
    }
    res["gross_sleeves"] = {
        f"{n} | {s}": float(
            b.gross.loc[START:end].mean() / b.gross.loc[START:end].std() * np.sqrt(252)
        )
        for (n, s), b in runs.items()
    }
    res["corr_spread_book_to_core"] = float(spread_book.corr(core))
    res["sleeve_types"] = {
        s: {
            k: v
            for k, v in book.tearsheet(
                book.combine(rets.xs(s, axis=1, level=1)), oil
            ).items()
            if k not in ("annual", "crises")
        }
        for s in SLEEVES
    }
    spread_books = pd.DataFrame({n: book.combine(rets[n]) for n in SPREADS})
    res["spread_correlations"] = spread_books.corr().round(2).to_dict()
    eq = (1 + oil.fillna(0.0)).cumprod()
    drops = {
        n: {
            "wti": float(eq.loc[b] / eq.loc[a] - 1),
            "wti_book": multi.window_return(wti_only, a, b),
            "spread_book": multi.window_return(spread_book, a, b),
            "core_plus_spreads": multi.window_return(both, a, b),
        }
        for n, (a, b) in book.DROPS.items()
    }
    res["oil_drops"] = drops

    grid = pd.DataFrame(
        {
            s: {
                n: res["spreads"][n]["sleeves"][s]["performance"]["sharpe"]
                for n in SPREADS
            }
            for s in SLEEVES
        }
    )
    grid["Spread book"] = {n: res["spreads"][n]["book"]["sharpe"] for n in SPREADS}
    grind = pd.DataFrame(
        {
            s: {n: res["spreads"][n]["sleeves"][s]["grind"] for n in SPREADS}
            for s in SLEEVES
        }
    )
    grind["Spread book"] = {n: res["spreads"][n]["grind"] for n in SPREADS}
    for k in ("Spread book", "Core outright book", "Core + spreads", "WTI only"):
        grind.loc[k] = [np.nan] * len(SLEEVES) + [res["books"][k]["grind"]]

    with open(os.path.join(OUT, "spreads.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    figures(lines, grid, grind, drops)

    print(grid.round(2).to_string())
    print(grind.round(2).to_string())
    for k, v in res["books"].items():
        p = v["performance"]
        print(
            f"{k:20s} CAGR {p['cagr']:6.1%} SR {p['sharpe']:5.2f} "
            f"MDD {p['max_drawdown']:6.1%} yrs+ {p['positive_years']} "
            f"2022-25 {v['grind']:+.0%}"
        )
    print("corr spread book to core", round(res["corr_spread_book_to_core"], 2))


if __name__ == "__main__":
    main()
