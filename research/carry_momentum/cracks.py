"""Crack spreads traded the way a NYMEX products desk would (CRACKS_PROTOCOL.md).

python cracks.py   -> results/cracks/cracks.json, results/cracks/*.png

Three NYMEX cracks (heating oil, RBOB, 3-2-1), 2006-2026, each with:
- seasonal strips on a fixed calendar (winter heating oil, summer gasoline);
- the prompt crack positioned on US product stocks against their seasonal norm;
- the two combined.
Annual risk budget, listed-spread costs, next-settlement execution.
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
from study import DATA, RESULTS, log_trial

START, END = "2006-01-03", "2026-04-02"
OUT = os.path.join(RESULTS, "cracks")
RISK = 0.10  # annual volatility budget per strategy, on $10M
CAPITAL = 10e6
GRIND = multi.GRIND

# name: [(market, qty, prefix)], crude lots per unit, stock weights
CRACKS = {
    "Heating oil crack": (
        [("heating_oil", 1, "NHO"), ("wti", -1, "NCL")],
        1,
        {"distillate": 1.0},
    ),
    "RBOB crack": (
        [("gasoline", 1, "NRB"), ("wti", -1, "NCL")],
        1,
        {"gasoline": 1.0},
    ),
    "3-2-1 crack": (
        [("gasoline", 2, "NRB"), ("heating_oil", 1, "NHO"), ("wti", -3, "NCL")],
        3,
        {"gasoline": 2 / 3, "distillate": 1 / 3},
    ),
}
# (window months, delivery months as (year offset, month)) per seasonal trade
WINTER = ((8, 9, 10), [(0, 12), (1, 1), (1, 2)])
SUMMER = ((1, 2, 3), [(0, 5), (0, 6), (0, 7), (0, 8)])
SEASONS = {
    "Heating oil crack": [WINTER],
    "RBOB crack": [SUMMER],
    "3-2-1 crack": [WINTER, SUMMER],
}
STRATEGIES = ["Seasonal strips", "Inventories", "Combined"]
# Diagnostics, not strategies: the same strips three months earlier, and the
# prompt crack held long all year. Added after the first run (logged as trials).
EARLY_SUMMER = ((10, 11, 12), [(1, 5), (1, 6), (1, 7), (1, 8)])
EARLY_WINTER = ((5, 6, 7), WINTER[1])
PLACEBO_SEASONS = {
    "Heating oil crack": [EARLY_WINTER],
    "RBOB crack": [EARLY_SUMMER],
    "3-2-1 crack": [EARLY_WINTER, EARLY_SUMMER],
}


def placebos(name, data, idx, size, oil) -> dict:
    out = {}
    w, on = strip_weights(idx, PLACEBO_SEASONS[name])
    log_trial(f"crack {name} placebo early window", {"diagnostic": True})
    r = engine.run_legs(
        on, legs_for(name, data, w), idx, size=size, unit_cost=listed_cost(name)
    ).returns.loc[START:END]
    t = season_trades(r, on.loc[START:END])
    out["strips three months earlier"] = {
        "sharpe": book.tearsheet(r, oil)["sharpe"],
        "hit_rate": float(np.mean([v > 0 for v in t.values()])),
        "return_per_year_in_market": float(r.mean() * 252 / on.loc[START:END].mean()),
    }
    log_trial(f"crack {name} placebo always long prompt", {"diagnostic": True})
    prompt = legs_for(name, data, data["wti"]["w2"])
    r = engine.run_legs(
        pd.Series(1.0, index=idx), prompt, idx, size=size, unit_cost=listed_cost(name)
    ).returns.loc[START:END]
    out["long the prompt crack all year"] = {
        "sharpe": book.tearsheet(r, oil)["sharpe"],
        "return_per_year_in_market": float(r.mean() * 252),
    }
    return out


def listed_cost(name: str) -> float:
    legs, lots, _ = CRACKS[name]
    return lots * 1000 * 0.01 + sum(abs(q) for _, q, _ in legs) * engine.FEES


def strip_weights(idx: pd.DatetimeIndex, seasons) -> tuple[pd.DataFrame, pd.Series]:
    """WTI-contract weights for the seasonal strips, and the in-trade indicator."""
    rows, on = {}, pd.Series(0.0, index=idx)
    for d in idx:
        for months, deliveries in seasons:
            if d.month in months:
                w = 1 / len(deliveries)
                rows[d] = {
                    f"NCL{m:02d}{(d.year + dy) % 100:02d}": w for dy, m in deliveries
                }
                on[d] = 1.0
    w = pd.DataFrame.from_dict(rows, orient="index").reindex(idx).fillna(0.0)
    return w, on


def legs_for(name: str, data: dict, w_ref: pd.DataFrame) -> list[engine.Leg]:
    legs, _, _ = CRACKS[name]
    out = []
    for mk, qty, prefix in legs:
        m = data[mk]["market"]
        w = w_ref if prefix == "NCL" else engine.map_contracts(w_ref, prefix)
        out.append(engine.Leg(w, data[mk]["px"], qty, m.multiplier, m.costs))
    return out


def load_stocks(name: str) -> pd.Series:
    s = pd.read_csv(
        os.path.join(DATA, f"US{name.upper()}.csv"), index_col=0, parse_dates=True
    ).iloc[:, 0]
    return s.sort_index()


def stock_z(stocks: pd.Series) -> pd.Series:
    """z of stocks against the same week's average over the previous five years,
    indexed by the date it is first used (six days after the week ends)."""
    df = pd.DataFrame({"v": stocks})
    df["week"] = np.minimum(df.index.isocalendar().week.astype(int), 52)
    df["year"] = df.index.year
    norm = []
    for d, r in df.iterrows():
        prior = df[(df.week == r.week) & (df.year < r.year) & (df.year >= r.year - 5)]
        norm.append(prior.v.mean() if len(prior) >= 3 else np.nan)
    gap = df.v - pd.Series(norm, index=df.index)
    z = gap / gap.rolling(52, min_periods=26).std()
    z.index = z.index + pd.Timedelta(days=6)
    return z


def inventory_signal(name: str, z: dict, idx) -> pd.Series:
    _, _, weights = CRACKS[name]
    comb = sum(w * z[k] for k, w in weights.items())
    daily = comb.reindex(idx.union(comb.index)).ffill().reindex(idx)
    return pd.Series(
        np.where(daily < -0.5, 1.0, np.where(daily > 0.5, -1.0, 0.0)), index=idx
    )


def annual_size(unit_pnl: pd.Series, idx) -> pd.Series:
    """Units per signal, set on each year's first business day from up to three
    years of the prompt crack's P&L, held for the year."""
    out = pd.Series(np.nan, index=idx)
    for y in sorted(set(idx.year)):
        days = idx[idx.year == y]
        hist = unit_pnl.loc[: days[0] - pd.Timedelta(days=1)].dropna().iloc[-756:]
        hist = hist[hist != 0]
        if len(hist) < 60:
            continue
        vol = hist.std() * np.sqrt(252)
        out.loc[days] = np.floor(RISK * CAPITAL / vol)
    return out


def season_trades(r: pd.Series, on: pd.Series) -> dict:
    """P&L of each seasonal trade (each contiguous in-trade stretch, plus exit)."""
    start = (on > 0) & (on.shift(1, fill_value=0) == 0)
    trade_id = start.cumsum().where(on.shift(1, fill_value=0) + on > 0)
    out = {}
    for t, g in r.groupby(trade_id):
        if np.isnan(t):
            continue
        label = f"{g.index[0]:%Y-%m}"
        out[label] = float((1 + g).prod() - 1)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    data = {
        k: markets.inputs(markets.MARKETS[k])
        for k in ("wti", "heating_oil", "gasoline")
    }
    idx = data["wti"]["px"].index
    idx = idx[(idx >= "2003-01-01") & (idx <= END)]
    stocks = {
        "distillate": stock_z(load_stocks("distillate")),
        "gasoline": stock_z(load_stocks("gasoline")),
    }

    rets, books_, trades, trading, plac = {}, {}, {}, {}, {}
    oil0 = multi.held_return(data["wti"]).loc[START:END]
    for name in CRACKS:
        prompt = legs_for(name, data, data["wti"]["w2"])
        size = annual_size(engine.legs_unit_pnl(prompt, idx), idx)
        w_strip, on = strip_weights(idx, SEASONS[name])
        strip = legs_for(name, data, w_strip)
        sigs = {
            "Seasonal strips": (on, strip),
            "Inventories": (inventory_signal(name, stocks, idx), prompt),
        }
        for strat, (sig, legs) in sigs.items():
            log_trial(
                f"crack {name} {strat}",
                {
                    "protocol": "CRACKS_PROTOCOL.md",
                    "spread": name,
                    "strategy": strat,
                    "risk": RISK,
                    "start": START,
                },
            )
            b = engine.run_legs(sig, legs, idx, size=size, unit_cost=listed_cost(name))
            books_[(name, strat)] = b
            rets[(name, strat)] = b.returns.loc[START:END]
        log_trial(
            f"crack {name} Combined",
            {
                "protocol": "CRACKS_PROTOCOL.md",
                "spread": name,
                "strategy": "Combined",
                "start": START,
            },
        )
        rets[(name, "Combined")] = (
            rets[(name, "Seasonal strips")] + rets[(name, "Inventories")]
        ) / 2
        trades[name] = season_trades(rets[(name, "Seasonal strips")], on.loc[START:END])
        plac[name] = placebos(name, data, idx, size, oil0)
        plac[name]["actual strips"] = {
            "sharpe": book.tearsheet(rets[(name, "Seasonal strips")], oil0)["sharpe"],
            "return_per_year_in_market": float(
                rets[(name, "Seasonal strips")].mean() * 252 / on.loc[START:END].mean()
            ),
        }
        vol = pd.concat([data[k]["vol"] for k in data], axis=1)
        oi = pd.concat([data[k]["oi"] for k in data], axis=1)
        for strat in ("Seasonal strips", "Inventories"):
            b = books_[(name, strat)]
            s = engine.Book(
                *(
                    x.loc["2010-01-01":]
                    for x in (b.returns, b.gross, b.holdings, b.units, b.cost)
                )
            )
            trading[(name, strat)] = {
                "gross_ann": float(b.gross.loc[START:END].mean() * 252),
                "cost_ann": float(b.cost.loc[START:END].mean() * 252 / CAPITAL),
                "avg_units_when_on": float(
                    b.units.loc[START:END].replace(0, np.nan).abs().mean()
                ),
                "time_in_market": float((b.units.loc[START:END] != 0).mean()),
                "capacity_q95_since_2010": engine.capacity(s, vol, oi, quantile=0.95)[
                    "capacity"
                ],
            }

    R = pd.DataFrame(rets)
    oil = multi.held_return(data["wti"]).loc[START:END]
    core_runs = {
        m: multi.run_market(markets.inputs(markets.MARKETS[m]))
        for m in ("wti", "brent", "natural_gas")
    }
    core_own = pd.DataFrame(
        {(m, s): b.returns for m, bs in core_runs.items() for s, b in bs.items()}
    ).loc[START:END]
    core = book.combine(core_own.fillna(0.0).where(core_own.notna().cummax()))
    log_trial(
        "crack Crack book",
        {
            "protocol": "CRACKS_PROTOCOL.md",
            "sleeves": "seasonal + inventories, 3 cracks",
        },
    )
    base = R.xs("Seasonal strips", axis=1, level=1).join(
        R.xs("Inventories", axis=1, level=1),
        lsuffix=" seasonal",
        rsuffix=" inventories",
    )
    crack_book = book.combine(base)
    log_trial(
        "crack Core + cracks",
        {"protocol": "CRACKS_PROTOCOL.md", "halves": ["core outright", "crack book"]},
    )
    both = book.combine(pd.concat([core, crack_book], axis=1).dropna(how="all"))

    res = {
        "start": START,
        "end": END,
        "risk_budget": RISK,
        "sleeves": {},
        "seasonal_trades": trades,
        "books": {},
    }
    for (name, strat), r in R.items():
        p = book.tearsheet(r, oil)
        res["sleeves"][f"{name} | {strat}"] = {
            **{k: v for k, v in p.items() if k not in ("annual", "crises")},
            "annual": p["annual"],
            "grind": multi.window_return(r, *GRIND),
            "corr_to_core": float(r.corr(core)),
            **({"trading": trading[(name, strat)]} if (name, strat) in trading else {}),
        }
    for k, r in {
        "Core outright book": core,
        "Crack book": crack_book,
        "Core + cracks": both,
    }.items():
        res["books"][k] = {
            "performance": book.tearsheet(r, oil),
            "grind": multi.window_return(r.dropna(), *GRIND),
            "corr_to_core": float(r.corr(core)),
        }
    res["placebos"] = plac
    res["seasonal_hit_rate"] = {
        n: float(np.mean([v > 0 for v in t.values()])) for n, t in trades.items()
    }
    with open(os.path.join(OUT, "cracks.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    figures(core, crack_book, both, trades, R)

    for k, v in res["sleeves"].items():
        t = v.get("trading", {})
        print(
            f"{k:38s} CAGR {v['cagr']:6.1%} SR {v['sharpe']:5.2f} "
            f"MDD {v['max_drawdown']:6.1%} 22-25 {v['grind']:+.0%} "
            f"cost {t.get('cost_ann', np.nan):.2%} gross {t.get('gross_ann', np.nan):.2%}"
        )
    for k, v in res["books"].items():
        p = v["performance"]
        print(
            f"{k:20s} CAGR {p['cagr']:6.1%} SR {p['sharpe']:5.2f} "
            f"MDD {p['max_drawdown']:6.1%} 22-25 {v['grind']:+.0%} "
            f"corr core {v['corr_to_core']:.2f}"
        )
    print("seasonal hit rates", res["seasonal_hit_rate"])


def figures(core, crack_book, both, trades, R) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    for r, label, c, lw in (
        (core, "Core outright book", "#1f4e79", 1.6),
        (crack_book, "Crack book", "#c55a11", 1.8),
        (both, "Core + cracks", "black", 2.4),
    ):
        ax.plot((1 + r.dropna()).cumprod(), label=label, color=c, lw=lw)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log), net of costs")
    ax.set_title(
        "Crack spreads traded as seasonal strips and on inventories, 2006-2026"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "equity.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(
        len(trades), 1, figsize=(11, 2.4 * len(trades)), sharex=False
    )
    for ax, (name, t) in zip(np.atleast_1d(axes), trades.items()):
        k = list(t)
        v = np.array(list(t.values())) * 100
        ax.bar(range(len(k)), v, color=np.where(v > 0, "#548235", "#c00000"))
        ax.set_xticks(range(len(k)), k, rotation=90, fontsize=7)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("% of capital")
        ax.set_title(
            f"{name}: each seasonal trade, {np.mean(v > 0):.0%} profitable", fontsize=10
        )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "seasonal_trades.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
