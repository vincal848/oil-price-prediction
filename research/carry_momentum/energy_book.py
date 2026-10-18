"""The energy book: six outright markets plus the seasonal crack strips.

python energy_book.py   -> results/energy/energy.json, results/energy/*.png

Follows the addendum to CRACKS_PROTOCOL.md: the 18 outright sleeves of the
all-markets book (multi.py) and the three seasonal strip sleeves (cracks.py),
weighted equally and scaled to 15% volatility on trailing one-year volatility.
"""

from __future__ import annotations

import json
import os

import book
import cracks
import engine
import markets
import multi
import numpy as np
import pandas as pd
from study import RESULTS, log_trial

START = book.START
OUT = os.path.join(RESULTS, "energy")
OUTRIGHT = ["wti", "brent", "natural_gas", "heating_oil", "gasoline", "gasoil"]


def outright_sleeves(end) -> pd.DataFrame:
    own = {}
    for k in OUTRIGHT:
        d = markets.inputs(markets.MARKETS[k])
        for s, b in multi.run_market(d).items():
            own[(d["market"].name, s)] = b.returns
    own = pd.DataFrame(own).loc[START:end]
    return own, own.fillna(0.0).where(own.notna().cummax())


def strip_sleeves() -> pd.DataFrame:
    data = {
        k: markets.inputs(markets.MARKETS[k])
        for k in ("wti", "heating_oil", "gasoline")
    }
    idx = data["wti"]["px"].index
    idx = idx[(idx >= "2003-01-01") & (idx <= cracks.END)]
    out = {}
    for name in cracks.CRACKS:
        prompt = cracks.legs_for(name, data, data["wti"]["w2"])
        size = cracks.annual_size(engine.legs_unit_pnl(prompt, idx), idx)
        w, on = cracks.strip_weights(idx, cracks.SEASONS[name])
        log_trial(
            f"crack {name} Seasonal strips",
            {
                "protocol": "CRACKS_PROTOCOL.md",
                "spread": name,
                "strategy": "Seasonal strips",
                "risk": cracks.RISK,
                "start": cracks.START,
            },
        )
        b = engine.run_legs(
            on,
            cracks.legs_for(name, data, w),
            idx,
            size=size,
            unit_cost=cracks.listed_cost(name),
        )
        out[(name, "Seasonal strips")] = b.returns.loc[cracks.START : cracks.END]
    return pd.DataFrame(out)


COMPONENTS = {
    "WTI": ["WTI"],
    "Brent": ["Brent"],
    "Natural gas": ["Natural gas"],
    "Products, outright": ["Heating oil", "Gasoline", "Gasoil"],
    "Crack strips": list(cracks.CRACKS),
}


def attribution(sleeves: pd.DataFrame, oil: pd.Series) -> dict:
    """Split the book into what each component and strategy contributes.

    The book is lev_t times the mean of the sleeves live on day t, so each
    sleeve contributes lev_t * r_it / n_t, and the contributions add up to the
    book exactly. Risk share is each part's covariance with the book over the
    book's variance. Leave-one-out books drop one component and are rebuilt.
    """
    eq = sleeves.mean(axis=1)
    vol = eq.rolling(252, min_periods=63).std().shift(1) * np.sqrt(252)
    lev = (0.15 / vol).clip(upper=3.0).fillna(1.0)
    n = sleeves.notna().sum(axis=1)
    contrib = sleeves.mul(lev / n, axis=0)
    total = contrib.sum(axis=1)

    def summarise(groups: dict) -> dict:
        out = {}
        for g, cols in groups.items():
            c = contrib[cols].sum(axis=1)
            out[g] = {
                "return_contribution": float(c.mean() * 252),
                "risk_share": float(c.cov(total) / total.var()),
                "grind_contribution": float(c.loc[slice(*multi.GRIND)].sum()),
            }
        return out

    by_component = {
        k: [c for c in sleeves.columns if c[0] in v] for k, v in COMPONENTS.items()
    }
    by_strategy = {}
    for c in sleeves.columns:
        by_strategy.setdefault(c[1], []).append(c)
    loo = {}
    for k, cols in by_component.items():
        r = book.combine(sleeves.drop(columns=cols))
        p = book.tearsheet(r, oil)
        loo[k] = {
            "cagr": p["cagr"],
            "sharpe": p["sharpe"],
            "max_drawdown": p["max_drawdown"],
            "grind": multi.window_return(r.dropna(), *multi.GRIND),
        }
    return {
        "check_max_abs_diff": float((total - book.combine(sleeves)).abs().max()),
        "by_component": summarise(by_component),
        "by_strategy": summarise(by_strategy),
        "leave_one_out": loo,
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    wti = markets.inputs(markets.MARKETS["wti"])
    end = wti["panel"].date.max()
    own, out = outright_sleeves(end)
    strips = strip_sleeves().reindex(out.index)
    strips = strips.fillna(0.0).where(strips.notna().cummax())

    without = book.combine(out)
    log_trial(
        "energy All markets + crack strips",
        {
            "protocol": "CRACKS_PROTOCOL.md addendum",
            "outright_sleeves": 18,
            "strip_sleeves": 3,
            "weights": "equal",
            "vol_target": 0.15,
        },
    )
    with_ = book.combine(pd.concat([out, strips], axis=1))
    wti_only = book.combine(own["WTI"].dropna(how="all"))  # WTI's own calendar

    oil = multi.held_return(wti).loc[START:end]
    books = {
        "WTI only": wti_only,
        "All markets": without,
        "All markets + crack strips": with_,
    }
    res = {"start": START, "end": str(end.date()), "books": {}}
    for k, r in books.items():
        p = book.tearsheet(r, oil)
        res["books"][k] = {**p, "grind": multi.window_return(r.dropna(), *multi.GRIND)}
    eq = (1 + oil.fillna(0.0)).cumprod()
    res["oil_drops"] = {
        n: {
            "wti": float(eq.loc[b] / eq.loc[a] - 1),
            **{k: multi.window_return(r, a, b) for k, r in books.items()},
        }
        for n, (a, b) in book.DROPS.items()
    }
    res["strip_corr_to_all_markets"] = float(strips.mean(axis=1).corr(without))
    log_trial("energy leave-one-out decomposition", {"diagnostic": True})
    res["attribution"] = attribution(pd.concat([out, strips], axis=1), oil)
    with open(os.path.join(OUT, "energy.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    figure(books)
    attribution_figure(res["attribution"])

    for k, p in res["books"].items():
        print(
            f"{k:28s} CAGR {p['cagr']:6.1%} SR {p['sharpe']:5.2f} "
            f"Sortino {p['sortino']:4.2f} MDD {p['max_drawdown']:6.1%} "
            f"longest DD {p['longest_drawdown_days']} yrs+ {p['positive_years']} "
            f"2022-25 {p['grind']:+.0%} x{p['total_return_multiple']:.2f}"
        )
    print(
        "drops",
        {
            n: {k: round(v * 100) for k, v in d.items()}
            for n, d in res["oil_drops"].items()
        },
    )
    print("strip corr", round(res["strip_corr_to_all_markets"], 2))


def attribution_figure(att: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    comp = att["by_component"]
    names = list(comp)[::-1]
    for ax, key, title, scale in (
        (axes[0], "return_contribution", "Return contribution, % a year", 100),
        (axes[1], "risk_share", "Share of book risk, %", 100),
        (axes[2], "grind_contribution", "Contribution over 2022-25, %", 100),
    ):
        v = [comp[n][key] * scale for n in names]
        ax.barh(names, v, color=["#548235" if x > 0 else "#c00000" for x in v])
        for i, x in enumerate(v):
            ax.text(
                x,
                i,
                f" {x:+.1f}" if key != "risk_share" else f" {x:.0f}",
                va="center",
                fontsize=9,
            )
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(title, fontsize=10)
    fig.suptitle("What each component adds to the energy book, 2003-2026")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "attribution.png"), dpi=150)
    plt.close(fig)


def figure(books: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    style = {
        "WTI only": ("#9dc3e6", 1.3),
        "All markets": ("#1f4e79", 1.6),
        "All markets + crack strips": ("black", 2.6),
    }
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1], sharex=True)
    for k, r in books.items():
        c, lw = style[k]
        a.plot((1 + r.dropna()).cumprod(), label=k, color=c, lw=lw)
        b.plot(book.drawdown(r.dropna()) * 100, color=c, lw=lw * 0.5)
    a.set_yscale("log")
    a.set_ylabel("Growth of $1 (log), net of costs")
    a.set_title("The energy book, 2003-2026, 15% vol target, net of costs")
    a.legend(frameon=False, loc="upper left")
    b.set_ylabel("Drawdown %")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "equity.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
