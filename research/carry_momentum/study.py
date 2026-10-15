"""Run the pre-registered carry-momentum study and apply the kill criteria.

python study.py     -> results/study.json, results/*.png, prints the verdict

Every number here is defined in PROTOCOL.md; departures are in DEVIATIONS.md.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from itertools import product

import curve
import engine
import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap
from reconcile import contracts

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
CAPSTONE = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "quant-pipeline-testbed")
)
sys.path.insert(0, CAPSTONE)

from capstone.pbo import cscv
from capstone.validation.sharpe import (
    deflated_sharpe_ratio,
    sharpe_variance,
)
from capstone.validation.spanning import spanning_test

START = "1990-01-01"
SEEN = ("2008-01-01", "2024-04-05")
UNSEEN = [("1990-01-01", "2007-12-31"), ("2024-04-08", "2026-04-02")]
PRIOR_TRIALS = 150  # every WTI configuration run in this repository before the protocol
LEDGER = os.path.join(RESULTS, "trials.jsonl")

GRID = [
    (s, l, t)
    for s, l, t in product(
        (1, 5), (10, 21, 42, 63), (("1m", "13m"), ("2m", "7m"), ("1m", "4m"))
    )
]
PRIMARY = (1, 21, ("1m", "13m"))

KILL = {
    "K1": ("H1 net Sharpe, unseen windows", "<", 0.20),
    "K2": ("H1 90% bootstrap lower bound, full sample", "<", 0.0),
    "K3": ("H1 deflated Sharpe probability", "<", 0.90),
    "K4": ("PBO across 24 variants", ">", 0.50),
    "K5": ("H1 alpha t-stat after factors", "<", 2.0),
    "K6": ("H1 net Sharpe, costs doubled", "<", 0.20),
    "K7": ("Capacity at 5% ADV / 2% OI ($)", "<", 10e6),
}


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    return {
        "sharpe": sharpe(r),
        "ann_return": float(r.mean() * 252),
        "ann_vol": float(r.std() * np.sqrt(252)),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "days": len(r),
    }


def window(r: pd.Series, *spans) -> pd.Series:
    return pd.concat([r.loc[a:b] for a, b in spans]).sort_index()


def log_trial(name: str, params: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(
            json.dumps(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "name": name,
                    "params": params,
                }
            )
            + "\n"
        )


def trial_count() -> int:
    if not os.path.exists(LEDGER):
        return PRIOR_TRIALS
    names = {json.loads(line)["name"] for line in open(LEDGER)}
    return PRIOR_TRIALS + len(names)


def bootstrap_sharpe_ci(
    r: pd.Series, reps: int = 10000, block: int = 21, seed: int = 0
) -> tuple[float, float]:
    x = r.dropna().to_numpy()
    bs = StationaryBootstrap(block, x, seed=seed)
    draws = bs.apply(lambda a: a.mean() / a.std() * np.sqrt(252), reps).ravel()
    return float(np.percentile(draws, 5)), float(np.percentile(draws, 95))


def bootstrap_63d(
    r: pd.Series, reps: int = 10000, block: int = 21, seed: int = 1
) -> dict:
    """Distribution of 63-day compounded returns, for the live test."""
    x = r.dropna().to_numpy()
    bs = StationaryBootstrap(block, x, seed=seed)
    draws = np.array([np.prod(1 + d[0][0][:63]) - 1 for d in bs.bootstrap(reps)])
    return {f"p{q}": float(np.percentile(draws, q)) for q in (5, 25, 50, 75, 95)}


# ---------------------------------------------------------------------------


def build_inputs() -> dict:
    p = contracts()
    p = p[p.date >= "1988-01-01"]
    # Drop stale holiday rows: dates with fewer than five listed contracts (DEVIATIONS #8).
    n = p.groupby("date").contract.transform("count")
    p = p[n >= 5]
    path = os.path.join(DATA, "cm_curve.parquet")
    if os.path.exists(path):
        cm = pd.read_parquet(path)
    else:
        cm = curve.constant_maturity(p)
        cm.to_parquet(path)
    px = p.pivot(index="date", columns="contract", values="settle")
    vol = p.pivot(index="date", columns="contract", values="volume")
    oi = p.pivot(index="date", columns="contract", values="oi")
    legs = {k: engine.leg_weights(p, k) for k in (1, 6)}
    return {
        "panel": p,
        "cm": cm,
        "px": px,
        "vol": vol,
        "oi": oi,
        "w2": legs[1],
        "w7": legs[6],
    }


def strategy(
    d: dict,
    short: int,
    long: int,
    tenors: tuple[str, str],
    spread: bool = True,
    costs: engine.Costs = engine.Costs(),
) -> engine.Book:
    sig = engine.carry_momentum(engine.carry(d["cm"], *tenors), short, long)
    legs = (
        {"near": (1.0, d["w2"]), "far": (-1.0, d["w7"])}
        if spread
        else {"leg": (1.0, d["w2"])}
    )
    return engine.run(sig, legs, d["px"], costs=costs, spread_trade=spread)


def ts_factors(d: dict) -> pd.DataFrame:
    """WTI time-series factors from the same contracts (DEVIATIONS.md #5), daily returns.

    Each holds the 2nd listed contract, 15% vol target, positions at t+1 like the strategy.
    """
    cm, px, w2 = d["cm"], d["px"], d["w2"]
    near = curve.held_excess_return(d["panel"], curve.gsci_targets(d["panel"]))
    nxt = curve.held_excess_return(d["panel"], curve.gsci_targets(d["panel"], offset=1))
    mom = lambda r, n: np.log1p(r).rolling(n, min_periods=n).sum()
    signals = {
        "tsmom_wti": np.sign(mom(near, 252)),
        "carry_wti": np.sign(engine.carry(cm, "1m", "13m")),
        "basis_mom_wti": np.sign(mom(near, 252) - mom(nxt, 252)),
    }
    out = {}
    for k, s in signals.items():
        b = engine.run(s.fillna(0.0), {"leg": (1.0, w2)}, px, spread_trade=False)
        out[k] = b.gross  # gross of costs, as published factors are (DEVIATIONS #8)
    return pd.DataFrame(out)


def aqr_factor() -> pd.Series:
    x = pd.read_excel(
        os.path.join(DATA, "aqr_tsmom_monthly.xlsx"),
        sheet_name="TSMOM Factors",
        header=None,
    )
    hdr = x.index[x.iloc[:, 2].astype(str).eq("TSMOM^CM")][0]
    df = x.iloc[hdr + 1 :, [0, 2]].dropna()
    df = df[pd.to_datetime(df[0], errors="coerce").notna()]
    return pd.Series(
        df[2].astype(float).values,
        index=pd.to_datetime(df[0]).dt.to_period("M"),
        name="tsmom_cm_aqr",
    )


def monthly(r: pd.Series) -> pd.Series:
    m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    return m


def novelty(h: pd.Series, d: dict) -> dict:
    f = ts_factors(d)
    fm = pd.concat([f.apply(monthly), aqr_factor()], axis=1)
    y = monthly(h)
    fm, y = fm.loc[y.index.min() :], y
    fm.index = fm.index.to_timestamp("M")
    y.index = y.index.to_timestamp("M")
    res = spanning_test(y.loc[START:], fm.dropna())
    return {
        "alpha_annual": res.alpha / 252 * 12,
        "t_stat": res.t_stat,
        "loadings": res.loadings,
        "r_squared": res.r_squared,
        "n_months": res.n_obs,
        "nw_lags": res.lags,
    }


def decomposition(d: dict, book: engine.Book) -> dict:
    """Attribute the strategy's gross P&L to roll yield and spot moves.

    For each leg, the held contract's daily $ change splits into the move of the
    1-month constant-maturity point (spot) and the rest (roll-down along the curve).
    """
    cm1 = d["cm"]["1m"]
    spot_dollar = cm1.diff()
    px = d["px"]
    h = book.holdings.shift(1).fillna(0.0)
    dp = px.reindex(columns=h.columns).diff().fillna(0.0)
    total = (h * dp).sum(axis=1) * engine.BBL
    net_contracts = h.sum(axis=1)
    spot_part = (net_contracts * spot_dollar.reindex(h.index).fillna(0.0)) * engine.BBL
    roll_part = total - spot_part
    cap = 10e6
    return {
        "gross_ann_return": float(total.loc[START:].mean() * 252 / cap),
        "spot_ann_return": float(spot_part.loc[START:].mean() * 252 / cap),
        "roll_and_curve_ann_return": float(roll_part.loc[START:].mean() * 252 / cap),
    }


def mechanism(d: dict) -> dict:
    """Bouchouev's storage hedgers: does lagged commercial short hedging deepen contango?"""
    import statsmodels.api as sm

    cot = pd.read_csv(
        os.path.join(DATA, "cot_wti_disagg.csv"), parse_dates=["date"]
    ).set_index("date")
    pm_net_short = (
        cot.prod_merc_positions_short - cot.prod_merc_positions_long
    ) / cot.open_interest_all
    c = engine.carry(d["cm"], "1m", "13m")
    cot = cot.loc[: c.index.max()]
    pm_net_short = pm_net_short.loc[: c.index.max()]
    c_tue = c.reindex(cot.index, method="ffill")
    df = pd.DataFrame(
        {"d_carry": c_tue.diff(), "d_hedge_lag": pm_net_short.diff().shift(1)}
    ).dropna()
    m = sm.OLS(df.d_carry, sm.add_constant(df.d_hedge_lag)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4}
    )
    return {
        "coef": float(m.params["d_hedge_lag"]),
        "t_stat": float(m.tvalues["d_hedge_lag"]),
        "n_weeks": int(m.nobs),
        "predicted_sign": "negative",
    }


def mechanism_conditional(d: dict, book: engine.Book, sig: pd.Series) -> dict:
    """Weekly strategy returns when the signal agrees vs disagrees with lagged hedger flow."""
    cot = pd.read_csv(
        os.path.join(DATA, "cot_wti_disagg.csv"), parse_dates=["date"]
    ).set_index("date")
    pm = (
        cot.prod_merc_positions_short - cot.prod_merc_positions_long
    ) / cot.open_interest_all
    # More hedger shorting -> deeper contango -> carry falls -> carry-momentum short.
    flow = -np.sign(pm.diff())
    wk = (1 + book.returns).resample("W-TUE").prod() - 1
    s = sig.resample("W-TUE").last().shift(1)
    f = flow.reindex(wk.index).shift(1)
    agree = (s * f) > 0
    a, b = wk[agree].loc["2006-07":], wk[~agree & f.notna()].loc["2006-07":]
    return {
        "agree_ann_return": float(a.mean() * 52),
        "disagree_ann_return": float(b.mean() * 52),
        "agree_weeks": len(a),
        "disagree_weeks": len(b),
    }


def main() -> None:
    d = build_inputs()
    out = {"run_time": datetime.now(timezone.utc).isoformat()}

    # Variants: log every one before looking at its result.
    books, rets = {}, {}
    for s, lng, t in GRID:
        name = f"H1 spread s={s} l={lng} {t[0]}-{t[1]}"
        log_trial(
            name, {"short": s, "long": lng, "tenors": t, "instrument": "2nd-7th spread"}
        )
        books[(s, lng, t)] = strategy(d, s, lng, t)
        rets[name] = books[(s, lng, t)].returns.loc[START:]
    log_trial("H2 outright s=1 l=21 1m-13m", {"instrument": "2nd outright"})
    h1 = books[PRIMARY]
    h2 = strategy(d, *PRIMARY, spread=False)
    r1, r2 = h1.returns.loc[START:], h2.returns.loc[START:]
    sig = engine.carry_momentum(engine.carry(d["cm"], "1m", "13m"), 1, 21)

    out["H1"] = {
        "full": stats(r1),
        "seen": stats(r1.loc[SEEN[0] : SEEN[1]]),
        "unseen": stats(window(r1, *UNSEEN)),
        "unseen_1990_2007": stats(r1.loc[UNSEEN[0][0] : UNSEEN[0][1]]),
        "unseen_2024_2026": stats(r1.loc[UNSEEN[1][0] : UNSEEN[1][1]]),
    }
    out["H2"] = {
        "full": stats(r2),
        "seen": stats(r2.loc[SEEN[0] : SEEN[1]]),
        "unseen": stats(window(r2, *UNSEEN)),
    }

    lo, hi = bootstrap_sharpe_ci(r1)
    out["H1"]["bootstrap_90ci"] = [lo, hi]
    n_trials = trial_count()
    rr = r1.dropna()
    v = sharpe_variance(rr, "hac")
    trial_sharpes = np.array([sharpe(x) for x in rets.values()])
    out["H1"]["deflated_sharpe"] = deflated_sharpe_ratio(
        sharpe(rr),
        n_trials,
        len(rr),
        skew=float(rr.skew()),
        kurtosis=float(rr.kurt() + 3),
        variance=v,
        trials_sharpe_variance=float(np.var(trial_sharpes))
        if len(trial_sharpes) > 1
        else None,
    )
    out["n_trials"] = n_trials
    pbo = cscv(pd.DataFrame(rets).dropna(), n_blocks=16)
    out["pbo"] = {"pbo": pbo["pbo"], "degradation_slope": pbo.get("degradation_slope")}
    out["variants"] = {k: sharpe(v) for k, v in rets.items()}

    out["H1"]["novelty"] = novelty(r1, d)
    out["H2"]["novelty"] = novelty(r2, d)
    out["H1"]["costs_doubled"] = stats(
        strategy(d, *PRIMARY, costs=engine.Costs().scaled(2)).returns.loc[START:]
    )
    h1s = engine.Book(h1.returns, h1.gross, h1.holdings.loc[START:], h1.units, h1.cost)
    h2s = engine.Book(h2.returns, h2.gross, h2.holdings.loc[START:], h2.units, h2.cost)
    out["H1"]["capacity"] = engine.capacity(h1s, d["vol"], d["oi"])
    out["H1"]["capacity_q95"] = engine.capacity(h1s, d["vol"], d["oi"], quantile=0.95)
    out["H1"]["capacity_since_2010"] = engine.capacity(
        engine.Book(h1.returns, h1.gross, h1.holdings.loc["2010":], h1.units, h1.cost),
        d["vol"],
        d["oi"],
    )
    out["H2"]["capacity"] = engine.capacity(h2s, d["vol"], d["oi"])
    out["H1"]["decomposition"] = decomposition(d, h1)
    out["H2"]["decomposition"] = decomposition(d, h2)
    out["H1"]["turnover_units_per_year"] = float(
        h1.units.diff().abs().loc[START:].mean() * 252
    )
    out["H1"]["contracts_traded_per_year_incl_rolls"] = float(
        h1.holdings.diff().abs().sum(axis=1).loc[START:].mean() * 252
    )
    out["H1"]["cost_ann"] = float(h1.cost.loc[START:].mean() * 252 / 10e6)
    out["mechanism"] = mechanism(d)
    out["mechanism_conditional_H1"] = mechanism_conditional(d, h1, sig)
    out["live_expectation_63d"] = bootstrap_63d(r1)

    # H2 through the same criteria (DEVIATIONS.md #7).
    rets2 = {}
    for s_, lng, t in GRID:
        name = f"H2 outright s={s_} l={lng} {t[0]}-{t[1]}"
        log_trial(
            name, {"short": s_, "long": lng, "tenors": t, "instrument": "2nd outright"}
        )
        rets2[name] = strategy(d, s_, lng, t, spread=False).returns.loc[START:]
    lo2, hi2 = bootstrap_sharpe_ci(r2)
    rr2 = r2.dropna()
    out["n_trials"] = n_trials = trial_count()
    out["H2"]["bootstrap_90ci"] = [lo2, hi2]
    out["H2"]["deflated_sharpe"] = deflated_sharpe_ratio(
        sharpe(rr2),
        n_trials,
        len(rr2),
        skew=float(rr2.skew()),
        kurtosis=float(rr2.kurt() + 3),
        variance=sharpe_variance(rr2, "hac"),
        trials_sharpe_variance=float(np.var([sharpe(x) for x in rets2.values()])),
    )
    out["H1"]["deflated_sharpe"] = deflated_sharpe_ratio(
        sharpe(rr),
        n_trials,
        len(rr),
        skew=float(rr.skew()),
        kurtosis=float(rr.kurt() + 3),
        variance=v,
        trials_sharpe_variance=float(np.var(trial_sharpes)),
    )
    out["pbo_H2"] = {"pbo": cscv(pd.DataFrame(rets2).dropna(), n_blocks=16)["pbo"]}
    out["variants_H2"] = {k: sharpe(x) for k, x in rets2.items()}
    out["H2"]["costs_doubled"] = stats(
        strategy(d, *PRIMARY, spread=False, costs=engine.Costs().scaled(2)).returns.loc[
            START:
        ]
    )
    out["H2"]["capacity_q95"] = engine.capacity(h2s, d["vol"], d["oi"], quantile=0.95)
    out["H2"]["capacity_q99"] = engine.capacity(h2s, d["vol"], d["oi"], quantile=0.99)
    out["H2"]["capacity_since_2010"] = engine.capacity(
        engine.Book(h2.returns, h2.gross, h2.holdings.loc["2010":], h2.units, h2.cost),
        d["vol"],
        d["oi"],
    )
    out["H2"]["turnover_units_per_year"] = float(
        h2.units.diff().abs().loc[START:].mean() * 252
    )
    out["H2"]["contracts_traded_per_year_incl_rolls"] = float(
        h2.holdings.diff().abs().sum(axis=1).loc[START:].mean() * 252
    )
    out["H2"]["cost_ann"] = float(h2.cost.loc[START:].mean() * 252 / 10e6)
    out["H2"]["by_window"] = {
        "unseen_1990_2007": stats(r2.loc[UNSEEN[0][0] : UNSEEN[0][1]]),
        "unseen_2024_2026": stats(r2.loc[UNSEEN[1][0] : UNSEEN[1][1]]),
    }
    out["H1"]["zero_cost"] = stats(
        strategy(d, *PRIMARY, costs=engine.Costs(0, 0)).returns.loc[START:]
    )
    out["H2"]["zero_cost"] = stats(
        strategy(d, *PRIMARY, spread=False, costs=engine.Costs(0, 0)).returns.loc[
            START:
        ]
    )
    vals2 = {
        "K1": out["H2"]["unseen"]["sharpe"],
        "K2": lo2,
        "K3": out["H2"]["deflated_sharpe"],
        "K4": out["pbo_H2"]["pbo"],
        "K5": out["H2"]["novelty"]["t_stat"],
        "K6": out["H2"]["costs_doubled"]["sharpe"],
        "K7": out["H2"]["capacity"]["capacity"],
    }
    out["live_expectation_63d_H2"] = bootstrap_63d(r2)
    np.save(
        os.path.join(RESULTS, "_h2.npy"),
        np.c_[r2.index.values.astype("int64"), r2.values],
    )

    vals = {
        "K1": out["H1"]["unseen"]["sharpe"],
        "K2": lo,
        "K3": out["H1"]["deflated_sharpe"],
        "K4": out["pbo"]["pbo"],
        "K5": out["H1"]["novelty"]["t_stat"],
        "K6": out["H1"]["costs_doubled"]["sharpe"],
        "K7": out["H1"]["capacity"]["capacity"],
    }
    verdict = {}
    for k, (label, op, thr) in KILL.items():
        x = vals[k]
        fail = (x < thr) if op == "<" else (x > thr)
        verdict[k] = {
            "criterion": label,
            "value": x,
            "threshold": f"{op} {thr}",
            "killed": bool(fail),
        }
    out["kill_criteria"] = verdict
    out["deploy"] = not any(v["killed"] for v in verdict.values())
    verdict2 = {}
    for k, (label, op, thr) in KILL.items():
        x = vals2[k]
        fail = (x < thr) if op == "<" else (x > thr)
        verdict2[k] = {
            "criterion": label.replace("H1", "H2"),
            "value": x,
            "threshold": f"{op} {thr}",
            "killed": bool(fail),
        }
    out["kill_criteria_H2"] = verdict2
    out["deploy_H2"] = not any(v["killed"] for v in verdict2.values())

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "study.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    np.save(
        os.path.join(RESULTS, "_h1.npy"),
        np.c_[r1.index.values.astype("int64"), r1.values],
    )
    for k, v in verdict.items():
        print(
            f"{k} {v['criterion']:<45} {v['value']:>14.3f}  {v['threshold']:<10} {'KILLED' if v['killed'] else 'pass'}"
        )
    print("deploy H1:", out["deploy"])
    for k, v in verdict2.items():
        print(
            f"{k} {v['criterion']:<45} {v['value']:>14.3f}  {v['threshold']:<10} {'KILLED' if v['killed'] else 'pass'}"
        )
    print("deploy H2:", out["deploy_H2"], " trials:", out["n_trials"])


if __name__ == "__main__":
    main()
