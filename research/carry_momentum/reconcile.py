"""Reconcile the rebuilt curve against independent published series.

1. Ranks 1-4 vs EIA's Cushing WTI contract 1-4 settlements (RCLC1-4).
2. Roll-scheduled excess return vs a published crude excess-return index.

Run: python reconcile.py   (writes results/reconciliation.json)
"""

import json
import os

import numpy as np
import pandas as pd

import curve

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def contracts() -> pd.DataFrame:
    path = os.path.join(DATA, "contracts.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    raw = [pd.read_csv(os.path.join(DATA, f), compression="gzip") for f in ("ncl_1983_2005.csv.gz", "ncl_2006_2028.csv.gz")]
    p = curve.load_contracts(raw)
    p.to_parquet(path)
    return p


def eia_contracts() -> pd.DataFrame:
    return pd.concat(
        {k: pd.read_csv(os.path.join(DATA, f"F{k}.csv"), index_col=0, parse_dates=True).iloc[:, 0] for k in range(1, 5)},
        axis=1,
        sort=True,
    )


def rank_match(panel: pd.DataFrame, eia: pd.DataFrame) -> dict:
    w = curve.by_rank(panel, "settle", 4)
    out = {}
    for k in range(1, 5):
        a, b = w[k].align(eia[k], join="inner")
        diff = (a - b).abs().dropna()
        out[f"C{k}"] = {
            "days": int(len(diff)),
            "share_to_the_cent": float((diff < 0.005).mean()),
            "share_within_2c": float((diff < 0.02).mean()),
            "max_abs_diff": float(diff.max()),
            "worst_dates": [str(x.date()) for x in diff.nlargest(3).index],
        }
    return out


def main() -> None:
    p = contracts()
    g = gsci()
    excl = (g.index >= "2020-03-16") & (g.index <= "2020-06-30")
    res = {
        "rank_match_vs_EIA": rank_match(p, eia_contracts()),
        "gsci_crude_all": index_match(p, g),
        "gsci_crude_ex_2020_intervention": index_match(p, g.mask(excl)),
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "reconciliation.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def gsci() -> pd.Series:
    x = pd.read_excel(os.path.join(DATA, "gsci_crude.xls"), header=None)
    start = x.index[x[0].astype(str).str.startswith("Effective date")][0] + 1
    s = x.iloc[start:].dropna()
    s = s[pd.to_datetime(s[0], errors="coerce").notna()]
    return pd.Series(s[1].astype(float).values, index=pd.to_datetime(s[0]), name="gsci").sort_index()


def index_match(panel: pd.DataFrame, index: pd.Series) -> dict:
    """Roll-scheduled front excess return vs a published crude excess-return index."""
    er = curve.held_excess_return(panel, curve.gsci_targets(panel))
    r_idx = index.pct_change()
    a, b = er.align(r_idx, join="inner")
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    te = (a - b).std() * np.sqrt(252)
    return {
        "days": int(ok.sum()),
        "start": str(a.index.min().date()),
        "end": str(a.index.max().date()),
        "correlation": float(a.corr(b)),
        "tracking_error_annual": float(te),
        "mean_diff_annual": float((a - b).mean() * 252),
        "cumulative_rebuilt": float((1 + a).prod() - 1),
        "cumulative_index": float((1 + b).prod() - 1),
    }


if __name__ == "__main__":
    main()
