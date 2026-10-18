"""Contract specs and panels for every market in the multi-market book.

Each market is a family of Datastream futures mnemonics PREFIX+MMYY (delivery
month MM, year YY), pulled from WRDS like WTI (DATA.md). Expiries follow the
exchange rule where it is simple (NYMEX), and Datastream's LastTrdDate for ICE
contracts, whose rules changed during the sample; `check_expiries` compares the
two against the last day each contract actually settled.
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, field

import curve
import engine
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from roll import BDAY


def _delivery(contract: str) -> tuple[int, int]:
    mm, yy = int(contract[-4:-2]), int(contract[-2:])
    return (1900 + yy if yy >= 80 else 2000 + yy), mm


def wti_expiry(contract: str) -> pd.Timestamp:
    return curve.rule_expiry("NCL" + contract[-4:])


def last_bd_prior_month(contract: str) -> pd.Timestamp:
    """NYMEX heating oil, RBOB and unleaded gasoline: last business day of the
    month before delivery."""
    y, m = _delivery(contract)
    first = pd.Timestamp(year=y, month=m, day=1)
    return (first - BDAY).normalize()


def ng_expiry(contract: str) -> pd.Timestamp:
    """NYMEX Henry Hub natural gas: three business days before the first calendar
    day of the delivery month."""
    y, m = _delivery(contract)
    return (pd.Timestamp(year=y, month=m, day=1) - 3 * BDAY).normalize()


@dataclass(frozen=True)
class Market:
    name: str
    prefixes: tuple[str, ...]  # Datastream mnemonic roots; later ones take over
    multiplier: float  # price units per contract (bbl, gal, MMBtu, t)
    tick: float  # $ value of one minimum tick per contract
    expiry: object = None  # callable(contract) -> Timestamp, or None = Datastream
    switch: dict = field(default_factory=dict)  # prefix -> first date it is used
    tier: str = (
        "core"  # "core" markets set the headline book, "downstream" is reported apart
    )
    complex: str = "crude"  # "crude" or "gas", the primary each product comes from

    @property
    def key(self) -> str:
        return self.name.lower().replace(" ", "_")

    @property
    def costs(self) -> engine.Costs:
        return engine.Costs.for_tick(self.tick)


MARKETS = {
    m.key: m
    for m in [
        # Core: the primaries. These set the headline book.
        Market("WTI", ("NCL",), 1000, 10.0, wti_expiry),
        Market("Brent", ("LLC",), 1000, 10.0, None),
        Market("Natural gas", ("NNG",), 10000, 10.0, ng_expiry, complex="gas"),
        # Downstream: refined products and gas liquids, reported as their own layer.
        Market(
            "Heating oil",
            ("NHO",),
            42000,
            4.20,
            last_bd_prior_month,
            tier="downstream",
        ),
        Market(
            "Gasoline",
            ("NHU", "NRB"),  # unleaded gasoline, replaced by RBOB in 2006
            42000,
            4.20,
            last_bd_prior_month,
            switch={"NRB": "2006-01-01"},
            tier="downstream",
        ),
        Market("Gasoil", ("LLE",), 100, 25.0, None, tier="downstream"),
    ]
}


def raw_files(m: Market) -> list[str]:
    out = []
    for p in m.prefixes:
        out += sorted(glob.glob(os.path.join(DATA, f"{p.lower()}_*.csv.gz")))
    return out


def load_raw(m: Market) -> pd.DataFrame:
    d = pd.concat(
        [pd.read_csv(f, compression="gzip") for f in raw_files(m)], ignore_index=True
    )
    d.columns = [c.lower() for c in d.columns]
    d = d.rename(
        columns={
            "date_": "date",
            "settlement": "settle",
            "openinterest": "oi",
            "dsmnem": "contract",
            "lasttrddate": "expiry_ds",
        }
    )
    d["date"] = pd.to_datetime(d["date"])
    d["expiry_ds"] = pd.to_datetime(d["expiry_ds"], errors="coerce")
    return d


def panel(m: Market) -> pd.DataFrame:
    """Tidy contract panel (date, contract, expiry, expiry_ds, settle, volume, oi),
    cached in data/<key>_contracts.parquet."""
    path = os.path.join(DATA, f"{m.key}_contracts.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    d = load_raw(m)
    d = d[d.contract.str[:3].isin(m.prefixes)]
    if m.expiry is None:
        d["expiry"] = d.expiry_ds
    else:
        rule = {c: m.expiry(c) for c in d.contract.unique()}
        d["expiry"] = d.contract.map(rule)
    d = d.dropna(subset=["settle", "expiry"])
    d = d[d.date <= d.expiry]
    # Product switch (unleaded gasoline -> RBOB): each date uses one product only.
    for prefix, start in m.switch.items():
        new = d.contract.str.startswith(prefix)
        d = d[(new & (d.date >= start)) | (~new & (d.date < start))]
    d = d.drop_duplicates(["date", "contract"], keep="last")
    # Stale holiday rows (DEVIATIONS #8): dates with fewer than five contracts.
    n = d.groupby("date").contract.transform("count")
    d = d[n >= 5]
    cols = ["date", "contract", "expiry", "expiry_ds", "settle", "volume", "oi"]
    d = d[cols].sort_values(["date", "expiry"]).reset_index(drop=True)
    d.to_parquet(path)
    return d


def check_expiries(p: pd.DataFrame) -> dict:
    """How the expiry used compares with Datastream and with the last settlement."""
    last = p.groupby("contract").date.max()
    exp = p.groupby("contract").expiry.first()
    ds = p.groupby("contract").expiry_ds.first()
    done = exp < p.date.max()
    gap_last = (exp[done] - last[done]).dt.days
    return {
        "contracts": len(exp),
        "rule_vs_datastream_diff_days": (exp - ds)
        .dt.days.value_counts()
        .head(5)
        .to_dict(),
        "expiry_minus_last_settle_days": gap_last.value_counts().head(5).to_dict(),
        "contracts_ending_over_5d_early": int((gap_last > 5).sum()),
    }


def inputs(m: Market, cache: bool = True) -> dict:
    """Everything the engine needs for one market, like study.build_inputs."""
    path = os.path.join(DATA, f"{m.key}_cm.parquet")
    p = panel(m)
    if cache and os.path.exists(path):
        cm = pd.read_parquet(path)
    else:
        cm = curve.constant_maturity(p)
        cm.to_parquet(path)
    return {
        "market": m,
        "panel": p,
        "cm": cm,
        "px": p.pivot(index="date", columns="contract", values="settle"),
        "vol": p.pivot(index="date", columns="contract", values="volume"),
        "oi": p.pivot(index="date", columns="contract", values="oi"),
        "w2": engine.leg_weights(p, 1),
    }
