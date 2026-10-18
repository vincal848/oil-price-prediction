"""Contract specs for the multi-market book."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine
import markets


def test_tick_costs_reproduce_wti_protocol_costs():
    assert engine.Costs.for_tick(engine.TICK) == engine.Costs()


def test_nymex_product_and_gas_expiry_rules():
    # Published CME last trading days for January 2024 delivery.
    assert markets.last_bd_prior_month("NHO0124") == pd.Timestamp("2023-12-29")
    assert markets.ng_expiry("NNG0124") == pd.Timestamp("2023-12-27")
    # Natural gas counts back across Thanksgiving: Dec 2023 contract, 2023-11-28.
    assert markets.ng_expiry("NNG1223") == pd.Timestamp("2023-11-28")


def test_wti_expiry_matches_study_rule():
    assert markets.wti_expiry("NCL0124") == pd.Timestamp("2023-12-19")


def test_dollar_pnl_scales_with_contract_multiplier():
    idx = pd.bdate_range("2024-01-01", periods=3)
    px = pd.DataFrame({"A": [1.0, 1.1, 1.3]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    a = engine.unit_dollar_pnl(px, w, multiplier=1000)
    b = engine.unit_dollar_pnl(px, w, multiplier=42000)
    assert (b - a * 42).abs().max() < 1e-9
