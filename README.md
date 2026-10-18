# WTI Crude: From Price Forecasting to a Systematic Futures Book

[![tests](https://github.com/vincal848/oil-price-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/vincal848/oil-price-prediction/actions/workflows/tests.yml)

This project started as a neural-network forecast of the WTI crude price and ended as
a systematic trading book on WTI futures. Each stage answered a question the one
before it raised, so the README follows that path. The result at the end is a
three-strategy book that trades the shape of the futures curve.

## The result: a WTI curve strategy book

Three rules on the second-month NYMEX WTI contract, run as one risk-balanced book. It
trades whole contracts at the next day's settlement and is net of fees, ticks and
every roll, January 2003 to April 2026:

| | |
|---|---|
| Annual return / Sharpe | **8.0% / 0.56** (Sortino 0.74) at a 15% vol target |
| Growth of $1 | **$5.99** |
| Positive years | **17 of 24** |
| Correlation to WTI | **-0.07** |
| Oil crashes (2008, 2014-16, Q4 2018, 2020) | **+22%, +54%, +18%, +16%** |
| Worst stretch | -53% from 2022 to 2025 (-39% at a 10% vol target) |

![Equity curves](research/carry_momentum/results/book/equity.png)

- **Carry-momentum** (6.6% a year, Sharpe 0.49) goes long when backwardation is
  steepening and short when the curve is sliding toward contango.
- **Curve carry** (6.2%, 0.46) is long in backwardation and short in contango. It
  changes sides less than once a year.
- **Trend** (2.6%, 0.27) follows the 1, 3 and 12-month price direction.

Carry-momentum and curve carry are almost uncorrelated (0.03), which is why the
book beats each strategy on its own. The book wins big when oil moves violently and
loses slowly when oil grinds sideways or lower. The 2022-25 drawdown, two fixes
that were tried and failed, sizing, execution and capacity are all in
**[BOOK.md](research/carry_momentum/BOOK.md)**.

![Oil declines](research/carry_momentum/results/book/oil_drops.png)

## How it got here

### 1. Forecasting the price: nothing beats "tomorrow = today"

The original project compared an LSTM with an echo state network on the continuous
front-month series (`CL=F`). Measured against the naive forecast that tomorrow's
price equals today's, both lose: the LSTM's error is 50% larger and the reservoir's
158% larger. Neither calls the direction better than a coin flip. Two pre-registered
follow-ups score once on unseen data:

- ridge models on returns at 1, 5 and 20 days;
- the Brent-WTI spread, the spot basis and the dollar index as predictors.

Neither beats persistence. The contract roll was checked as an excuse and moves the
answer by about one percentage point.

**Takeaway:** the next-day price of WTI is close to unpredictable from public price
data, so the question has to change from *where will price go* to *what position
to hold*.
[Full write-up](docs/1_FORECASTING.md) · [methods](docs/METHODS.md) ·
[protocol 1](docs/PROTOCOL.md) · [protocol 2](docs/PROTOCOL-2.md)

### 2. Positions instead of forecasts: trend, carry and volatility targeting

The three rules behind most CTA books were run on WTI, Brent, RBOB and heating oil.
Two findings carried forward:

- **Carry is the strongest rule.** Measured as next contract vs front contract, WTI
  carry has a Sharpe of 0.49, and trend plus carry reaches 0.48 with a -31% drawdown.
  Daily returns have no usable autocorrelation, but their *size* does, which is the
  case for volatility targeting.
- **The curve matters more than the price.** Measured as front contract vs spot, the
  same carry rule loses money.

[Full write-up](docs/2_TREND_CARRY_VOL.md)

### 3. Virtual Barrels: one strategy per financial footprint

Following Ilia Bouchouev's *Virtual Barrels*, twelve strategies read the footprints
that financial players leave in oil: the curve, CFTC positioning, EIA inventories
and option volatility. They ran on EIA's four-contract WTI curve, 2008-2024.
Carry-momentum (0.61), the inventory cycle (0.60), curve carry (0.45) and following
managed-money flow (0.40) were the strongest. Twelve nearly uncorrelated strategies
made a 0.53 Sharpe book. **Carry-momentum stood out, so it got its own study.**
[Full write-up](docs/3_VIRTUAL_BARRELS.md)

### 4. Carry-momentum on raw contracts, then the book

EIA stopped publishing its contract series in 2024. So the curve was rebuilt from
all 547 individual NYMEX WTI contract months since 1983, from LSEG Datastream via
WRDS, and two checks confirmed it:

- contracts 1-4 match EIA's prices to the cent on 99.6% of days;
- the rolled series tracks the S&P GSCI Crude index at 0.9992 correlation.

**The formal study.** Carry-momentum went through a pre-registered study: rules
committed before any data, kill criteria, bootstrap, deflated Sharpe, PBO and a
factor-spanning test.
- **What worked:** the outright held up with a 0.44-0.48 Sharpe.
- **What didn't:** the calendar-spread version could not pay for its own turnover.
- **The mechanism:** the storage-hedger explanation was not in the positioning data.

[Study note](research/carry_momentum/NOTE.md) ·
[protocol](research/carry_momentum/PROTOCOL.md) ·
[deviations](research/carry_momentum/DEVIATIONS.md) ·
[data](research/carry_momentum/DATA.md)

**The book.** That study fed straight into the book at the top of this page. It
starts in 2003, after the thin pit-traded years. It keeps the outright, with a
rebalance buffer to cut costs, and adds curve carry and trend. Every configuration
run anywhere in this repository is logged in a trial ledger (254 so far).

## Next: more markets

The book's weak spot is a single market grinding sideways. The code to run the same
three strategies on Brent, heating oil, gasoline, natural gas and ICE gasoil is in
place (`markets.py`, `multi.py`). Rebuilding WTI through it reproduces the book
exactly. It is waiting on the contract data for the other markets.

## Repository map

| Stage | Code | Run |
|---|---|---|
| 1. Forecasting | `data.py`, `models.py`, `forecast.py`, `metrics.py`, `roll.py` | `python run.py` (`--holdout`, `--holdout2`, `--no-lstm`) |
| 2. Trend, carry, vol | `strategies.py` | `python run_strategies.py` |
| 3. Virtual Barrels | `virtual_barrels.py` | `python run_virtual_barrels.py` |
| 4. Curve and study | `research/carry_momentum/curve.py`, `reconcile.py`, `engine.py`, `study.py` | `python study.py` (needs the private capstone validation library) |
| 4. Book | `research/carry_momentum/book.py` | `python book.py` (about a minute) |
| Next | `research/carry_momentum/markets.py`, `multi.py` | `python multi.py` once data is in |

Results that the write-ups quote are tracked in `results/` and
`research/carry_momentum/results/`. Price data is cached locally and not committed.
The Datastream contracts are licensed through WRDS, so
[DATA.md](research/carry_momentum/DATA.md) says how to rebuild them. `legacy/` holds
the original script, annotated.

```bash
pip install -r requirements-dev.txt
pytest tests research/carry_momentum/tests -q   # 89 tests run without network
```

Two more smoke tests run the LSTM and the reservoir when TensorFlow and ReservoirPy
are installed. Everything is seeded, and every signal uses only data known at the
previous close.

Sources: Bouchouev, *Virtual Barrels: Quantitative Trading in the Oil Market*
(Springer, 2023) and his *Energy Quantamentals* comments for the Oxford Institute for
Energy Studies. Moskowitz, Ooi and Pedersen, "Time Series Momentum" (2012).
