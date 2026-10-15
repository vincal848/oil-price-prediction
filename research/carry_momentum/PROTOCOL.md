# WTI carry-momentum: pre-registered protocol

Committed 2026-10-15, before any individual-contract data was downloaded. The commit
hash and GitHub timestamp are the record. Nothing below is edited after the first
result; any deviation is written in `DEVIATIONS.md` with its date and reason.

## The question

Does WTI carry-momentum (Bouchouev, *Virtual Barrels*, ch. 5; OIES *Energy
Quantamentals* 2024) earn a return that is (a) real after costs and multiple testing,
(b) not explained by published commodity factors, (c) explained by the storage-hedger
mechanism he proposes, and (d) tradable at a size worth running?

What has already been seen: the outright version on EIA contracts 2-4, 2008-01 to
2024-04 (`run_virtual_barrels.py`, Sharpe 0.61). Everything in this protocol is
reported separately for that **seen** window and for the **unseen** windows: 1990-01 to
2007-12, 2024-04 to the last settlement downloaded, and the live paper trade.

## Data (raw, first-hand)

1. Daily settlement, volume and open interest for every NYMEX CL contract month,
   1986 onward. Source in order of preference: WRDS (Datastream futures, if in CMU's
   subscription), CME DataMine, Databento. The source used goes in `DATA.md`.
2. EIA Weekly Petroleum Status Report crude stocks as first released (ALFRED vintages
   of WCESTUS1 and Cushing), not today's revised series.
3. CFTC disaggregated futures-only COT, raw annual files, NYMEX WTI (067651),
   including spreading positions.
4. Factors: AQR time-series momentum (commodities) and commodity carry, published by
   the authors; basis-momentum (Boons and Prado 2019) rebuilt from the same contract
   data by the published definition.
5. Reconciliation targets: EIA contract 1-4 series (RCLC1-4) and the S&P GSCI Crude
   Oil Excess Return index (or BCOM Crude Oil ER if GSCI history is unavailable).

## Series construction

- **Roll-scheduled continuous series:** hold the listed contract, roll evenly over
  business days 5-9 of the month before it becomes front (the GSCI roll window).
  Returns are computed contract by contract, never across a splice.
- **Constant-maturity curve:** log-linear interpolation in days to expiry between
  adjacent listed months, at 1, 2, 3, 6, 12 and 13 months.
- **Reconciliation pass marks** (failing either halts the study; it is not a result):
  rebuilt contract 1-4 settlements match EIA RCLC1-4 on 99% of days to the cent;
  daily excess return of the roll-scheduled series has correlation > 0.98 and
  tracking error < 2% a year against the GSCI Crude ER index.

## Signal (frozen)

- carry_t = ln(F_1m / F_13m) x (12 / 12): annualised slope from the 1-month to the
  13-month constant-maturity point, Bouchouev's seasonal-free definition. Positive in
  backwardation.
- signal_t = sign(carry_t - mean(carry over the last 21 trading days)).
- Computed from day t settlements.

## Instruments

- **H1 (primary, deployable): calendar spread.** Long the 2nd listed contract, short
  the 7th, in equal barrels when signal = +1; the reverse when -1. Rolled with the
  schedule above.
- **H2 (replication): outright.** Long or short the 2nd listed contract.
- Both sized to 15% annualised volatility from a 60-day EWMA, positions rounded to
  whole 1,000-barrel contracts on a $10m notional book.

## Execution and costs

- Signal at day t's settlement; position changed at day t+1's settlement (TAS), so it
  earns from t+1 to t+2. No trade is ever filled at the settlement it was computed
  from.
- Costs per contract per side: 1 tick ($10) half-spread for outrights and 1 tick for
  each calendar spread (exchange-listed spread), $1.50 exchange and clearing fees,
  plus a TAS premium of half a tick. Every roll is charged as a spread trade.
- Capacity: largest book for which the daily trade is under 5% of the traded
  contract's 20-day average volume and the position under 2% of its open interest.

## Tests

1. **Return decomposition:** daily P&L split into roll yield (convergence of the held
   contract to the next one) and spot move, by the standard
   excess return = spot return + roll yield identity on the rebuilt series.
2. **Mechanism (Bouchouev's storage hedgers):** weekly regression of the change in
   carry on the lagged change in producer/merchant net short, with Newey-West errors.
   His story predicts a negative coefficient (more storage hedging, deeper contango).
   And: strategy returns are higher in weeks when the signal agrees with the lagged
   hedger flow. Reported, not a kill criterion.
3. **Novelty:** monthly H1 and H2 returns regressed on the commodity TSMOM, carry and
   basis-momentum factors, OLS with Newey-West errors (lags = floor(4(T/100)^(2/9))),
   using `capstone.validation.spanning`.
4. **Robustness:** 90% stationary block-bootstrap confidence intervals for Sharpe
   (expected block 21 days, 10,000 draws); deflated Sharpe from
   `capstone.validation.sharpe.deflated_sharpe_ratio` with the trial count read from
   the ledger, which starts at 150 (every WTI configuration run in this repository
   before today); probability of backtest overfitting from `capstone.pbo.cscv` over
   the pre-declared grid: short MA in {1, 5}, long MA in {10, 21, 42, 63}, tenors
   {1m-13m, 2m-7m, 1m-4m}, so 24 variants, 16 blocks.

## Kill criteria (any one ends the case for deployment)

| | criterion | threshold |
|---|---|---|
| K1 | H1 net Sharpe on the unseen windows combined | below 0.20 |
| K2 | H1 90% bootstrap lower bound, full sample | below 0 |
| K3 | H1 deflated Sharpe probability | below 0.90 |
| K4 | PBO across the 24-variant grid | above 0.50 |
| K5 | H1 alpha t-statistic after the three factors | below 2.0 |
| K6 | H1 net Sharpe with costs doubled, full sample | below 0.20 |
| K7 | Capacity at the 5% / 2% limits | below $10m notional |

If H1 fails but H2 passes, the conclusion is that the edge is in the outright price,
not the curve, and the note says so.

## Live test

- The rules are frozen at a tagged commit before the paper trade starts.
- Every trading day for three months: signal from that day's settlement, order for
  the next day's TAS session in the IBKR paper account, fills logged.
- Expected: distribution of 63-day H1 returns from the stationary bootstrap of the
  full backtest.
- Fail: realised 3-month return below the 5th percentile of that distribution, or
  fills differing from settlement by more than 2 ticks on average.

## Stopping rule

The note reports every result above, pass or fail. No signal, tenor, window, cost or
threshold is changed after the first result is seen.
