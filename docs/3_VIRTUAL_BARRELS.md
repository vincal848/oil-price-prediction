# Part 3. Virtual Barrels: one strategy per financial footprint in oil

[Back to the project README](../README.md)

The rules in [Part 2](2_TREND_CARRY_VOL.md) use only prices. Ilia Bouchouev's *Virtual Barrels* (2023) argues
that oil is now priced as much by financial barrels as physical ones: storage
hedgers, trend followers, slow inflation-hedging money and option sellers each
leave a footprint in the curve, the positioning data or the volatility surface. I
built one strategy family per footprint, from the book's chapter structure and the
rules he spells out in his public *Energy Quantamentals* comments for the Oxford
Institute for Energy Studies.

That needed more data than a front-month price: EIA's daily Cushing futures
contracts 1-4 (the curve, 2000 to April 2024, when EIA stopped publishing it), EIA
weekly crude stocks, CFTC disaggregated positions for NYMEX WTI, and the CBOE OVX
index. With four contracts the roll is exact rather than zeroed out: on the day after
an expiry, yesterday's second contract is today's first, so the held return is
C1(t)/C2(t-1).

WTI, 2008 to April 2024, every strategy scaled to 15% volatility, net of costs:

| family | rule | Sharpe | max drawdown |
|---|---|---|---|
| benchmark | buy and hold (contract 2, rolled) | -0.12 | -71% |
| curve | carry: long backwardation, short contango | 0.45 | -28% |
| curve | **carry-momentum**: carry vs its 1-month average | **0.61** | -28% |
| price | momentum M(1,20) | 0.15 | -48% |
| price | multi-horizon M(1,5)+M(5,20)+M(20,250) | 0.20 | -41% |
| price | M(1,20) sized by the reaction function | 0.25 | -26% |
| positioning | **follow the flow** (managed money MA crossover) | **0.40** | -29% |
| positioning | fade the crowd (range extremes) | -0.04 | -21% |
| positioning | hedging pressure (Keynes premium) | -0.28 | -69% |
| inventories | **inventory cycle** (stocks vs 5-year seasonal norm) | **0.60** | -38% |
| inventories | squeeze spread: C2-C3 on Cushing fill | 0.16 | -28% |
| quantamental | WTI-Brent convergence | -0.49 | -72% |
| volatility | short one-month variance at OVX | 0.42 | -80% |
| book | all twelve, equal risk | 0.53 | -32% |
| book | simple: M(1,20) + carry | 0.40 | -44% |
| book | core four (bold rows, chosen after seeing this table) | 0.90 | -25% |

![Strategy families](img/vb_families.png)

**The curve carries most of the information, and its change carries more than its
level.** Carry-momentum buys when backwardation is steepening or contango flattening.
Bouchouev reports Sharpe 0.48 for carry and 0.63 for carry-momentum on WTI over
1994-2023; I get 0.45 and 0.61 on a different window, and every one of seven
lookback pairs I tried is between 0.51 and 0.70. Inventory flows show up in the
curve before they show up in the outright price, which is why trend followers
watch it.

**Price momentum on its own is the weakest of the systematic rules (0.15).** The
reaction function, which builds the position up to one standard deviation of the
signal and then cuts it, does not change the Sharpe much (0.25) but halves the
drawdown, because it stops adding to a move that is already stretched.

**In positioning, the information is in the flow, not the level.** Following the
managed-money crossover earns 0.40, and all sixteen parameter pairs are positive.
Fading the crowd at range extremes is a coin flip across its twenty parameter sets,
and Bouchouev finds the same until about 2022. The classic hedging-pressure premium
(long when producers and swap dealers are unusually net short) loses: in WTI those
shorts are mostly swap dealers hedging shale producers' OTC trades, and their level
moves slowly against a market the funds are already trading.

**Inventories work through the seasonal surplus, not the level.** Going long when
US crude stocks are drawing relative to the five-year same-week average earns 0.60,
and is positive in every period except 2020, when stocks built faster than ever and
prices recovered anyway. The squeeze-model spread trade on Cushing (short the C2-C3
spread when Cushing fills near its recent maximum, long when it drains toward
empty) is weak at 0.16: the extremes it is built for, like April 2020, are rare.

**WTI-Brent does not mean-revert on a three-month horizon.** The spread trended
through every structural change in the period: the Cushing glut of 2011-13, the end
of the US export ban in 2015, the pipeline build-out after it.

**The volatility premium is real but dangerous.** OVX averaged 39 against 35%
realised front-month volatility. Sold as one-month variance swaps, marked to market
every day, the premium earns 0.42 with an -80% drawdown, nearly all of it in 2020.
The first version of this (Sharpe 1.22) booked only the daily accrual, which hides
the swings in implied volatility, so the risk looked half as large and the sizing
ran at the leverage cap; an independent review of the code caught it.

**The book is better than its parts because the parts are nearly uncorrelated.**
The average pairwise correlation across the twelve strategies is 0.07; carry,
carry-momentum, the inventory cycle and following the flow are correlated 0.10 to
0.22 with each other. An equal-risk book of all twelve has a Sharpe of 0.53 and a
-32% drawdown against -71% for buy and hold. Keeping only the four strong families
gives 0.90 and is positive in every period, but that book was chosen by looking at
this table, so 0.90 is an upper bound, not an estimate.

![Correlations](img/vb_correlation.png)

Notes:

- Timing: a signal from data known at the close of day t sets the position for day
  t+1. COT reports (as of Tuesday) are released Friday at 3:30pm ET, after the oil
  settlement, so they are first used at Monday's close, Tuesday's in holiday weeks,
  and after the catch-up in the 2013 and 2018-19 shutdowns. EIA weekly stocks are
  first used at Thursday's close. A truncation test (rerun everything with all data
  cut at a date, compare every strategy's returns before that date) finds no
  differences at three cut dates.
- Costs: 2bp per side on outright trades plus a full roll each month, 4bp on spreads,
  1 vol point of bid-ask per variance swap, and 2bp on each monthly re-sizing of
  the books.
- Limits: OVX is implied volatility on USO options, a proxy for options on the WTI
  future, and a variance swap at OVX is not something you can trade at that level.
  Cushing utilisation uses stocks over their trailing three-year maximum, because
  EIA's capacity survey is semi-annual and capacity roughly doubled after 2010. The
  ensembles average each rule over its parameter grid rather than drawing parameters
  at random as Bouchouev does, which is the deterministic version of the same idea.
- Tests in `tests/test_virtual_barrels.py`: the roll mapping, carry sign and scale,
  the reaction function's peak, COT and EIA release timing, Good Friday rolls, that
  a variance swap settles to strike minus realised and marks implied-vol moves, that
  moving P&L between calendars loses none of it, and that carry and carry-momentum
  find nothing on a random-walk curve.

Sources: Bouchouev, *Virtual Barrels: Quantitative Trading in the Oil Market*
(Springer, 2023); *Energy Quantamentals* comments, Oxford Institute for Energy
Studies: "Myths and Realities about CTAs" (2024), "Who is who in Financial
Barrels" (2024), "The Revival of the Volatility Risk Premium" (2025), "A Tale of Two
Algorithms and OPEC+" (2025), "To Follow the Crowd or to Fade the Extremes?" (2025);
"A Lesson from Physics on Oil Prices" (OIES, 2021).

Reproduce with `python run_virtual_barrels.py` (data in `data/vb/`).

