# Part 2. Trend, carry and volatility targeting on four energy futures

[Back to the project README](../README.md)

[Part 1](1_FORECASTING.md) asked what tomorrow's price will be, and the answer was today's price.
Systematic futures traders mostly ask a different question: what position to hold,
given what the market is already showing. That does not need a next-day forecast. So I
tried the three rules that make up most of a CTA book, on WTI, Brent, RBOB and heating
oil from 2001 to October 2026, monthly rebalance, net of 2bp per side plus a
close-and-reopen on every contract roll:

- **Trend:** long or short on the sign of the trailing 1-, 3- or 12-month return
  (time-series momentum, Moskowitz, Ooi and Pedersen 2012).
- **Carry:** long in backwardation, short in contango.
- **Volatility targeting:** scale any position to 20% annualised ex-ante volatility
  (EWMA, centre of mass 60 days).

WTI, 2001-08 to 2026-10, all vol-targeted except the first row:

| rule | Sharpe | ann. return | max drawdown |
|---|---|---|---|
| buy and hold | 0.34 | 13.7% (40% vol) | -94% |
| buy and hold, vol-targeted | 0.26 | 5.3% | -72% |
| trend 1m | 0.35 | 7.1% | -41% |
| trend 12m | 0.21 | 4.2% | -66% |
| carry, front vs spot | -0.22 | -4.5% | -92% |
| **carry, next contract vs front** | **0.49** | **9.9%** | **-41%** |
| trend 12m + carry | 0.48 | 7.2% (15% vol) | -31% |

![WTI equity curves](img/strat_equity.png)

What I learned from it:

**You can forecast the size of tomorrow's move, not its direction.** The daily return
autocorrelation is about -0.02 at every lag, on all four contracts. The autocorrelation
of the absolute return is 0.17 to 0.25 at one day and still 0.13 to 0.16 at a month.
That is the whole case for volatility targeting, and the reason a price-level model had
nothing to find.

**Carry works, but only once it is measured properly.** My first version used the
futures-minus-spot basis, the same feature that failed in protocol 2, and it lost money
on every contract. The front month sits almost on top of spot for most of its life, so
that basis is mostly noise. The slope that matters is next contract vs front, and a
continuous front-month series does contain it: on each roll day the futures-minus-spot
basis jumps by roughly log(next/front), because the series switches contracts while
spot does not. Read that way, WTI carry has a Sharpe of 0.49, is positive in every
regime I split out, and its best year is 2020, when it was short through the super
contango. It is also negatively correlated with buy and hold (-0.21).

![Carry](img/strat_carry.png)

**Single-contract trend is weak and unstable.** 12-month trend on WTI is 0.21, and the
lookback grid has no smooth ridge: neighbouring lookbacks swing from 0.38 to -0.07. The
Sharpe that the trend literature reports comes from diversifying across dozens of
futures, not from any one of them. Where trend does earn is the big directional
regimes, 2014-2016 especially (0.95), which is the "crisis alpha" pattern.

![Lookbacks](img/strat_lookbacks.png)

**Volatility targeting halves the drawdown but does not raise the Sharpe of a long
position in oil.** It cuts exposure after a volatility spike, and in oil the spike comes
with the crash and with the rebound. In 2020 it was underweight for the recovery. On
trend and carry, whose returns do not depend on the market going up, it is what keeps
the position sizes comparable through time.

**Combining the two lowers risk more than it adds return.** Trend 12m and carry are
correlated 0.11, so the blend runs at 15% volatility against a 20% target, with a
-31% drawdown against -94% for buy and hold.

**Gasoline is different.** RBOB's curve is seasonal, because the summer and winter
specifications differ, so its contango and backwardation mostly reflect the calendar
rather than inventories. Carry is flat there (-0.04) and 12-month trend loses (-0.18).
A seasonally adjusted carry signal would be the next thing to try.

Notes:

- Rolls: the WTI splice days come from the CME rule in `roll.py`. RBOB and heating oil
  splice on the first trading day of each month (they expire on the last business day
  of the month before delivery); Brent follows ICE's two expiry rules, before and after
  2016. Each calendar is checked against the data: the futures-minus-spot basis moves
  2.1-3.0x more on its splice days than on other days for WTI, RBOB and heating oil.
  For Brent it is only 1.3x, because FRED's Brent spot is Dated Brent rather than the
  ICE future's underlying, so the Brent rows (in `results/strategies.json`) are rough.
- Returns on splice days are set to zero, since part of that day's change is the
  spread between two contracts, not a return. 2020-04-20, when the May WTI contract
  settled negative, is also set to zero.
- Sharpe ratios over 25 years carry a standard error of roughly 0.2, and I ran about a
  dozen rules on four contracts, so differences of 0.1-0.2 between rows are noise. The
  gap between the two carry measures (0.7) is not.
- Data: Yahoo continuous front-month closes (`CL=F`, `BZ=F`, `RB=F`, `HO=F`) and FRED
  spot prices (`DCOILWTICO`, `DCOILBRENTEU`, `DGASNYH`, `DHOILNYH`), cached in
  `data/futures/`. Tests in `tests/test_strategies.py` check that no rule sees the
  future, that trend finds planted regimes and finds nothing in a random walk, and
  that the splice spread reads a known contract spread.

Reproduce with `python run_strategies.py`.

