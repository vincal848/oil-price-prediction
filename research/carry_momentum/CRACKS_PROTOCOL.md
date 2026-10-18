# Crack spreads, rebuilt the way products desks trade them

Written and committed before any result of this design was computed. The first
spread test (`spreads.py`, README Part 6) applied the outright book's machinery
to spreads mechanically: a rolling 2nd-month position, price-only signals, a 15%
volatility target, and legs that settle on different exchanges. This protocol
replaces those choices with the ones a NYMEX products desk would make.

## Instruments

All three spreads are NYMEX, settle at the same time and trade as listed spreads.

| Spread | Legs per unit |
|---|---|
| Heating oil crack | +1 heating oil (ULSD), −1 WTI |
| RBOB crack | +1 RBOB, −1 WTI |
| 3-2-1 crack | +2 RBOB, +1 heating oil, −3 WTI |

Every leg holds the same delivery month. The sample is 2006-01-03 to 2026-04-02,
so RBOB is used throughout and the unleaded gasoline splice drops out. The diesel
arb, the gasoil crack and Brent-WTI are not part of this test: the first lacks
freight data and the other two have legs that settle at different times.

## Strategies

1. **Seasonal strips.** Long only, on a fixed calendar every year.
   - Heating oil crack: long the December-February delivery strip (a third in
     each month) from the first to the last business day of August-October.
   - RBOB crack: long the May-August delivery strip (a quarter in each month)
     from the first to the last business day of January-March.
   - 3-2-1 crack: both trades, each in its own window.
2. **Inventories.** The prompt crack (2nd listed WTI month and the same month in
   the product legs, rolled on business days 5-9) positioned on US product stocks
   against their seasonal norm.
   - Norm: the same week's average over the previous five years (at least three).
   - z = (stocks - norm) / the trailing 52-week standard deviation of that gap.
   - Long when z < -0.5 (tight stocks), short when z > +0.5, flat in between.
   - Heating oil crack uses distillate stocks, RBOB uses total motor gasoline, the
     3-2-1 uses two thirds gasoline z plus one third distillate z.
   - Timing: EIA reports stocks for the week ending Friday on the following
     Wednesday at 10:30 ET. Each report is first used at the settlement six days
     after the week ends (Thursday), a day after release, to cover holiday delays.
3. **Combined.** The two strategies above, equally weighted, per spread.

## Sizing, execution and costs

- Risk budget set once a year: on the first business day of each year, units are
  set so that the prompt crack's trailing three-year volatility is 10% a year on
  $10M, and held for the year. No daily volatility targeting.
- Trades at the next day's settlement, in whole units.
- Listed-spread costs: $0.01/bbl on the crude barrels plus $1.50 per lot per
  trade. Rolls in the inventory strategy pay each leg's calendar-spread cost.
- Capacity: 5% of 20-day volume and 2% of open interest per leg, since 2010.

## What will be reported

For each spread and strategy: annual return, Sharpe, maximum drawdown, cost, and
capacity, plus the P&L of every seasonal trade by year and its hit rate. Then a
crack book (all nine, equal weight) and that book combined 50/50 with the core
outright book (WTI, Brent, natural gas), each with their 2022-25 return and
correlation to the core.

Nine strategy configurations and two books are logged to the trial ledger before
they run. No parameter is searched. The only inputs not in the repository before
this protocol are EIA weekly US distillate and total motor gasoline stocks
(WDISTUS1, WGTSTUS1), recovered from the Part 3 pull.

Not modelled: RIN prices, refinery utilisation and outages, export data. A
result here says what product stocks and the calendar alone are worth.
