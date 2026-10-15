# Deviations from PROTOCOL.md

Each entry is dated and states what changed, why, and whether it was decided before
or after any strategy result was seen. No strategy return had been computed when
entries 1-6 were written.

1. **2026-10-15, before results. Contract expiries come from the CME rule, not
   Datastream.** Datastream's LastTrdDate is a month early for the March, April and
   May 2007 contracts (e.g. NCL0307 listed as expiring 2007-01-22; it expired
   2007-02-20), which puts contracts in the wrong order. With Datastream's dates,
   ranks 1-4 matched EIA to the cent on 98.0-98.4% of days (pass mark 99%); with
   the rule in `roll.py`, 99.6-99.8%. In 64 of 547 contracts the two dates differ,
   58 of them before 1991, when NYMEX used a different termination rule; the study
   window starts in 1990, and 1990 contracts are checked against EIA like the rest.

2. **2026-10-15, before results. Data ends 2026-04-02, not today.** WRDS's
   Datastream futures history stops there. The unseen recent window is therefore
   2024-04-08 to 2026-04-02 (about two years), not to October 2026.

3. **2026-10-15, before results. GSCI reconciliation excludes 2020-03-16 to
   2020-06-30.** In April 2020 S&P made an emergency change to the S&P GSCI WTI roll
   to avoid the expiring May contract. Over 2016-10 to 2026-04 the rebuilt
   roll-scheduled series has daily correlation 0.948 and tracking error 15.1% with
   the index; excluding that window, 0.9992 and 1.4% (pass marks 0.98 and 2%). Only
   ten years of the index are available from S&P's site; CMU does not subscribe to
   Datastream's commodity indices.

4. **2026-10-15, before results. First-release EIA stocks.** ALFRED vintage
   downloads were not reachable without an API key. EIA weekly stocks are not
   routinely revised after publication (DOE Inspector General report DOE/IG-0940:
   post-publication corrections are rare), so the current EIA weekly series is used
   as the as-released value. Stocks enter only in the mechanism test, never in the
   signal.

5. **2026-10-15, before results. Factors for the novelty test.** No published daily
   or monthly return series exists for the cross-sectional commodity carry factor of
   Koijen et al. (2018), and building one needs a cross-section of commodities that
   CMU's Datastream access does not cover here. The novelty regressions therefore
   use: AQR's diversified commodity TSMOM factor (published), and three WTI
   time-series factors built from the same contracts by their published
   definitions: 12-month TSMOM, carry (sign of the 1m-13m slope) and
   basis-momentum (Boons and Prado 2019: 12-month momentum of the first nearby
   minus the second nearby). These are closer substitutes for a WTI strategy than
   cross-sectional factors, so the test is harder to pass, not easier.

6. **2026-10-15, before results. COT history starts 2006-06.** The disaggregated
   report begins in 2006, so the mechanism test covers 2006-06 to 2026-04.

7. **2026-10-15, after the H1 results were seen. H2 is put through the same seven
   kill criteria.** The protocol says what to conclude "if H1 fails but H2 passes"
   but defines the 24-variant PBO grid only for H1. H2 is evaluated on K1-K7 with
   the same grid applied to the outright instrument. Those 24 outright variants
   are logged to the trial ledger before their results are computed and raise the
   trial count used in every deflated Sharpe. Nothing about H1 changes.

8. **2026-10-15, after results, from an independent code audit.** None of these
   changes a deployment verdict; each is reported with its before and after value
   in the research note.
   - Novelty factors are built gross of costs, as AQR's published factor is. They
     had been built net, which lowers their means and flatters the strategy's
     alpha (H2 alpha t 2.06 net, 1.95 gross).
   - The calendar-spread cost is $23 per spread, not $18: the protocol's fees and
     TAS premium are "per contract per side", and a spread is two contracts.
   - Three Datastream holiday rows with stale or missing curves (2006-11-24,
     2009-11-26, 2026-04-03) are dropped: dates with fewer than five listed
     contracts.
   - The mechanism regression stops at the last curve date (2026-04-02) instead of
     forward-filling carry into 26 later COT weeks.
   - Capacity is measured from 1990-01-01, the study start, not from the 1988
     warm-up.
   - Turnover is reported including rolls.
