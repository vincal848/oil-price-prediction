# Protocol 2: public price/spread information vs persistence (declared before download)

Written and committed before any data after 2024-05-30 was downloaded or looked at. Not
edited afterwards; deviations go in the README with a reason.

The 2019-01-02 to 2024-05-30 holdout of [PROTOCOL.md](PROTOCOL.md) is spent (ridge on
own lagged returns, no skill). This protocol asks one question on fresh data: does
information persistence does not have predict the next WTI front-month return?

## Windows

| segment | dates | use |
|---|---|---|
| fit | up to 2018-12-31 | initial fit, feature standardisation constants |
| validation | 2019-01-01 to 2024-05-31 | choose the shrinkage weight `w` (already seen; not a holdout) |
| new holdout | 2024-06-01 to the latest date the downloads return | scored once |

"Latest" is whatever Yahoo and FRED return at fetch time (the shortest of the series);
the end date is recorded in `results/holdout2.json`. Expanding window, refit at the
start of each calendar quarter through validation and the new holdout. h = 1 only.

## Data (keyless, public)

- WTI front-month `CL=F` (Yahoo, the repo's source). Cached history to 2024-05-30 is
  kept as is; only rows after it are downloaded.
- FRED keyless CSV `fredgraph.csv?id=`: `DCOILBRENTEU` (Brent spot), `DCOILWTICO` (WTI
  Cushing spot), `DTWEXBGS` (broad dollar index).
- Every exogenous feature is lagged one trading day (FRED values are published after
  the close), forward-filled at most 5 days, standardised with the fit-segment std,
  missing set to 0.

## Features (chosen from economics, not from 2019-2024 residuals)

- `spread`: log(Brent/WTI spot) minus its trailing 252-day mean. Transatlantic
  arbitrage pulls WTI toward Brent when the spread is stretched.
- `basis`: log(front future / WTI spot). Carry / term-structure proxy: front above spot
  is contango (storage-constrained).
- `usd`: 5-day log change of the dollar index. Oil is dollar-denominated.
- `ret1`: the previous daily log return.

## Candidates (6, fixed; the whole budget)

| id | features |
|---|---|
| C1 | spread |
| C2 | basis |
| C3 | usd |
| C4 | spread + basis |
| C5 | spread + basis + usd |
| C6 | spread + basis + usd + ret1 |

Each is a no-intercept ridge on the next-day log return, alpha = 1000 fixed (strong
shrinkage; features are standardised), forecast r_hat = w * ridge with w in [0, 1] fit
by least squares on validation only. Price forecast = last * exp(r_hat). No other
configuration will be tried; alpha and the lags are not tuned.

## Baselines and tests

Persistence and random walk with drift (as in PROTOCOL.md). Per candidate: RMSE, MAE,
skill vs persistence, Diebold-Mariano (Newey-West) p-value on squared dollar error,
directional accuracy and two-sided binomial p, and the h = 1 sign strategy net of 2bp
per side plus the roll charge.

## Multiple testing

Bonferroni across the 6 candidates: a candidate is called significant only if its DM
p < 0.05 / 6 = 0.0083 and its skill is positive. Directional binomial p-values are held
to the same threshold. The count of configurations is 6.

## Checks that must pass first (in `tests/`)

- Null: permuted returns with a noise exogenous feature give skill <= 0.
- Planted: returns generated from a lagged exogenous feature give skill > 0.
- The exogenous feature builder lags by one day (no same-day information).

## Stopping rule

If no candidate is significant after Bonferroni, the README says so plainly and the
work stops: that is the evidence that one-step WTI is efficient against this public
price/spread information.
