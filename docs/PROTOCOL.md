# Evaluation protocol (fixed before any return model was fitted)

Written and committed before the return models existed. It is not edited after the
holdout is scored; deviations go in the README with a reason.

## Data and split

WTI front-month `CL=F`, daily closes, 2000-08-30 to 2024-05-30 (the repo's existing
Yahoo source, cached by `run.py`).

| segment | dates | use |
|---|---|---|
| train | up to 2014-12-31 | fit |
| validation | 2015-01-01 to 2018-12-31 | choose hyperparameters and the shrinkage weight `w` |
| holdout | 2019-01-01 to 2024-05-30 | scored once |

The holdout is scored exactly once, with the configuration frozen from validation. No
re-splitting or retuning after seeing it. The same walk-forward code scores validation
and holdout.

## Models

- **Persistence**: `P_hat(t+h) = P(t)`.
- **Random walk with drift**: `P(t) * exp(h * mu)`, `mu` = mean daily log return over
  everything known at the refit.
- **Ridge on lagged log returns**: target `log(P(t+h)/P(t))`, features the last `k`
  daily log returns, no intercept, `P_hat = P(t) * exp(r_hat)`. `r_hat = 0` is
  persistence exactly.
- **Combination**: `r_hat = w * r_ridge`, `w` in [0, 1] fit by least squares on the
  validation predictions only, then frozen.

Grid: `k` in {1, 5, 20} x ridge `alpha` in {0.01, 0.1, 1, 10, 100}, chosen by
validation return MSE separately for each horizon.

Prices at or below zero (2020-04-20) have no log return. Origins whose last price or
target is non-positive are skipped for every model alike, and missing lagged returns
are set to 0.

## Walk-forward

Expanding window, refit at the start of each calendar quarter on every sample whose
target was already realised at that date. Nothing is refit inside a quarter.

## Horizons

h = 1, 5, 20 trading days. For h > 1 only every h-th origin is scored, so targets do
not overlap. The Diebold-Mariano test additionally uses a Newey-West (HAC) variance.

## Metrics, per horizon

- RMSE and MAE in dollars, skill = `1 - RMSE_model / RMSE_persistence`.
- Diebold-Mariano test on squared dollar error vs persistence, Newey-West bandwidth
  `floor(n^(1/3))`, two-sided, normal reference. Positive statistic = model better.
- Directional accuracy (days the model calls a non-zero move) and a two-sided
  binomial test against 0.5.
- If a sign strategy is reported (h = 1): long/short by `sign(r_hat)`, net of 2bp per
  side on every position change plus 2bp per side to close and reopen on a contract
  roll day (CME rule dates, `roll.py`) while in a position.

## Trial count

Every configuration fitted is counted and reported: grid size x horizons for
selection, plus the baselines and the weight fit. The holdout is read once.

## Checks the evaluation must pass (in `tests/`)

- **Null**: the same pipeline run on returns that were randomly permuted must show
  skill <= 0 vs persistence.
- **Planted signal**: synthetic AR(1) returns with phi = 0.1 must show skill > 0.
