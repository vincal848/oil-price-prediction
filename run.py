"""Run the comparison and write results/ and docs/img/.

    python run.py                # full run, window=120
    python run.py --window 60    # the memory length originally tried
    python run.py --no-lstm      # skip TensorFlow, for a quick check
    python run.py --holdout      # score the return models once (docs/PROTOCOL.md)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import forecast
import metrics
import roll
from data import aligned_targets, make_sequences, split_and_scale
from models import fit_esn, fit_lstm, fit_naive

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
IMG = os.path.join(HERE, "docs", "img")

TICKER = "CL=F"
START = "2000-08-30"
END = "2024-05-31"
CACHE = os.path.join(HERE, "data", "wti.csv")

WINDOW = 120
TRAIN_FRAC = 0.75


def load_prices(start: str = START, end: str = END, use_cache: bool = True) -> pd.Series:
    """WTI front-month closes as a Series indexed by date, cached to CSV.

    Cached so re-runs are offline and reproducible even if Yahoo revises history.
    This is the only network/file access for prices; data.py stays pure.
    """
    if use_cache and os.path.exists(CACHE):
        s = pd.read_csv(CACHE, index_col=0, parse_dates=True).iloc[:, 0]
        return s.loc[start:end]

    import yfinance as yf

    df = yf.download(TICKER, start=start, end=end, progress=False, auto_adjust=False)
    close = df["Close"]
    # Recent yfinance returns MultiIndex columns, so this can be a one-column frame.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.name = "close"

    if use_cache:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        close.to_csv(CACHE)
    return close


def run(
    window: int = WINDOW, train_frac: float = TRAIN_FRAC, include_lstm: bool = True
) -> tuple[dict, dict]:
    """Fit every model and return (results, context) for reporting."""
    prices = load_prices()
    train_scaled, val_scaled, scaler, cut = split_and_scale(prices, train_frac)

    actual = aligned_targets(val_scaled, window, scaler)
    # The last observed price before each target, for directional accuracy and for
    # the naive forecast. Taken from the input windows so it cannot drift out of
    # alignment with the targets.
    X_val, _ = make_sequences(val_scaled, window)
    previous = scaler.inverse_transform(X_val[:, -1].reshape(-1, 1)).reshape(-1)

    train_prices = np.asarray(prices, dtype=float)[:cut]
    dates = prices.index[cut + window:]

    fitted = {"naive": fit_naive(val_scaled, window, scaler),
              "esn": fit_esn(train_scaled, val_scaled, window, scaler)}
    if include_lstm:
        fitted["lstm"] = fit_lstm(train_scaled, val_scaled, window, scaler)

    # Days where the change spans two contracts rather than one market move. Scored
    # separately so the roll's effect on the comparison is measured, not assumed.
    is_roll = roll.roll_mask(dates).to_numpy()

    results = {}
    for name, (predicted, elapsed) in fitted.items():
        results[name] = metrics.evaluate(actual, predicted, train_prices, previous)
        results[name]["seconds"] = elapsed
        results[name]["predicted"] = predicted
        ex = metrics.evaluate(actual[~is_roll], predicted[~is_roll],
                              train_prices, previous[~is_roll])
        results[name]["rmse_ex_roll"] = ex["rmse"]
        results[name]["mae_ex_roll"] = ex["mae"]

    for name, row in results.items():
        if name != "naive":
            row["skill_ex_roll_rmse"] = metrics.skill_score(
                {"rmse": row["rmse_ex_roll"]}, {"rmse": results["naive"]["rmse_ex_roll"]}
            )
            row["skill_vs_naive_rmse"] = metrics.skill_score(row, results["naive"], "rmse")
            row["skill_vs_naive_mae"] = metrics.skill_score(row, results["naive"], "mae")

    context = {
        "window": window,
        "train_frac": train_frac,
        "n_train": int(cut),
        "n_eval": len(actual),
        "eval_start": str(dates[0].date()),
        "eval_end": str(dates[-1].date()),
        "actual": actual,
        "previous": previous,
        "dates": dates,
        "is_roll": is_roll,
        "n_roll_days_in_eval": int(is_roll.sum()),
        "roll_diagnostics": roll.roll_diagnostics(prices),
        "scaler_min_train_only": float(scaler.data_min_[0]),
        "price_min_overall": float(np.min(prices)),
        "price_min_date": str(prices.idxmin().date()),
    }
    return results, context


def print_table(results: dict, context: dict) -> None:
    cols = ["rmse", "rmse_ex_roll", "mae", "mae_ex_roll", "mase",
            "directional_accuracy", "seconds"]
    print(f"\nEvaluation window: {context['eval_start']} to {context['eval_end']}  "
          f"({context['n_eval']} days, memory {context['window']})")
    print(f"{'model':<8}" + "".join(f"{c:>22}" for c in cols))
    for name, row in results.items():
        cells = []
        for c in cols:
            v = row.get(c, float("nan"))
            cells.append(f"{'nan':>22}" if not np.isfinite(v) else f"{v:>22.4f}")
        print(f"{name:<8}" + "".join(cells))

    rd = context["roll_diagnostics"]
    print(f"\nroll: {rd['n_roll_days']} of {rd['n_days']} days "
          f"({rd['roll_share_of_days']*100:.1f}%), "
          f"{rd['volatility_ratio']:.2f}x the mean absolute change of other days, "
          f"{rd['roll_share_of_variance']*100:.1f}% of total variance. "
          f"{context['n_roll_days_in_eval']} fall in the evaluation window.")

    print("\nskill vs naive persistence (positive = better than repeating today's price)")
    for name, row in results.items():
        if "skill_vs_naive_rmse" in row:
            print(f"  {name:<6} RMSE {row['skill_vs_naive_rmse']:+.4f}   "
                  f"MAE {row['skill_vs_naive_mae']:+.4f}   "
                  f"RMSE excluding roll days {row['skill_ex_roll_rmse']:+.4f}")


def save(results: dict, context: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    payload = {
        "context": {k: v for k, v in context.items()
                    if k not in ("actual", "previous", "dates", "is_roll")},
        "models": {name: {k: v for k, v in row.items() if k != "predicted"}
                   for name, row in results.items()},
    }
    with open(os.path.join(RESULTS, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {os.path.join(RESULTS, 'metrics.json')}")


def figures(results: dict, context: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(IMG, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})
    dates, actual = context["dates"], context["actual"]

    # Predictions against actual. Plotted with the naive forecast included, because
    # the point of the figure is that all three lines sit on top of each other.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, actual, color="black", lw=1.2, label="actual")
    for name, color in (("naive", "0.6"), ("esn", "tab:red"), ("lstm", "tab:blue")):
        if name in results:
            ax.plot(dates, results[name]["predicted"], color=color, lw=0.9,
                    alpha=0.85, label=name)
    ax.set(xlabel="", ylabel="WTI front-month close ($/bbl)",
           title="One-step-ahead forecasts over the validation window")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(IMG, "forecasts.png"))
    plt.close(fig)

    # The same thing as errors, which is where the models actually differ.
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, color in (("naive", "0.6"), ("esn", "tab:red"), ("lstm", "tab:blue")):
        if name in results:
            ax.plot(dates, actual - results[name]["predicted"], color=color, lw=0.7,
                    alpha=0.8, label=f"{name} (RMSE {results[name]['rmse']:.2f})")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="", ylabel="actual - predicted ($/bbl)",
           title="Forecast errors: the models differ far less than the price moves")
    ax.legend(loc="lower left")
    fig.savefig(os.path.join(IMG, "errors.png"))
    plt.close(fig)

    # Zoom on April 2020, the negative-settlement window.
    mask = (dates >= "2020-03-01") & (dates <= "2020-06-30")
    if mask.any():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dates[mask], actual[mask], color="black", lw=1.4, label="actual")
        for name, color in (("naive", "0.6"), ("esn", "tab:red"), ("lstm", "tab:blue")):
            if name in results:
                ax.plot(dates[mask], results[name]["predicted"][mask], color=color,
                        lw=1.0, label=name)
        ax.axhline(0, color="0.4", lw=0.8, ls=":")
        ax.set(ylabel="$/bbl",
               title="April 2020: no model anticipates the negative settlement")
        ax.legend()
        fig.savefig(os.path.join(IMG, "april_2020.png"))
        plt.close(fig)

    print(f"wrote figures to {IMG}")


def run_holdout() -> None:
    """Score docs/PROTOCOL.md once, print the table, write results/holdout.json."""
    prices = load_prices()
    out = forecast.protocol(
        prices.to_numpy(float), prices.index, roll.roll_mask(prices.index).to_numpy())
    print(f"configurations fitted: {out['n_configs']}")
    print(f"{'h':>3} {'model':<12}{'n':>5}{'rmse':>8}{'mae':>8}{'skill':>8}{'dm_p':>8}"
          f"{'dir_acc':>9}{'binom_p':>9}")
    for h, hz in out["horizons"].items():
        print(f"h={h}: k={hz['k']} alpha={hz['alpha']} w={hz['w']:.3f}")
        for name, s in hz["rows"].items():
            print(f"{h:>3} {name:<12}{s['n']:>5}{s['rmse']:>8.3f}{s['mae']:>8.3f}"
                  f"{s['skill']:>+8.4f}{s['dm_p']:>8.3f}{s['dir_acc']:>9.3f}{s['binom_p']:>9.3f}")
    print("sign strategy h=1 (ridge):", out["horizons"][1]["sign_pnl"])
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "holdout.json"), "w") as fh:
        json.dump(out, fh, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window", type=int, default=WINDOW,
                   help="memory length in trading days (default 120)")
    p.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    p.add_argument("--no-lstm", action="store_true", help="skip the TensorFlow model")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--holdout", action="store_true", help="score the return models once")
    args = p.parse_args()

    if args.holdout:
        run_holdout()
        return

    results, context = run(args.window, args.train_frac, include_lstm=not args.no_lstm)
    print_table(results, context)
    save(results, context)
    if not args.no_figures:
        figures(results, context)


if __name__ == "__main__":
    main()
