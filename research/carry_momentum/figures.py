"""Figures for NOTE.md. Run after study.py."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import engine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import study

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def main() -> None:
    d = study.build_inputs()
    h1 = study.strategy(d, *study.PRIMARY)
    h2 = study.strategy(d, *study.PRIMARY, spread=False)
    h1g = study.strategy(d, *study.PRIMARY, costs=engine.Costs(0, 0))
    with open(os.path.join(OUT, "study.json")) as fh:
        res = json.load(fh)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for r, lab, st in [
        (h2.returns, "H2 outright, net", "-"),
        (h1.returns, "H1 spread, net", "-"),
        (h1g.returns, "H1 spread, before costs", "--"),
    ]:
        r = r.loc[study.START :]
        ax.plot(
            (1 + r).cumprod(), st, lw=1.1, label=f"{lab} (Sharpe {study.sharpe(r):.2f})"
        )
    for a, b in study.UNSEEN:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="grey", alpha=0.12, lw=0)
    ax.set_yscale("log")
    ax.set_title(
        "WTI carry-momentum, $10m book at 15% vol (grey: unseen windows)", fontsize=10
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    k1, k2 = res["kill_criteria"], res["kill_criteria_H2"]
    labels = [
        "K1 unseen Sharpe",
        "K2 bootstrap low",
        "K3 deflated Sharpe",
        "K4 PBO",
        "K5 alpha t",
        "K6 costs x2",
        "K7 capacity",
    ]
    keys = list(k1)
    ax.axis("off")
    rows = []
    for lab, k in zip(labels, keys):
        f = (lambda v: f"${v / 1e6:.1f}m") if k == "K7" else (lambda v: f"{v:.2f}")
        rows.append(
            [
                lab,
                k1[k]["threshold"].replace("10000000.0", "$10m"),
                f(k1[k]["value"]) + (" x" if k1[k]["killed"] else " ok"),
                f(k2[k]["value"]) + (" x" if k2[k]["killed"] else " ok"),
            ]
        )
    t = ax.table(
        cellText=rows,
        colLabels=["criterion", "kill if", "H1 spread", "H2 outright"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        if r > 0 and c >= 2 and cell.get_text().get_text().endswith(" x"):
            cell.set_facecolor("#f6d5d5")
        elif r > 0 and c >= 2:
            cell.set_facecolor("#d9eedb")
    ax.set_title("Pre-registered kill criteria (x = killed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "verdict.png"), dpi=130)
    plt.close(fig)

    # Where H1's money goes: gross vs costs by year.
    yr = pd.DataFrame(
        {
            "gross": h1.gross.loc[study.START :],
            "cost": -h1.cost.loc[study.START :] / 10e6,
        }
    )
    yr = yr.groupby(yr.index.year).sum()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(yr.index, yr.gross * 100, color="#4c78a8", label="gross P&L")
    ax.bar(yr.index, yr.cost * 100, color="#e45756", label="costs")
    ax.plot(yr.index, (yr.gross + yr.cost) * 100, "k.-", lw=1, label="net")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("% of capital")
    ax.set_title(
        "H1 spread by year: the curve signal is there, the costs are larger",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "h1_costs.png"), dpi=130)
    plt.close(fig)
    print("ok", np.round(yr.sum().values * 100, 1))


if __name__ == "__main__":
    main()
