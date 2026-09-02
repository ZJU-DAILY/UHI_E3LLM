#!/usr/bin/env python3
"""Render Fig. 7e PPO invalid-output penalty sensitivity."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
        "mathtext.fontset": "stix",
        "font.size": 22,
        "axes.labelsize": 28,
        "xtick.labelsize": 23,
        "ytick.labelsize": 23,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

LINE_COLOR = "#287FB8"
BEST_COLOR = "#D6452F"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/sensitivity/ppo_lambda_summary.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    if not args.summary.is_file():
        raise SystemExit(
            f"missing {args.summary}; run scripts/run_ppo_penalty_sensitivity.sh first"
        )
    table = pd.read_csv(args.summary).sort_values("lambda")
    required = {"lambda", "mae", "mse", "rmse", "cities", "samples"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"lambda summary is missing columns: {sorted(missing)}")
    expected = [0, 1, 5, 10, 20]
    if table["lambda"].tolist() != expected:
        raise ValueError(
            f"lambda summary must contain exactly {expected}; "
            f"found {table['lambda'].tolist()}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.5, 4.7))
    axis.plot(table["lambda"], table["mae"], color=LINE_COLOR,
              linewidth=3.2, zorder=2)
    axis.scatter(table["lambda"], table["mae"], s=150,
                 color=LINE_COLOR, edgecolors=LINE_COLOR, zorder=3)
    best = table.loc[table["mae"].idxmin()]
    axis.scatter([best["lambda"]], [best["mae"]], s=190,
                 color=BEST_COLOR, edgecolors=BEST_COLOR, zorder=4)
    low, high = float(table["mae"].min()), float(table["mae"].max())
    pad = max((high - low) * 0.55, 0.001)
    axis.set_ylim(low - pad, high + pad)
    axis.set_xlim(-0.8, 20.8)
    axis.set_xticks(table["lambda"])
    axis.set_xlabel(r"Penalty coefficient $\lambda$")
    axis.set_ylabel("MAE")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.out_dir / f"fig7e_lambda_sensitivity.{suffix}",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(f"[fig] {args.out_dir / 'fig7e_lambda_sensitivity'}.png/.pdf")


if __name__ == "__main__":
    main()
