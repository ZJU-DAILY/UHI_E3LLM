#!/usr/bin/env python3
"""Render the final Fig. 5a five-city prediction comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_data import CASE_CITIES  # noqa: E402


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
        "font.size": 14,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_predictions(directories: list[Path]) -> pd.DataFrame:
    frames = []
    for directory in directories:
        files = sorted(directory.glob("city_*_predictions.csv"))
        if not files:
            raise FileNotFoundError(f"no city_*_predictions.csv below {directory}")
        for file in files:
            frame = pd.read_csv(file).rename(
                columns={"year": "target_year", "month": "target_month"}
            )
            frames.append(
                frame[
                    [
                        "city_idx",
                        "target_year",
                        "target_month",
                        "true_uhi",
                        "predicted_uhi",
                    ]
                ]
            )
    merged = pd.concat(frames, ignore_index=True)
    keys = ["city_idx", "target_year", "target_month"]
    if merged.duplicated(keys).any():
        raise ValueError("case-study prediction inputs contain duplicate city-month keys")
    expected_cities = {city_id for city_id, _ in CASE_CITIES}
    actual_cities = set(merged["city_idx"].astype(int))
    if actual_cities != expected_cities:
        raise ValueError(
            f"case-study predictions require cities {sorted(expected_cities)}; "
            f"found {sorted(actual_cities)}"
        )
    expected_targets = {
        (year, month) for year in range(2004, 2021) for month in range(1, 13)
    }
    for city_id, rows in merged.groupby("city_idx"):
        targets = set(
            rows[["target_year", "target_month"]].itertuples(index=False, name=None)
        )
        if targets != expected_targets:
            raise ValueError(
                f"case-study city {city_id} must contain all 204 targets from "
                "2004-01 through 2020-12"
            )
    merged["absolute_error"] = (
        merged["predicted_uhi"] - merged["true_uhi"]
    ).abs()
    merged["date"] = pd.to_datetime(
        merged["target_year"].astype(int).astype(str)
        + "-"
        + merged["target_month"].astype(int).astype(str).str.zfill(2)
        + "-01"
    )
    return merged.sort_values(["city_idx", "date"])


def plot_fig5a(predictions: pd.DataFrame, out_dir: Path) -> None:
    figure, axes = plt.subplots(
        len(CASE_CITIES),
        1,
        figsize=(4.0, 3.2 * len(CASE_CITIES)),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.3},
    )
    plotted = []
    for axis, (city_id, label) in zip(np.atleast_1d(axes), CASE_CITIES):
        rows = predictions[predictions["city_idx"] == city_id].copy()
        axis.plot(rows["date"], rows["true_uhi"], color="#111111",
                  linewidth=2.0, label="True UHI")
        axis.plot(rows["date"], rows["predicted_uhi"], color="#3A86FF",
                  linewidth=2.0, linestyle="--", label="Predicted UHI")
        axis.set_title(label)
        axis.set_ylabel("UHI")
        axis.set_xlabel("Year")
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
        plotted.extend([rows["true_uhi"], rows["predicted_uhi"]])
    if plotted:
        lower = min(values.min() for values in plotted)
        upper = max(values.max() for values in plotted)
        for axis in np.atleast_1d(axes):
            axis.set_ylim(lower, upper)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower right", frameon=False,
        bbox_to_anchor=(0.98, 0.06),
    )
    figure.autofmt_xdate(rotation=45)
    figure.subplots_adjust(bottom=0.15, right=0.92)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            out_dir / f"fig5a_case_city_predictions.{suffix}",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(f"[fig] {out_dir / 'fig5a_case_city_predictions'}.png/.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/case_study/inference_train"),
            Path("outputs/case_study/inference_valid"),
        ],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    directories = []
    for root in args.predictions:
        if not root.is_dir():
            continue
        nested = sorted(
            path
            for path in root.rglob("city_results")
            if any(path.glob("city_*_predictions.csv"))
        )
        directories.extend(
            nested or ([root] if any(root.glob("city_*_predictions.csv")) else [])
        )
    if not directories:
        raise FileNotFoundError(
            f"no case-study prediction directories found in {args.predictions}; "
            "run scripts/run_case_study_outputs.sh first"
        )
    predictions = load_predictions(directories)
    predictions.to_csv(data_dir / "fig5a_case_city_predictions.csv", index=False)
    plot_fig5a(predictions, args.out_dir)


if __name__ == "__main__":
    main()
