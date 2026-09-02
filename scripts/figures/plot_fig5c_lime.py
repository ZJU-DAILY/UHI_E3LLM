#!/usr/bin/env python3
"""Render Fig. 5c from the case-study workflow's aggregated LIME weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
        "font.size": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

POSITIVE_COLOR = "#d73027"
NEGATIVE_COLOR = "#4575b4"
FEATURE_LABELS = {
    "pop": "Population Density",
    "wind": "Average Wind Speed",
    "evi": "EVI",
    "precipitation": "Precipitation Flux",
    "cloud": "Average Cloud Cover",
    "temperature": "Average Temperature",
    "ndvi": "NDVI",
    "compactness": "City Compactness",
    "water": "Water Surfaces",
    "latitude": "Location",
    "solar": "Solar Radiation",
    "vapour": "Average Vapour Pressure",
    "snow": "Snow Thickness",
    "pm25": "PM2.5 Concentration",
    "built_surface": "Built Surface Area",
    "built_height": "Built Height",
    "road": "Road Surfaces",
    "urbanization_level": "Urbanization Level",
    "residential": "Residential",
    "area": "Area",
    "location": "Location",
    "open_space": "Open Spaces",
    "non_residential": "Non-Residential",
}


def load_weights(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "panel_order",
        "display_city",
        "source_city_id",
        "rank",
        "feature",
        "lime_importance",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    counts = frame.groupby("panel_order").size()
    if len(counts) != 5 or not counts.eq(10).all():
        raise ValueError("Fig. 5c requires five panels with ten LIME bars each")
    return frame.sort_values(["panel_order", "rank"]).reset_index(drop=True)


def plot_lime(frame: pd.DataFrame, out_dir: Path) -> None:
    figure, axes = plt.subplots(
        5,
        1,
        figsize=(3.5, 10.0),
        sharex=True,
        constrained_layout=True,
    )
    for axis, panel_order in zip(axes, sorted(frame["panel_order"].unique())):
        rows = frame[frame["panel_order"].eq(panel_order)].copy()
        positions = np.arange(len(rows), dtype=float)
        values = rows["lime_importance"].to_numpy(dtype=float)
        colors = np.where(values >= 0, POSITIVE_COLOR, NEGATIVE_COLOR)

        axis.barh(positions, values, height=0.72, color=colors)
        display_labels = rows["feature"].map(FEATURE_LABELS)
        if display_labels.isna().any():
            missing = sorted(rows.loc[display_labels.isna(), "feature"].unique())
            raise ValueError(f"Missing prompt labels for LIME features: {missing}")
        axis.set_yticks(positions, display_labels)
        axis.invert_yaxis()
        axis.axvline(0, color="#4c4c4c", linewidth=0.8)
        axis.set_xlim(-0.2, 0.2)
        axis.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="y", labelsize=10)

    for axis in axes[:-1]:
        axis.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Lime Importance", fontsize=12)

    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            out_dir / f"fig5c_case_city_lime.{suffix}",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(f"[fig] {out_dir / 'fig5c_case_city_lime'}.png/.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/figures/data/fig5c_lime_importance.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    plot_lime(load_weights(args.input), args.out_dir)


if __name__ == "__main__":
    main()
