#!/usr/bin/env python3
"""Figure 2a-2c and 2g: city-size distributions and regional city counts.

Reads only the versioned manifests under data/city_metrics/; no values are
hard-coded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_data import AREA_BINS, AREA_LABELS  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "pdf.fonttype": 42,
})

PANEL_COLORS = {"all": "#7FB3D5", "train": "#82C99B", "test": "#F4B860"}
REGION_COLORS = {
    "Africa": "#CB3A2E",
    "East Asia & Oceania": "#E8952F",
    "Europe": "#2E8B57",
    "Latin America": "#4E79A7",
    "North America": "#7B52A1",
    "West Asia": "#C97BA6",
}


def _save(figure, out_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(out_dir / f"{stem}.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"[fig] {out_dir / stem}.png/.pdf")


def size_panel(axis, areas: pd.Series, color: str, title: str) -> None:
    counts, _ = np.histogram(areas, bins=AREA_BINS)
    positions = np.arange(len(AREA_LABELS))
    axis.bar(positions, counts, color=color, width=0.7)
    for pos, count in zip(positions, counts):
        axis.text(pos, count, str(int(count)), ha="center", va="bottom", fontsize=7)

    cumulative = axis.twinx()
    cumulative.plot(positions, np.cumsum(counts) / counts.sum() * 100,
                    color="#444444", marker="o", markersize=3, linewidth=1.0)
    cumulative.set_ylim(0, 105)
    cumulative.set_ylabel("Cumulative percentage (%)", fontsize=8)

    axis.set_xticks(positions)
    axis.set_xticklabels(
        ["100-150", "150-450", "450-800", "800-1500", ">=1500"],
        rotation=30, ha="right", fontsize=7,
    )
    axis.set_xlabel("Area range (km$^2$)", fontsize=8)
    axis.set_ylabel("Count of cities", fontsize=8)
    axis.set_title(title, fontsize=9)
    axis.spines[["top"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-metrics-dir", type=Path, default=Path("data/city_metrics"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        "all": pd.read_csv(args.city_metrics_dir / "city_metrics_all.csv"),
        "train": pd.read_csv(args.city_metrics_dir / "city_metrics_train.csv"),
        "test": pd.read_csv(args.city_metrics_dir / "city_metrics_test.csv"),
    }
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    for axis, (name, title) in zip(
        axes,
        [("all", "All candidate cities"), ("train", "Training cities"), ("test", "Testing cities")],
    ):
        size_panel(axis, frames[name]["Area"], PANEL_COLORS[name], title)
    figure.tight_layout()
    _save(figure, args.out_dir, "fig2abc_city_size_distributions")

    regions_path = args.city_metrics_dir / "region_assignments_600.csv"
    if regions_path.is_file():
        regions = pd.read_csv(regions_path)
        column = "region" if "region" in regions.columns else "Region"
        counts = regions[column].value_counts()
        order = [name for name in REGION_COLORS if name in counts.index]
        figure, axis = plt.subplots(figsize=(4.6, 3.0))
        positions = np.arange(len(order))
        values = [int(counts[name]) for name in order]
        axis.bar(positions, values, color=[REGION_COLORS[name] for name in order], width=0.65)
        for pos, value in zip(positions, values):
            axis.text(pos, value, str(value), ha="center", va="bottom", fontsize=8)
        axis.set_xticks(positions)
        axis.set_xticklabels(order, rotation=30, ha="right", fontsize=7)
        axis.set_ylabel("Number of sampled cities")
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        _save(figure, args.out_dir, "fig2g_regional_city_counts")
    else:
        print(f"[skip] fig2g: missing {regions_path}")


if __name__ == "__main__":
    main()
