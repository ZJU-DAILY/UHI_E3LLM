#!/usr/bin/env python3
"""Figure 2h: evidence-driven factor selection (literature study counts).

The fixed counts below were collected through manual factor-specific Web of
Science (WoS) searches and represent literature-survey evidence. NDVI
and EVI share the survey's vegetation-index category, so the 18 model inputs
are represented by 17 bars.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors, font_manager
import numpy as np
import pandas as pd

try:
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    FIGURE_FONT = "Times New Roman"
except ValueError:
    FIGURE_FONT = "Liberation Serif"

plt.rcParams.update({
    "font.family": FIGURE_FONT,
    "font.size": 9,
    "pdf.fonttype": 42,
})

EXPECTED_FACTORS = {
    "Area",
    "Built Height",
    "Built Surface Area",
    "Urbanization Level",
    "City Compactness",
    "Urban Land Use",
    "Location",
    "Population Density",
    "Average Cloud Cover",
    "Solar Radiation",
    "Precipitation Flux",
    "Snow Thickness",
    "NDVI, EVI",
    "PM2.5 Concentration",
    "Average Temperature",
    "Average Vapour Pressure",
    "Average Wind Speed",
}

MANUAL_WOS_COUNTS = [
    ("Area", "structural", 1189),
    ("Built Height", "structural", 364),
    ("Built Surface Area", "structural", 145),
    ("Urbanization Level", "structural", 3659),
    ("City Compactness", "structural", 164),
    ("Urban Land Use", "structural", 2498),
    ("Location", "structural", 314),
    ("Population Density", "structural", 1688),
    ("Average Cloud Cover", "liquidity", 350),
    ("Solar Radiation", "liquidity", 686),
    ("Precipitation Flux", "liquidity", 916),
    ("Snow Thickness", "liquidity", 107),
    ("NDVI, EVI", "liquidity", 1247),
    ("PM2.5 Concentration", "liquidity", 248),
    ("Average Temperature", "liquidity", 2563),
    ("Average Vapour Pressure", "liquidity", 324),
    ("Average Wind Speed", "liquidity", 669),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    table = pd.DataFrame(
        MANUAL_WOS_COUNTS,
        columns=["factor", "group", "studies"],
    )
    factors = set(table["factor"])
    if factors != EXPECTED_FACTORS:
        missing = sorted(EXPECTED_FACTORS - factors)
        unexpected = sorted(factors - EXPECTED_FACTORS)
        raise SystemExit(
            "factor labels do not match the structured prompt; "
            f"missing={missing}, unexpected={unexpected}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    groups = [
        ("structural", "Structural Factors"),
        ("liquidity", "Liquidity Factors"),
    ]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(18.0, 6.5),
        layout="constrained",
        gridspec_kw={"wspace": 0.08},
    )

    cmap = matplotlib.colormaps["Oranges"]
    norm = colors.PowerNorm(
        gamma=0.75,
        vmin=float(table["studies"].min()),
        vmax=float(table["studies"].max()),
    )
    max_count = float(table["studies"].max())

    for axis, (group, title) in zip(axes, groups):
        rows = table[table["group"] == group].sort_values(
            "studies", ascending=False
        )
        positions = np.arange(len(rows))
        values = rows["studies"].astype(float).to_numpy()
        bars = axis.barh(
            positions,
            values,
            color=[cmap(norm(value)) for value in values],
            edgecolor="#e5e7eb",
            linewidth=0.8,
            height=0.60,
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(rows["factor"], fontsize=18, color="#111827")
        axis.invert_yaxis()
        axis.grid(
            axis="x",
            color="#e5e7eb",
            linewidth=0.8,
            linestyle="--",
            alpha=0.7,
        )
        axis.set_axisbelow(True)
        axis.set_facecolor("white")
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color("#d1d5db")
        axis.tick_params(axis="y", length=0)
        axis.tick_params(axis="x", labelsize=15)

        for bar, value in zip(bars, values):
            axis.text(
                value + max_count * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{int(value):,}",
                va="center",
                ha="left",
                fontsize=16,
                color="#78350f",
            )

        axis.set_title(
            title,
            loc="left",
            fontsize=22,
            fontweight="bold",
            color="#78350f",
            pad=10,
        )
        axis.set_xlabel(
            "Number of studies",
            fontsize=18,
            color="#111827",
            labelpad=10,
        )
        axis.set_xlim(0, max_count * 1.08)
        axis.set_xticks(np.arange(0, 3501, 500))

    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = figure.colorbar(scalar_map, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("Number of studies", fontsize=18, color="#111827", labelpad=10)
    colorbar.ax.tick_params(labelsize=14, colors="#111827")
    figure.patch.set_facecolor("white")

    for suffix in ("png", "pdf"):
        figure.savefig(args.out_dir / f"fig2h_factor_evidence.{suffix}",
                       dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"[fig] {args.out_dir / 'fig2h_factor_evidence'}.png/.pdf")


if __name__ == "__main__":
    main()
