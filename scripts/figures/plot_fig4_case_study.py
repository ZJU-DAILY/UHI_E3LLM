#!/usr/bin/env python3
"""Figure 4b-c: case-study UHI distributions and factor evolution.

Panels:
  4b  monthly UHI histograms read from the processed per-city datasets
  4c  normalized five-city temporal evolution of the factors used in the prompt

Panel 4a is prepared manually in QGIS. Panel 4b is reconstructed from
data/train and data/valid. Panel 4c reads the compact table
generated from raw factor CSVs by build_fig4c_source.py. No plotted value is
hard-coded, and factors absent from the final prompt are not plotted.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from datasets import concatenate_datasets, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_data import CASE_CITIES  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
    "pdf.fonttype": 42,
})

FIG4B_COLORS = {
    864: "#ffbe0b",
    677: "#fb5607",
    222: "#ff006e",
    661: "#8338ec",
    413: "#3a86ff",
}

CATEGORY_ORDER = [
    "Climate Data",
    "Built Surface Area",
    "Population Density",
    "Urbanization Level",
    "Vegetation Cover",
    "PM2.5 Concentration",
]
CATEGORY_COLORS = {
    "Climate Data": "#1f77b4",
    "Built Surface Area": "#ffbb78",
    "Population Density": "#d62728",
    "Urbanization Level": "#ff9896",
    "Vegetation Cover": "#9467bd",
    "PM2.5 Concentration": "#c5b0d5",
}
FACTOR_ORDER = {
    "Climate Data": [
        "Average Wind Speed",
        "Average Temperature",
        "Average Cloud Cover",
        "Precipitation Flux",
        "Snow Thickness",
        "Solar Radiation",
        "Average Vapour Pressure",
    ],
    "Vegetation Cover": ["NDVI", "EVI"],
}
FACTOR_COLORS = {
    "Average Wind Speed": "#17becf",
    "Average Temperature": "#1f77b4",
    "Average Cloud Cover": "#ff7f0e",
    "Precipitation Flux": "#d62728",
    "Snow Thickness": "#bcbd22",
    "Solar Radiation": "#2ca02c",
    "Average Vapour Pressure": "#7f7f7f",
    "NDVI": "#8c564b",
    "EVI": "#e377c2",
    "Built Surface Area": "#ffd3a7",
    "Population Density": "#e47373",
    "Urbanization Level": "#ffbdbb",
    "PM2.5 Concentration": "#d9cce4",
}
FULL_ALPHA_GROUPS = {
    "Built Surface Area",
    "Population Density",
    "Urbanization Level",
    "PM2.5 Concentration",
}
BOLD_MEAN_GROUPS = {
    "Climate Data",
    "Vegetation Cover",
    "PM2.5 Concentration",
}


def _save(figure, out_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(out_dir / f"{stem}.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"[fig] {out_dir / stem}.png/.pdf")


def scalar(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def load_city(data_dir: Path, city_id: int):
    parts = [
        load_from_disk(str(data_dir / split / f"UHI_city_{city_id}"))
        for split in ("train", "valid")
    ]
    dataset = concatenate_datasets(parts)
    rows = sorted(dataset, key=lambda r: (scalar(r["year_info"]), scalar(r["month_info"])))
    return rows


def complete_uhi_series(rows: list) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Recover 2003-2020 UHI values from the first history and all targets."""
    first = rows[0]
    target_start = pd.Timestamp(
        int(scalar(first["year_info"])), int(scalar(first["month_info"])), 1
    )
    history = np.asarray(scalar(first["x"]), dtype=float).ravel()
    if history.size != 12:
        raise ValueError(f"expected 12 historical UHI values, found {history.size}")
    targets = np.asarray([float(scalar(row["y"])) for row in rows], dtype=float)
    values = np.concatenate([history, targets])
    dates = pd.date_range(target_start - pd.DateOffset(months=12), periods=len(values), freq="MS")
    return dates, values


def plot_fig4b(city_rows: dict[int, list], out_dir: Path, data_dir_out: Path) -> None:
    """Render the five-city monthly UHI stacked histograms."""
    records = []
    values_by_city = {}
    for city_id, label in CASE_CITIES:
        dates, values = complete_uhi_series(city_rows[city_id])
        values_by_city[city_id] = values
        records.extend(
            {
                "city_idx": city_id,
                "city": label,
                "year": date.year,
                "month": date.month,
                "uhi": value,
            }
            for date, value in zip(dates, values)
        )

    all_values = np.concatenate(list(values_by_city.values()))
    lower, upper = float(all_values.min()), float(all_values.max())
    if lower == upper:
        lower, upper = lower - 0.5, upper + 0.5
    bins = np.linspace(lower, upper, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]

    figure, axis = plt.subplots(figsize=(10.5, 4.5))
    bottom = np.zeros(40, dtype=float)
    for city_id, label in CASE_CITIES:
        counts, _ = np.histogram(values_by_city[city_id], bins=bins)
        axis.bar(
            centers,
            counts,
            width=width,
            bottom=bottom,
            color=FIG4B_COLORS[city_id],
            edgecolor="white",
            linewidth=0.3,
            label=label,
            align="center",
        )
        bottom += counts
    axis.set_xlabel("Monthly UHI Intensity (°C)")
    axis.set_ylabel("Frequency")
    axis.legend(
        frameon=False,
        fontsize=11,
        ncol=2,
        loc="upper right",
        columnspacing=1.0,
        handletextpad=0.5,
    )
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, out_dir, "fig4b_case_city_uhi_distributions")
    pd.DataFrame(records).to_csv(data_dir_out / "fig4b_case_city_uhi.csv", index=False)


def plot_fig4c(
    source_path: Path,
    out_dir: Path,
    data_dir_out: Path,
    stem: str = "fig4c_case_city_factor_evolution",
) -> None:
    """Render the normalized temporal evolution of prompt factors."""
    source = pd.read_csv(source_path, parse_dates=["date"])
    required = {"date", "group", "factor", "raw_mean", "normalized_value"}
    if missing := required.difference(source.columns):
        raise ValueError(f"{source_path}: missing columns {sorted(missing)}")
    if missing := set(CATEGORY_ORDER).difference(source["group"]):
        raise ValueError(f"{source_path}: missing factor groups {sorted(missing)}")

    figure, (axis, legend_axis) = plt.subplots(
        1,
        2,
        figsize=(15.5, 6.0),
        gridspec_kw={"width_ratios": [4.2, 3.03], "wspace": 0.06},
    )
    legend_axis.axis("off")
    factor_handles: dict[str, list] = {group: [] for group in FACTOR_ORDER}
    factor_labels: dict[str, list[str]] = {group: [] for group in FACTOR_ORDER}
    mean_handles = []
    mean_labels = []

    for group in CATEGORY_ORDER:
        group_frame = source[source["group"].eq(group)]
        observed_factors = list(group_frame["factor"].drop_duplicates())
        ordered_factors = [
            factor for factor in FACTOR_ORDER.get(group, observed_factors)
            if factor in observed_factors
        ]
        for factor in ordered_factors:
            series = group_frame[group_frame["factor"].eq(factor)].sort_values("date")
            line, = axis.plot(
                series["date"],
                series["normalized_value"],
                linewidth=0.9,
                color=FACTOR_COLORS[factor],
                alpha=0.35 if group in FULL_ALPHA_GROUPS else 0.21,
            )
            if group in factor_handles:
                factor_handles[group].append(line)
                factor_labels[group].append(factor)

        if group in BOLD_MEAN_GROUPS:
            group_mean = (
                group_frame.groupby("date", as_index=False)["normalized_value"]
                .mean()
                .sort_values("date")
            )
            axis.plot(
                group_mean["date"],
                group_mean["normalized_value"],
                linewidth=2.0,
                color=CATEGORY_COLORS[group],
            )
        mean_handles.append(Line2D(
            [0],
            [0],
            linewidth=2.0,
            color=CATEGORY_COLORS[group],
        ))
        factor_count = group_frame["factor"].nunique()
        suffix = "var" if factor_count == 1 else "vars"
        mean_labels.append(f"{group} ({factor_count} {suffix})")

    axis.set_ylabel("Normalized value", fontsize=14, labelpad=10)
    axis.set_xlabel("Year", fontsize=14, labelpad=10)
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlim(source["date"].min(), source["date"].max())
    axis.grid(False)
    axis.xaxis.set_major_locator(mdates.YearLocator(base=1))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    axis.spines[["top", "right"]].set_visible(False)

    legend_axis.add_patch(plt.Rectangle(
        (0.01, 0.115),
        0.98,
        0.77,
        transform=legend_axis.transAxes,
        facecolor="white",
        edgecolor="#d0d0d0",
        linewidth=1.0,
    ))

    def draw_legend_entry(x, y, handle, label, fontsize=11.0, line_width=0.065):
        alpha = handle.get_alpha()
        legend_axis.plot(
            [x, x + line_width],
            [y, y],
            transform=legend_axis.transAxes,
            color=handle.get_color(),
            linewidth=handle.get_linewidth(),
            alpha=1.0 if alpha is None else alpha,
            solid_capstyle="butt",
            clip_on=False,
        )
        legend_axis.text(
            x + line_width + 0.025,
            y,
            label,
            transform=legend_axis.transAxes,
            va="center",
            ha="left",
            fontsize=fontsize,
        )

    legend_axis.text(
        0.5, 0.855, "Dataset means", transform=legend_axis.transAxes,
        ha="center", va="center", fontsize=14,
    )
    mean_columns = ((0.04, range(0, 3)), (0.52, range(3, 6)))
    mean_rows = (0.795, 0.74, 0.685)
    for x, indices in mean_columns:
        for y, index in zip(mean_rows, indices):
            draw_legend_entry(
                x,
                y,
                mean_handles[index],
                mean_labels[index],
                fontsize=10.6,
                line_width=0.05,
            )

    legend_axis.plot(
        [0.05, 0.95], [0.64, 0.64], transform=legend_axis.transAxes,
        color="#d8d8d8", linewidth=0.8,
    )
    legend_axis.text(
        0.5, 0.595, "Detailed Prompt Factors", transform=legend_axis.transAxes,
        ha="center", va="center", fontsize=14,
    )

    legend_axis.text(
        0.05, 0.54, "Climate Data", transform=legend_axis.transAxes,
        ha="left", va="center", fontsize=12.5, fontweight="bold",
    )
    climate_handles = factor_handles["Climate Data"]
    climate_labels = factor_labels["Climate Data"]
    climate_columns = (
        (0.05, range(0, 4)),
        (0.51, range(4, 7)),
    )
    climate_rows = (0.485, 0.42, 0.355, 0.29)
    for x, indices in climate_columns:
        for y, index in zip(climate_rows, indices):
            draw_legend_entry(
                x,
                y,
                climate_handles[index],
                climate_labels[index],
                fontsize=10.8,
                line_width=0.055,
            )

    legend_axis.text(
        0.05, 0.20, "Vegetation Cover", transform=legend_axis.transAxes,
        ha="left", va="center", fontsize=12.5, fontweight="bold",
    )
    for x, handle, label in zip(
        (0.05, 0.30),
        factor_handles["Vegetation Cover"],
        factor_labels["Vegetation Cover"],
    ):
        draw_legend_entry(x, 0.145, handle, label, fontsize=10.8)

    _save(figure, out_dir, stem)
    source.to_csv(data_dir_out / f"{stem}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument(
        "--fig4c-source",
        type=Path,
        default=Path(
            "outputs/figures/data/fig4c_factor_evolution_5cities.csv"
        ),
    )
    parser.add_argument(
        "--fig4c-stem",
        default="fig4c_case_city_factor_evolution",
        help="Output filename stem for panel 4c.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir_out = args.out_dir / "data"
    data_dir_out.mkdir(parents=True, exist_ok=True)

    missing = [
        city_id for city_id, _ in CASE_CITIES
        if not (args.data_dir / "train" / f"UHI_city_{city_id}").is_dir()
    ]
    if missing:
        raise SystemExit(f"missing case-study city datasets below {args.data_dir}: {missing}")
    if not args.fig4c_source.is_file():
        raise SystemExit(f"missing Fig. 4c source table: {args.fig4c_source}")
    city_rows = {city_id: load_city(args.data_dir, city_id) for city_id, _ in CASE_CITIES}

    plot_fig4b(city_rows, args.out_dir, data_dir_out)
    plot_fig4c(
        args.fig4c_source,
        args.out_dir,
        data_dir_out,
        stem=args.fig4c_stem,
    )


if __name__ == "__main__":
    main()
