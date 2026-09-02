#!/usr/bin/env python3
"""Render the final Fig. 3a/3b performance panels from normalized results."""

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
from figure_data import AREA_LABELS, attach_area_group  # noqa: E402


METHOD_ORDER = [
    "UHI-E3LLM",
    "ARMD",
    "FilterNet",
    "ST-MTM",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "MLP",
]
DISPLAY_NAMES = {
    "UHI-E3LLM": r"UHI-E$^3$LLM",
    "RandomForest": "Random Forest",
}
COLORS = {
    "UHI-E3LLM": "#0C6291",
    "ARMD": "#C0392B",
    "FilterNet": "#1E8A4D",
    "ST-MTM": "#7D3C98",
    "RandomForest": "#8C564B",
    "XGBoost": "#E67E22",
    "LightGBM": "#17A589",
    "MLP": "#D45087",
}
AREA_TITLES = {
    "Very small": r"100–150 km$^2$",
    "Small": r"150–450 km$^2$",
    "Medium": r"450–800 km$^2$",
    "Large": r"800–1500 km$^2$",
    "Very large": r"$\geq$1500 km$^2$",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
        "mathtext.fontset": "stix",
        "font.size": 17,
        "axes.labelsize": 21,
        "axes.titlesize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _display(method: str) -> str:
    return DISPLAY_NAMES.get(method, method)


def _save(figure: plt.Figure, out_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(out_dir / f"{stem}.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"[fig] {out_dir / stem}.png/.pdf")


def _bandwidth(values: np.ndarray) -> float:
    if values.size < 2:
        return 1e-3
    std = float(np.std(values, ddof=1))
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    positive = [value for value in (std, iqr / 1.349) if value > 0]
    sigma = min(positive) if positive else 1.0
    return max(0.9 * sigma * values.size ** (-1 / 5), 1e-3)


def _kde(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = float(values.min()), float(values.max())
    padding = max((high - low) * 0.1, 1e-3)
    grid = np.linspace(max(0.0, low - padding), high + padding, 512)
    bandwidth = _bandwidth(values)
    scaled = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1)
    density /= values.size * bandwidth * np.sqrt(2 * np.pi)
    if density.max() > 0:
        density /= density.max()
    return grid, density


def plot_fig3a(city: pd.DataFrame, out_dir: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(city["method"])]
    if methods != METHOD_ORDER:
        raise ValueError(f"Fig. 3a requires all methods in order {METHOD_ORDER}; got {methods}")
    positions = list(range(1, len(methods) + 1))
    rng = np.random.default_rng(0)
    figure, axis = plt.subplots(figsize=(8.5, max(4.2, 0.78 * len(methods))))

    values_by_method: dict[str, np.ndarray] = {}
    for position, method in zip(positions, methods):
        values = city.loc[city["method"] == method, "mae"].to_numpy(dtype=float)
        values_by_method[method] = values
        q10, q90 = np.quantile(values, [0.10, 0.90])
        kde_values = values[(values >= q10) & (values <= q90)]
        grid, density = _kde(kde_values)
        axis.fill_between(
            grid, position, position + density * 0.34,
            color=COLORS[method], alpha=0.50, linewidth=0,
        )
        axis.plot(grid, position + density * 0.34, color=COLORS[method], linewidth=1.2)
        scatter_y = position - 0.25 + rng.normal(0, 0.045, size=values.size)
        axis.scatter(
            values, scatter_y, color="#1F2933", alpha=0.45,
            s=10, linewidth=0, rasterized=True,
        )
        box = axis.boxplot(
            values, positions=[position - 0.25], vert=False, widths=0.18,
            patch_artist=True, manage_ticks=False, showfliers=False,
        )
        for patch in box["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor("#1D3557")
            patch.set_alpha(0.82)
        for element in ("whiskers", "caps", "medians"):
            for artist in box[element]:
                artist.set_color("#1D3557")
                artist.set_linewidth(1.0)

    axis.axvline(
        float(np.median(values_by_method["UHI-E3LLM"])),
        color="black", linestyle=(0, (5, 4)), linewidth=1.4, alpha=0.30,
    )
    axis.set_yticks(positions, [_display(method) for method in methods])
    axis.set_ylim(0.50, len(methods) + 0.65)
    axis.set_xlim(0, 2.0)
    axis.set_xticks(np.arange(0, 2.01, 0.25))
    axis.set_xlabel("MAE", labelpad=10)
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save(figure, out_dir, "fig3a_all_results")


def _size_year_summary(city_year: pd.DataFrame) -> pd.DataFrame:
    grouped = city_year.groupby(
        ["area_group", "method", "target_year"], observed=True
    )["mae"]
    summary = grouped.agg(
        mae_mean="mean", mae_std="std", city_count="count"
    ).reset_index()
    summary["mae_sem"] = summary["mae_std"] / np.sqrt(summary["city_count"])
    return summary


def plot_fig3b(city_year: pd.DataFrame, out_dir: Path) -> None:
    expected_years = set(range(2004, 2021))
    observed = city_year.groupby("method")["target_year"].apply(
        lambda values: set(values.astype(int))
    )
    invalid = {
        method: sorted(years)
        for method, years in observed.items()
        if years != expected_years
    }
    if invalid:
        raise ValueError(
            "Fig. 3b requires every method to cover 2004-2020; "
            f"invalid year sets: {invalid}"
        )
    summary = _size_year_summary(city_year)
    figure, axis_grid = plt.subplots(3, 2, figsize=(16.5, 17.0), sharex=True, sharey=True)
    axes = list(axis_grid.flat[:5])
    legend_axis = axis_grid.flat[5]
    handles: dict[str, plt.Line2D] = {}
    split_value, compression = 0.4, 0.2

    def forward(values):
        values = np.asarray(values)
        return np.where(values <= split_value, values,
                        split_value + (values - split_value) * compression)

    def inverse(values):
        values = np.asarray(values)
        return np.where(values <= split_value, values,
                        split_value + (values - split_value) / compression)

    for index, area_group in enumerate(AREA_LABELS):
        axis = axes[index]
        subset = summary[summary["area_group"].astype(str) == area_group]
        for method in METHOD_ORDER:
            rows = subset[subset["method"] == method].sort_values("target_year")
            if rows.empty:
                continue
            years = rows["target_year"].to_numpy()
            mean = rows["mae_mean"].to_numpy()
            sem = rows["mae_sem"].fillna(0).to_numpy()
            line = axis.plot(
                years, mean, color=COLORS[method],
                linewidth=3.1 if method == "UHI-E3LLM" else 2.4,
                label=_display(method),
            )[0]
            axis.fill_between(
                years, np.clip(mean - sem, 0, 2.0), np.clip(mean + sem, 0, 2.0),
                color=COLORS[method], alpha=0.12, linewidth=0,
            )
            handles.setdefault(method, line)
        axis.set_title(AREA_TITLES[area_group], fontsize=36, pad=17)
        axis.set_xlim(2003.8, 2020.2)
        axis.set_xticks([2004, 2008, 2012, 2016, 2020])
        axis.set_xticks(np.arange(2004, 2021), minor=True)
        axis.set_yscale("function", functions=(forward, inverse))
        axis.set_ylim(0.05, 2.0)
        axis.set_yticks([0.1, 0.2, 0.3, 0.4, 0.8, 1.2, 1.6, 2.0])
        axis.axhline(
            split_value, color="#777777", linestyle=(0, (5, 4)),
            linewidth=1.2, alpha=0.70,
        )
        axis.set_xlabel("Year", fontsize=34, labelpad=11)
        if index in (0, 2, 4):
            axis.set_ylabel("Mean MAE", fontsize=34, labelpad=12)
        axis.tick_params(axis="both", which="major", labelsize=28)
        axis.tick_params(axis="x", which="minor", length=3, width=0.8)
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)

    legend_axis.axis("off")
    legend_axis.legend(
        [handles[method] for method in METHOD_ORDER],
        [_display(method) for method in METHOD_ORDER],
        loc="center", ncol=2, frameon=False, fontsize=28,
        handlelength=2.8, handletextpad=0.8, columnspacing=1.5, labelspacing=1.1,
    )
    figure.subplots_adjust(
        left=0.09, right=0.98, bottom=0.07, top=0.975, wspace=0.26, hspace=0.52
    )
    _save(figure, out_dir, "fig3b_all_results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-data-dir", type=Path, default=Path("outputs/figures/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument(
        "--city-metrics", type=Path,
        default=Path("data/city_metrics/city_metrics_test.csv"),
    )
    parser.add_argument(
        "--full-period-city-year",
        type=Path,
        default=None,
        help=(
            "Aligned eight-method city-year MAE table covering 2004-2020. "
            "Defaults to <figure-data-dir>/method_city_year_metrics_2004_2020.csv."
        ),
    )
    args = parser.parse_args()
    city_path = args.figure_data_dir / "method_city_metrics.csv"
    year_path = args.full_period_city_year or (
        args.figure_data_dir / "method_city_year_metrics_2004_2020.csv"
    )
    if not city_path.is_file() or not year_path.is_file():
        raise SystemExit(f"missing {city_path} or {year_path}; run figure_data.py first")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    city = pd.read_csv(city_path)
    city_year = attach_area_group(pd.read_csv(year_path), args.city_metrics)
    city_year.to_csv(args.figure_data_dir / "method_city_size_yearly_mae.csv", index=False)
    plot_fig3a(city, args.out_dir)
    plot_fig3b(city_year, args.out_dir)


if __name__ == "__main__":
    main()
