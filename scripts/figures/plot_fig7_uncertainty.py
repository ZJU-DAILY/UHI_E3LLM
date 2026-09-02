#!/usr/bin/env python3
"""Render the final Fig. 7a bootstrap and Fig. 7b calibration-error ECDF."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_data import safe_name  # noqa: E402


METHOD_ORDER = [
    "UHI-E3LLM",
    "LightGBM",
    "RandomForest",
    "XGBoost",
    "MLP",
    "FilterNet",
    "ARMD",
    "ST-MTM",
]
DISPLAY_NAMES = {
    "UHI-E3LLM": r"UHI-E$^3$LLM",
    "RandomForest": "Random Forest",
}
COLORS = {
    "UHI-E3LLM": "#B2182B",
    "LightGBM": "#2166AC",
    "RandomForest": "#4393C3",
    "XGBoost": "#92C5DE",
    "MLP": "#67A9CF",
    "FilterNet": "#E08214",
    "ARMD": "#FDB863",
    "ST-MTM": "#B35806",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
        "mathtext.fontset": "stix",
        "font.size": 20,
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


def _require_bootstrap(bootstrap_root: Path) -> dict[str, pd.DataFrame]:
    replicates = {}
    missing = []
    for method in METHOD_ORDER:
        path = bootstrap_root / safe_name(method) / "bootstrap_replicates.csv"
        if not path.is_file():
            missing.append(str(path))
            continue
        table = pd.read_csv(path)
        required = {"mae", "mse", "rmse"}
        absent = required.difference(table.columns)
        if absent:
            raise ValueError(f"{path} is missing columns: {sorted(absent)}")
        replicates[method] = table
    if missing:
        raise FileNotFoundError("Incomplete bootstrap outputs; missing:\n" + "\n".join(missing))
    return replicates


def _require_calibration_errors(calibration_root: Path) -> pd.DataFrame:
    frames = []
    missing = []
    for method in METHOD_ORDER:
        path = calibration_root / safe_name(method) / "calibration_errors.csv"
        if not path.is_file():
            missing.append(str(path))
            continue
        table = pd.read_csv(path)
        required = {"city_idx", "absolute_error"}
        absent = required.difference(table.columns)
        if absent:
            raise ValueError(f"{path} is missing columns: {sorted(absent)}")
        table = table.copy()
        table["city_idx"] = pd.to_numeric(table["city_idx"], errors="raise").astype(int)
        table["absolute_error"] = pd.to_numeric(
            table["absolute_error"], errors="raise"
        )
        counts = table.groupby("city_idx").size()
        if len(table) != 3600 or table["city_idx"].nunique() != 100 or not counts.eq(36).all():
            raise ValueError(
                f"{path} must contain 100 calibration cities x 36 samples; "
                f"found {len(table)} rows, {table['city_idx'].nunique()} cities, "
                f"and per-city counts {sorted(counts.unique())}"
            )
        table.insert(0, "method", method)
        frames.append(table)
    if missing:
        raise FileNotFoundError(
            "Incomplete calibration outputs; missing:\n" + "\n".join(missing)
        )
    return pd.concat(frames, ignore_index=True)


def plot_fig7a(bootstrap_root: Path, out_dir: Path, data_dir: Path) -> None:
    replicates = _require_bootstrap(bootstrap_root)
    arrays = [replicates[method]["mae"].to_numpy() for method in METHOD_ORDER]
    y = np.arange(len(METHOD_ORDER))
    figure, axis = plt.subplots(figsize=(12.4, 7.2))
    parts = axis.violinplot(
        arrays, positions=y, vert=False, widths=0.72,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body, method in zip(parts["bodies"], METHOD_ORDER):
        body.set_facecolor(COLORS[method])
        body.set_edgecolor("none")
        body.set_alpha(0.86 if method == "UHI-E3LLM" else 0.72)
    violin_right = np.array(
        [float(body.get_paths()[0].vertices[:, 0].max()) for body in parts["bodies"]]
    )
    box = axis.boxplot(
        arrays, positions=y, vert=False, widths=0.16, patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.1},
        whiskerprops={"color": "#4D4D4D", "linewidth": 0.9},
        capprops={"color": "#4D4D4D", "linewidth": 0.9},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("#4D4D4D")
        patch.set_alpha(0.9)
    medians = np.array([np.median(values) for values in arrays])
    q05 = np.array([np.quantile(values, 0.05) for values in arrays])
    q95 = np.array([np.quantile(values, 0.95) for values in arrays])
    axis.scatter(medians, y, s=18, color="#111111", zorder=5)
    for yi, low, high in zip(y, q05, q95):
        axis.hlines(yi, low, high, color="#222222", linewidth=1.15, zorder=4)
        axis.vlines(
            [low, high], yi - 0.10, yi + 0.10,
            color="#222222", linewidth=1.0, zorder=4,
        )
    label_x = np.maximum(q95, violin_right) + 0.035
    for index, (yi, low, high, text_x) in enumerate(zip(y, q05, q95, label_x)):
        last = index == len(METHOD_ORDER) - 1
        axis.text(
            1.98 if last else text_x,
            yi + 0.62 if last else yi,
            f"90% CI: [{low:.3f}, {high:.3f}]",
            ha="right" if last else "left",
            va="center",
            fontsize=20,
            color="#222222",
        )
    axis.set_yticks(y)
    axis.set_yticklabels([_display(method) for method in METHOD_ORDER])
    axis.get_yticklabels()[0].set_fontweight("bold")
    axis.invert_yaxis()
    axis.set_ylim(len(METHOD_ORDER) + 0.25, -0.5)
    axis.set_xlabel("MAE", fontsize=30)
    axis.set_xlim(0.0, 2.0)
    axis.xaxis.set_major_locator(MultipleLocator(0.2))
    axis.grid(axis="x", color="#D8D8D8", linewidth=0.8, linestyle=(0, (3, 3)))
    axis.grid(axis="y", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", labelsize=24)
    figure.tight_layout()
    _save(figure, out_dir, "fig7a_bootstrap_uncertainty")

    summary = pd.DataFrame(
        {
            "method": METHOD_ORDER,
            "mae_mean": [values.mean() for values in arrays],
            "mae_ci90_low": q05,
            "mae_ci90_high": q95,
        }
    )
    summary.to_csv(data_dir / "fig7a_bootstrap_summary_all_methods.csv", index=False)


def plot_fig7b(calibration_root: Path, out_dir: Path, data_dir: Path) -> None:
    errors = _require_calibration_errors(calibration_root)
    errors.to_csv(
        data_dir / "fig7b_calibration_absolute_errors_all_methods.csv", index=False
    )

    figure, axis = plt.subplots(figsize=(12.0, 7.5))
    x_limit = 3.0
    for method in METHOD_ORDER:
        values = np.sort(
            errors.loc[errors["method"] == method, "absolute_error"].to_numpy()
        )
        cdf = np.arange(1, len(values) + 1, dtype=float) / len(values)
        x_grid = np.concatenate([[0.0], values])
        y_grid = np.concatenate([[0.0], cdf])
        axis.plot(
            x_grid, y_grid, color=COLORS[method],
            linewidth=2.8 if method == "UHI-E3LLM" else 1.8,
            alpha=1.0 if method == "UHI-E3LLM" else 0.88,
            label=_display(method),
        )
    axis.axhline(0.90, color="#555555", linewidth=1.15,
                 linestyle=(0, (4, 3)), alpha=0.8)
    axis.text(x_limit * 0.99, 0.895, "0.90", ha="right", va="top",
              color="#555555", fontsize=20)
    axis.set_xlim(-0.06, x_limit)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("Absolute error", fontsize=30)
    axis.set_ylabel("Empirical cumulative probability", fontsize=30)
    axis.set_xticks(np.arange(0.0, x_limit + 0.001, 0.25))
    axis.set_yticks(np.arange(0.0, 1.001, 0.2))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)
    axis.tick_params(axis="both", direction="out", length=7, width=1.2,
                     labelsize=24, pad=8)
    axis.legend(
        frameon=False, loc="lower right", bbox_to_anchor=(0.985, 0.035),
        borderaxespad=0.0, handlelength=2.4, fontsize=22, ncol=1,
    )
    figure.tight_layout()
    _save(figure, out_dir, "fig7b_calibration_error_ecdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--figure-data-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or args.output_root / "figures"
    data_dir = args.figure_data_dir or args.output_root / "figures" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_fig7a(args.output_root / "uncertainty" / "bootstrap", out_dir, data_dir)
    plot_fig7b(args.output_root / "uncertainty" / "calibration", out_dir, data_dir)


if __name__ == "__main__":
    main()
