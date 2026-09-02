#!/usr/bin/env python3
"""Build the three validated comparison tables reported in the manuscript."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "figures"))
from figure_data import (  # noqa: E402
    AREA_BINS,
    AREA_LABELS,
    METHOD_ORDER,
    collect,
    load_city_results_dir,
    load_sample_errors_dir,
    validate_collection,
)


REGIONS = [
    "Africa",
    "EastAsiaOceania",
    "Europe",
    "LatinAmerica",
    "NorthAmerica",
    "SCWAsia",
]
REGION_NAMES = {
    "Africa": "Africa",
    "EastAsiaOceania": "East Asia & Oceania",
    "Europe": "Europe",
    "LatinAmerica": "Latin America",
    "NorthAmerica": "North America",
    "SCWAsia": "West Asia",
}
REGION_TABLE_ORDER = [
    "Africa",
    "North America",
    "Europe",
    "Latin America",
    "East Asia & Oceania",
    "West Asia",
]
TABULAR_FILES = {
    "XGBoost": "xgb",
    "RandomForest": "rf",
    "LightGBM": "lgb",
    "MLP": "mlp",
}
TIMESERIES_DIRS = {"ARMD": "armd", "FilterNet": "filternet", "ST-MTM": "st_mtm"}
ABLATION_VARIANTS = [
    ("UHI-E3LLM", "main_ensemble/ensemble"),
    ("w/o UHI Knowledge Enhancer", "ablations/without_uhi_knowledge"),
    ("w/o Generalizability Enhancer", "ablations/without_generalizability"),
    ("w/o Explainability Enhancer", "ablations/without_explainability"),
    ("w/o LoRA fusion", "ablations/without_fusion"),
    ("w/o PPO", "ablations/without_ppo"),
    ("w/o ensemble inference", "ablations/without_ensemble"),
]
ABLATION_ORDER = [name for name, _ in ABLATION_VARIANTS]


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    error = frame["predicted_uhi"].to_numpy() - frame["true_uhi"].to_numpy()
    mse = float(np.mean(np.square(error)))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def validate_keyed_samples(
    samples: pd.DataFrame,
    expected_cities: int,
    expected_samples_per_city: int,
) -> None:
    validate_collection(
        samples,
        expected_cities=expected_cities,
        expected_samples_per_city=expected_samples_per_city,
        methods=METHOD_ORDER,
    )


def manuscript_table(
    long_frame: pd.DataFrame,
    group_column: str,
    group_order: list[str],
    row_order: list[str],
    row_column: str,
) -> pd.DataFrame:
    rows = []
    for name in row_order:
        row: dict[str, str | float] = {row_column: name}
        method_rows = long_frame.loc[long_frame[row_column] == name].set_index(
            group_column
        )
        for group in group_order:
            values = method_rows.loc[group]
            row[group] = (
                f"{values['MAE']:.4f}/"
                f"{values['MSE']:.4f}/"
                f"{values['RMSE']:.4f}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_main(data_dir: Path, output_root: Path) -> None:
    samples = collect(output_root, allow_missing=False)
    validate_keyed_samples(samples, expected_cities=100, expected_samples_per_city=36)

    city_info = pd.read_csv(data_dir / "city_metrics" / "city_metrics_test.csv")[["id", "Area"]]
    if city_info["id"].nunique() != 100:
        raise ValueError("main test manifest must contain exactly 100 unique cities")
    samples = samples.merge(city_info, left_on="city_idx", right_on="id", how="left")
    if samples["Area"].isna().any():
        raise ValueError("some predictions cannot be matched to test-city areas")
    samples["City size"] = pd.cut(
        samples["Area"], AREA_BINS, labels=AREA_LABELS, right=False, include_lowest=True
    )
    if samples["City size"].isna().any():
        raise ValueError("some test-city areas fall outside the configured area bins")

    rows = []
    group_order = ["Average", *AREA_LABELS]
    for method in METHOD_ORDER:
        method_samples = samples.loc[samples["method"] == method]
        for group in group_order:
            selected = (
                method_samples
                if group == "Average"
                else method_samples.loc[method_samples["City size"] == group]
            )
            row = metrics(selected)
            rows.append(
                {
                    "Method": method,
                    "City size": group,
                    "Cities": int(selected["city_idx"].nunique()),
                    "Samples": len(selected),
                    "MAE": row["mae"],
                    "MSE": row["mse"],
                    "RMSE": row["rmse"],
                }
            )

    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    long_frame = pd.DataFrame(rows)
    manuscript_table(
        long_frame, "City size", group_order, METHOD_ORDER, "Method"
    ).to_csv(table_dir / "table1_main_performance.csv", index=False)
    print(f"Wrote main comparison table below {table_dir}")


def load_regional_tabular(path: Path, method: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path).rename(
        columns={"city": "city_idx", "true": "true_uhi", "pred": "predicted_uhi"}
    )
    missing = {"target_year", "target_month"}.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{path} lacks {sorted(missing)}; rerun the regional tabular baselines"
        )
    frame["method"] = method
    error = frame["predicted_uhi"] - frame["true_uhi"]
    frame["absolute_error"] = error.abs()
    frame["squared_error"] = error.pow(2)
    return frame[
        [
            "method",
            "city_idx",
            "target_year",
            "target_month",
            "true_uhi",
            "predicted_uhi",
            "absolute_error",
            "squared_error",
        ]
    ]


def load_region(output_root: Path, region: str) -> pd.DataFrame:
    parts = [
        load_city_results_dir(
            output_root / "regional" / region / "ensemble" / "city_results", "UHI-E3LLM"
        )
    ]
    for method, directory in TIMESERIES_DIRS.items():
        parts.append(
            load_sample_errors_dir(
                output_root / "regional" / region / "baselines" / directory / "sample_errors",
                method,
            )
        )
    for method, filename in TABULAR_FILES.items():
        parts.append(
            load_regional_tabular(
                output_root
                / "regional"
                / "tabular_results"
                / "predictions"
                / filename
                / f"{region}_test_predictions.csv",
                method,
            )
        )
    return pd.concat(parts, ignore_index=True)


def build_regional(data_dir: Path, output_root: Path) -> None:
    rows = []
    for region in REGIONS:
        samples = load_region(output_root, region)
        expected = pd.read_csv(data_dir / "regional_folds" / f"fold_{region}_test.csv")
        expected_cities = expected["id"].nunique()
        expected_ids = set(expected["id"].astype(int))
        validate_collection(
            samples,
            expected_cities=expected_cities,
            expected_samples_per_city=36,
            methods=METHOD_ORDER,
        )

        for method in METHOD_ORDER:
            part = samples.loc[samples["method"] == method].copy()
            actual_ids = set(part["city_idx"].astype(int))
            if actual_ids != expected_ids:
                raise ValueError(f"{region}/{method}: city IDs differ from the fold manifest")
            row = metrics(part)
            rows.append(
                {
                    "Method": method,
                    "Region": REGION_NAMES[region],
                    "Region key": region,
                    "Cities": expected_cities,
                    "Samples": len(part),
                    "MAE": row["mae"],
                    "MSE": row["mse"],
                    "RMSE": row["rmse"],
                }
            )

    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    long_frame = pd.DataFrame(rows)
    group_order = REGION_TABLE_ORDER
    manuscript_table(
        long_frame, "Region", group_order, METHOD_ORDER, "Method"
    ).to_csv(table_dir / "table2_leave_one_region_out.csv", index=False)
    print(f"Wrote regional comparison table below {table_dir}")


def load_ablation_variant(root: Path, variant: str) -> pd.DataFrame:
    matches = sorted(
        path
        for path in root.rglob("city_results")
        if any(path.glob("city_*_predictions.csv"))
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"{variant}: expected one city_results directory below {root}; "
            f"found {matches}"
        )
    return load_city_results_dir(matches[0], variant)


def build_ablation(data_dir: Path, output_root: Path) -> None:
    parts = [
        load_ablation_variant(output_root / relative_path, variant)
        for variant, relative_path in ABLATION_VARIANTS
    ]
    samples = pd.concat(parts, ignore_index=True)
    validate_collection(
        samples,
        expected_cities=100,
        expected_samples_per_city=36,
        methods=ABLATION_ORDER,
    )

    city_info = pd.read_csv(
        data_dir / "city_metrics" / "city_metrics_test.csv"
    )[["id", "Area"]]
    if city_info["id"].nunique() != 100:
        raise ValueError("main test manifest must contain exactly 100 unique cities")
    samples = samples.merge(city_info, left_on="city_idx", right_on="id", how="left")
    if samples["Area"].isna().any():
        raise ValueError("some ablation predictions cannot be matched to test-city areas")
    samples["City size"] = pd.cut(
        samples["Area"],
        AREA_BINS,
        labels=AREA_LABELS,
        right=False,
        include_lowest=True,
    )
    if samples["City size"].isna().any():
        raise ValueError("some test-city areas fall outside the configured area bins")

    group_order = ["Average", *AREA_LABELS]
    rows = []
    for variant in ABLATION_ORDER:
        variant_samples = samples.loc[samples["method"] == variant]
        for group in group_order:
            selected = (
                variant_samples
                if group == "Average"
                else variant_samples.loc[variant_samples["City size"] == group]
            )
            row = metrics(selected)
            rows.append(
                {
                    "Method": variant,
                    "City size": group,
                    "Cities": int(selected["city_idx"].nunique()),
                    "Samples": len(selected),
                    "MAE": row["mae"],
                    "MSE": row["mse"],
                    "RMSE": row["rmse"],
                }
            )

    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    long_frame = pd.DataFrame(rows)
    manuscript_table(
        long_frame, "City size", group_order, ABLATION_ORDER, "Method"
    ).to_csv(table_dir / "table3_ablation.csv", index=False)
    print(f"Wrote ablation table below {table_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["main", "regional", "ablation", "all"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if args.mode in {"main", "all"}:
        build_main(args.data_dir, args.output_root)
    if args.mode in {"regional", "all"}:
        build_regional(args.data_dir, args.output_root)
    if args.mode in {"ablation", "all"}:
        build_ablation(args.data_dir, args.output_root)


if __name__ == "__main__":
    main()
