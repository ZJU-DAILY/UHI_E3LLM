#!/usr/bin/env python3
"""Collect per-sample test predictions for every compared method.

Reads the workflow outputs produced by the release pipeline and normalizes
them into one tidy table used by the manuscript figure scripts:

    method, city_idx, target_year, target_month, true_uhi, predicted_uhi,
    absolute_error, squared_error

Sources (relative to --output-root, default ``outputs``):

    UHI-E3LLM     main_ensemble/ensemble/city_results/city_*.csv
    XGBoost       baselines/tabular/predictions/xgb_test_predictions.csv
    RandomForest  baselines/tabular/predictions/rf_test_predictions.csv
    LightGBM      baselines/tabular/predictions/lgb_test_predictions.csv
    MLP           baselines/tabular/predictions/mlp_test_predictions.csv
    ARMD          baselines/armd/sample_errors/city_*.csv
    FilterNet     baselines/filternet/sample_errors/city_*.csv
    ST-MTM        baselines/st_mtm/sample_errors/city_*.csv

No numerical value is hard-coded; every figure quantity is recomputed from
these prediction files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = [
    "UHI-E3LLM",
    "ARMD",
    "FilterNet",
    "ST-MTM",
    "XGBoost",
    "RandomForest",
    "LightGBM",
    "MLP",
]

# Proposed method in red, time-series baselines in oranges, tabular in blues.
METHOD_COLORS = {
    "UHI-E3LLM": "#B2182B",
    "ARMD": "#E8952F",
    "FilterNet": "#F2B366",
    "ST-MTM": "#C97B1D",
    "XGBoost": "#4E79A7",
    "RandomForest": "#6FA8DC",
    "LightGBM": "#2C5F8A",
    "MLP": "#8FBBD9",
}

DISPLAY_NAMES = {"RandomForest": "Random Forest"}

AREA_BINS = [100, 150, 450, 800, 1500, np.inf]
AREA_LABELS = ["Very small", "Small", "Medium", "Large", "Very large"]

# Manuscript case-study cities (Fig. 4/5), ordered very small -> very large.
CASE_CITIES = [
    (864, "Norton Shores, United States"),
    (677, "Solna, Sweden"),
    (222, "Wenzhou, China"),
    (661, "Madrid, Spain"),
    (413, "Delhi, India"),
]
CASE_COLORS = {
    864: "#F2B10E",
    677: "#4E79A7",
    222: "#CB3A2E",
    661: "#7B52A1",
    413: "#2E8B57",
}

COLUMNS = [
    "method",
    "city_idx",
    "target_year",
    "target_month",
    "true_uhi",
    "predicted_uhi",
    "absolute_error",
    "squared_error",
]


def safe_name(method: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in method)


def display_name(method: str) -> str:
    return DISPLAY_NAMES.get(method, method)


def _finalize(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["method"] = method
    error = frame["predicted_uhi"] - frame["true_uhi"]
    frame["absolute_error"] = error.abs()
    frame["squared_error"] = error.pow(2)
    frame["city_idx"] = frame["city_idx"].astype(int)
    frame["target_year"] = frame["target_year"].astype(int)
    frame["target_month"] = frame["target_month"].astype(int)
    return frame[COLUMNS]


def load_city_results_dir(path: Path, method: str) -> pd.DataFrame:
    files = sorted(path.glob("city_*_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"no city_*_predictions.csv below {path}")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return _finalize(
        frame[["city_idx", "target_year", "target_month", "true_uhi", "predicted_uhi"]],
        method,
    )


def load_tabular_predictions(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"target_year", "target_month"}.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{path} lacks {sorted(missing)}; rerun baseline/tabular_baselines.py "
            "(the release version stores target dates in the prediction files)"
        )
    frame = frame.rename(
        columns={"city": "city_idx", "true": "true_uhi", "pred": "predicted_uhi"}
    )
    return _finalize(frame, method)


def load_sample_errors_dir(path: Path, method: str) -> pd.DataFrame:
    files = sorted(path.glob("city_*.csv"))
    if not files:
        raise FileNotFoundError(f"no city_*.csv below {path}")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame = frame.rename(
        columns={"city_id": "city_idx", "true": "true_uhi", "pred": "predicted_uhi"}
    )
    period = frame["target_year_month"].astype(str).str.extract(
        r"^(?P<year>\d{4})\D?(?P<month>\d{2})$"
    )
    if period.isna().any().any():
        invalid = frame.loc[period.isna().any(axis=1), "target_year_month"].head(5)
        raise ValueError(f"invalid target_year_month values in {path}: {invalid.tolist()}")
    frame["target_year"] = period["year"].astype(int)
    frame["target_month"] = period["month"].astype(int)
    return _finalize(frame, method)


def method_sources(output_root: Path) -> dict[str, tuple[str, Path]]:
    tabular = output_root / "baselines" / "tabular" / "predictions"
    return {
        "UHI-E3LLM": ("city_results", output_root / "main_ensemble" / "ensemble" / "city_results"),
        "ARMD": ("sample_errors", output_root / "baselines" / "armd" / "sample_errors"),
        "FilterNet": ("sample_errors", output_root / "baselines" / "filternet" / "sample_errors"),
        "ST-MTM": ("sample_errors", output_root / "baselines" / "st_mtm" / "sample_errors"),
        "XGBoost": ("tabular", tabular / "xgb_test_predictions.csv"),
        "RandomForest": ("tabular", tabular / "rf_test_predictions.csv"),
        "LightGBM": ("tabular", tabular / "lgb_test_predictions.csv"),
        "MLP": ("tabular", tabular / "mlp_test_predictions.csv"),
    }


LOADERS = {
    "city_results": load_city_results_dir,
    "tabular": load_tabular_predictions,
    "sample_errors": load_sample_errors_dir,
}


def collect(output_root: Path, allow_missing: bool = False) -> pd.DataFrame:
    parts = []
    for method in METHOD_ORDER:
        kind, path = method_sources(output_root)[method]
        try:
            part = LOADERS[kind](path, method)
        except (FileNotFoundError, ValueError) as error:
            if allow_missing:
                print(f"[skip] {method}: {error}")
                continue
            raise
        print(f"[ok] {method}: {len(part)} samples, {part['city_idx'].nunique()} cities")
        parts.append(part)
    if not parts:
        raise SystemExit("no method predictions found; run the pipeline workflows first")
    return pd.concat(parts, ignore_index=True)


def validate_collection(
    samples: pd.DataFrame,
    expected_cities: int = 100,
    expected_samples_per_city: int = 36,
    methods: list[str] | None = None,
) -> None:
    """Require identical complete city-month targets and labels for all methods."""
    methods = methods or METHOD_ORDER
    key_columns = ["city_idx", "target_year", "target_month"]
    expected_samples = expected_cities * expected_samples_per_city
    period_bounds = {
        36: (2018, 2020),
        204: (2004, 2020),
    }
    expected_period = None
    if expected_samples_per_city in period_bounds:
        start_year, end_year = period_bounds[expected_samples_per_city]
        expected_period = {
            (year, month)
            for year in range(start_year, end_year + 1)
            for month in range(1, 13)
        }
    reference = None
    reference_method = methods[0]

    for method in methods:
        part = samples.loc[samples["method"] == method].copy()
        numeric_columns = [
            "true_uhi",
            "predicted_uhi",
            "absolute_error",
            "squared_error",
        ]
        finite = np.isfinite(part[numeric_columns].to_numpy(dtype=float))
        if not finite.all():
            bad_rows = part.loc[~finite.all(axis=1), key_columns].head().to_dict("records")
            raise ValueError(f"{method}: non-finite prediction or metric values at {bad_rows}")
        if len(part) != expected_samples:
            raise ValueError(
                f"{method}: expected {expected_samples} samples, found {len(part)}"
            )
        if part["city_idx"].nunique() != expected_cities:
            raise ValueError(
                f"{method}: expected {expected_cities} cities, "
                f"found {part['city_idx'].nunique()}"
            )
        counts = part.groupby("city_idx").size()
        if not counts.eq(expected_samples_per_city).all():
            bad = counts[counts.ne(expected_samples_per_city)].head().to_dict()
            raise ValueError(f"{method}: invalid per-city sample counts: {bad}")
        if part.duplicated(key_columns).any():
            raise ValueError(f"{method}: duplicate city/month prediction keys")
        observed_period = set(
            part[["target_year", "target_month"]].drop_duplicates().itertuples(
                index=False, name=None
            )
        )
        if expected_period is not None and observed_period != expected_period:
            start_year, end_year = period_bounds[expected_samples_per_city]
            raise ValueError(
                f"{method}: target period is not exactly "
                f"{start_year}-01 through {end_year}-12"
            )

        keyed = part[key_columns + ["true_uhi"]].sort_values(key_columns).reset_index(drop=True)
        if reference is None:
            reference = keyed
            continue
        if not keyed[key_columns].equals(reference[key_columns]):
            raise ValueError(f"{method}: prediction keys differ from {reference_method}")
        if not np.allclose(keyed["true_uhi"], reference["true_uhi"], atol=1e-6):
            raise ValueError(f"{method}: true labels differ from {reference_method}")


def city_metrics(samples: pd.DataFrame) -> pd.DataFrame:
    grouped = samples.groupby(["method", "city_idx"])
    frame = grouped.agg(
        samples=("true_uhi", "size"),
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
    ).reset_index()
    frame["rmse"] = np.sqrt(frame["mse"])
    return frame


def city_year_metrics(samples: pd.DataFrame) -> pd.DataFrame:
    grouped = samples.groupby(["method", "city_idx", "target_year"])
    frame = grouped.agg(
        samples=("true_uhi", "size"),
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
    ).reset_index()
    frame["rmse"] = np.sqrt(frame["mse"])
    return frame


def attach_area_group(frame: pd.DataFrame, city_metrics_csv: Path) -> pd.DataFrame:
    info = pd.read_csv(city_metrics_csv)[["id", "Area"]]
    merged = frame.merge(info, left_on="city_idx", right_on="id", how="left")
    if merged["Area"].isna().any():
        missing = sorted(merged.loc[merged["Area"].isna(), "city_idx"].unique())
        raise ValueError(f"cities missing from {city_metrics_csv}: {missing[:10]}")
    merged["area_group"] = pd.cut(
        merged["Area"], bins=AREA_BINS, labels=AREA_LABELS,
        right=False, include_lowest=True,
    )
    return merged.drop(columns=["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--figure-data-dir",
        type=Path,
        default=None,
        help="Defaults to <output-root>/figures/data.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--expected-cities", type=int, default=100)
    parser.add_argument("--expected-samples-per-city", type=int, default=36)
    args = parser.parse_args()

    data_dir = args.figure_data_dir or args.output_root / "figures" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    samples = collect(args.output_root, allow_missing=args.allow_missing)
    present_methods = [method for method in METHOD_ORDER if method in set(samples["method"])]
    validate_collection(
        samples,
        expected_cities=args.expected_cities,
        expected_samples_per_city=args.expected_samples_per_city,
        methods=present_methods,
    )
    samples.to_csv(data_dir / "method_sample_errors.csv", index=False)
    city_metrics(samples).to_csv(data_dir / "method_city_metrics.csv", index=False)
    city_year_metrics(samples).to_csv(
        data_dir / "method_city_year_metrics.csv", index=False
    )
    print(f"Wrote figure data below {data_dir}")


if __name__ == "__main__":
    main()
