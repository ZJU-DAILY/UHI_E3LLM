#!/usr/bin/env python3
"""Build method-specific empirical q0.90 prediction intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


EXPECTED_PERIOD = {(year, month) for year in range(2018, 2021) for month in range(1, 13)}


def read_predictions(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)
    files = sorted(path.rglob("city_*_predictions.csv"))
    if not files:
        files = sorted(path.rglob("city_*.csv"))
    if not files:
        raise FileNotFoundError(path)
    frames = []
    for file in files:
        frame = pd.read_csv(file)
        if not {"city_idx", "city_id", "city"}.intersection(frame.columns):
            match = re.search(r"city_(\d+)", file.stem)
            if match:
                frame.insert(0, "city_idx", int(match.group(1)))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def read_manifest_ids(path: Path) -> set[int]:
    manifest = pd.read_csv(path)
    id_column = "city_id" if "city_id" in manifest.columns else "id"
    if id_column not in manifest.columns:
        raise ValueError(f"City manifest has no city_id/id column: {path}")
    values = pd.to_numeric(manifest[id_column], errors="raise").astype(int)
    if values.duplicated().any():
        raise ValueError(f"City manifest contains duplicate IDs: {path}")
    return set(values)


def normalize_predictions(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = frame.copy()
    if "city_idx" not in frame.columns and "city_id" in frame.columns:
        frame.rename(columns={"city_id": "city_idx"}, inplace=True)
    frame.rename(
        columns={"city": "city_idx", "true": "true_uhi", "pred": "predicted_uhi"},
        inplace=True,
    )
    missing = {"city_idx", "true_uhi", "predicted_uhi"}.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} predictions missing {sorted(missing)}")

    for column in ("city_idx", "true_uhi", "predicted_uhi"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[["city_idx", "true_uhi", "predicted_uhi"]].to_numpy()).all():
        raise ValueError(f"{name} predictions contain non-finite IDs, labels, or predictions")
    if not np.equal(frame["city_idx"], np.floor(frame["city_idx"])).all():
        raise ValueError(f"{name} predictions contain non-integer city IDs")
    frame["city_idx"] = frame["city_idx"].astype(int)

    if {"target_year", "target_month"}.issubset(frame.columns):
        years = pd.to_numeric(frame["target_year"], errors="raise").astype(int)
        months = pd.to_numeric(frame["target_month"], errors="raise").astype(int)
    elif "target_year_month" in frame.columns:
        compact = frame["target_year_month"].astype(str).str.replace(r"\.0$", "", regex=True)
        numeric_mask = compact.str.fullmatch(r"\d{6}")
        if numeric_mask.all():
            numeric = compact.astype(int)
            years = numeric // 100
            months = numeric % 100
        else:
            parsed = pd.to_datetime(compact, errors="raise")
            years = parsed.dt.year.astype(int)
            months = parsed.dt.month.astype(int)
    else:
        raise ValueError(
            f"{name} predictions must contain target_year/target_month or target_year_month"
        )
    frame["target_year"] = years
    frame["target_month"] = months
    if not frame["target_month"].between(1, 12).all():
        raise ValueError(f"{name} predictions contain invalid target months")
    keys = ["city_idx", "target_year", "target_month"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{name} predictions contain duplicate city-month keys")
    return frame


def validate_city_periods(
    frame: pd.DataFrame,
    name: str,
    expected_ids: set[int],
    expected_samples_per_city: int,
) -> None:
    observed_ids = set(frame["city_idx"])
    if observed_ids != expected_ids:
        raise ValueError(
            f"{name} city IDs differ from the fixed manifest: "
            f"missing={sorted(expected_ids - observed_ids)}, "
            f"unexpected={sorted(observed_ids - expected_ids)}"
        )
    if expected_samples_per_city != len(EXPECTED_PERIOD):
        raise ValueError(
            f"This workflow requires {len(EXPECTED_PERIOD)} monthly targets per city"
        )
    for city_id, city in frame.groupby("city_idx"):
        periods = set(zip(city["target_year"], city["target_month"]))
        if periods != EXPECTED_PERIOD:
            raise ValueError(
                f"{name} city {city_id} does not contain exactly 2018-01 through 2020-12"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_predictions", type=Path, required=True)
    parser.add_argument("--test_predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--expected_calibration_cities", type=int, default=100)
    parser.add_argument("--expected_test_cities", type=int, default=100)
    parser.add_argument("--expected_samples_per_city", type=int, default=36)
    parser.add_argument("--calibration_city_manifest", type=Path, required=True)
    parser.add_argument("--test_city_manifest", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.quantile < 1.0:
        parser.error("--quantile must be strictly between 0 and 1")

    calibration = normalize_predictions(
        read_predictions(args.calibration_predictions), "calibration"
    )
    test = normalize_predictions(read_predictions(args.test_predictions), "test")

    calibration_manifest_ids = read_manifest_ids(args.calibration_city_manifest)
    test_manifest_ids = read_manifest_ids(args.test_city_manifest)
    if len(calibration_manifest_ids) != args.expected_calibration_cities:
        raise ValueError(
            f"Expected {args.expected_calibration_cities} calibration manifest cities; "
            f"found {len(calibration_manifest_ids)}"
        )
    if len(test_manifest_ids) != args.expected_test_cities:
        raise ValueError(
            f"Expected {args.expected_test_cities} test manifest cities; "
            f"found {len(test_manifest_ids)}"
        )
    manifest_overlap = calibration_manifest_ids.intersection(test_manifest_ids)
    if manifest_overlap:
        raise ValueError(f"Calibration/test manifest overlap: {sorted(manifest_overlap)}")
    validate_city_periods(
        calibration,
        "calibration",
        calibration_manifest_ids,
        args.expected_samples_per_city,
    )
    validate_city_periods(
        test, "test", test_manifest_ids, args.expected_samples_per_city
    )

    calibration_ids = set(calibration["city_idx"].astype(int))
    test_ids = set(test["city_idx"].astype(int))
    overlap = calibration_ids.intersection(test_ids)
    if overlap:
        raise ValueError(f"Calibration/test city overlap: {sorted(overlap)}")
    if len(calibration_ids) != args.expected_calibration_cities:
        raise ValueError(
            f"Expected {args.expected_calibration_cities} calibration cities; "
            f"found {len(calibration_ids)}"
        )
    if len(test_ids) != args.expected_test_cities:
        raise ValueError(
            f"Expected {args.expected_test_cities} test cities; found {len(test_ids)}"
        )
    calibration["absolute_error"] = (
        calibration["predicted_uhi"] - calibration["true_uhi"]
    ).abs()
    q_value = float(np.quantile(calibration["absolute_error"], args.quantile))
    test["interval_lower"] = test["predicted_uhi"] - q_value
    test["interval_upper"] = test["predicted_uhi"] + q_value
    test["covered"] = test["true_uhi"].between(
        test["interval_lower"], test["interval_upper"], inclusive="both"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(args.output_dir / "calibration_errors.csv", index=False)
    test.to_csv(args.output_dir / "test_prediction_intervals.csv", index=False)
    city_coverage = (
        test.groupby("city_idx", as_index=False)["covered"]
        .mean()
        .rename(columns={"covered": "coverage"})
    )
    city_coverage.to_csv(args.output_dir / "test_city_coverage.csv", index=False)
    summary = {
        "quantile": args.quantile,
        "q_value": q_value,
        "calibration_cities": len(calibration_ids),
        "calibration_samples": len(calibration),
        "test_cities": len(test_ids),
        "test_samples": len(test),
        "test_coverage": float(test["covered"].mean()),
    }
    (args.output_dir / "prediction_interval_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
