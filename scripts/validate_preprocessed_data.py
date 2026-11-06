#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validate_regional_fold import REGION_NAMES, validate_fold
from sequence_dataset import (
    FEATURE_NAMES,
    FEATURE_INDEX,
    DYNAMIC_FIELDS,
    load_city_dataset,
    read_city_series,
)


CITY_PATTERN = re.compile(r"UHI_city_(\d+)")
CANONICAL_DATASET_FILES = {
    "data-00000-of-00001.arrow",
    "dataset_info.json",
    "state.json",
}


def discover_city_ids(directory: Path) -> set[int]:
    ids: set[int] = set()
    for path in directory.iterdir():
        match = CITY_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            ids.add(int(match.group(1)))
    return ids


def expected_months(start_year: int, start_month: int, count: int) -> list[tuple[int, int]]:
    start = start_year * 12 + start_month - 1
    return [((start + i) // 12, (start + i) % 12 + 1) for i in range(count)]


def sample_date(sample: dict) -> tuple[int, int]:
    year = int(np.asarray(sample["year_info"]).ravel()[0])
    month = int(np.asarray(sample["month_info"]).ravel()[0])
    return year, month


def validate_city_dataset(data_dir: Path, city_id: int) -> None:
    sequences = {}
    date_expectations = {
        "train": np.asarray([
            year * 100 + month
            for year, month in expected_months(2003, 1, 180)
        ], dtype=np.int32),
        "valid": np.asarray([
            year * 100 + month
            for year, month in expected_months(2018, 1, 36)
        ], dtype=np.int32),
    }
    for split, expected_dates in date_expectations.items():
        path = data_dir / split / f"UHI_city_{city_id}"
        files = {entry.name for entry in path.iterdir() if entry.is_file()}
        if files != CANONICAL_DATASET_FILES:
            raise ValueError(
                f"{path}: expected only {sorted(CANONICAL_DATASET_FILES)}, "
                f"found {sorted(files)}"
            )
        dates, values = read_city_series(path)
        if not np.array_equal(dates, expected_dates):
            raise ValueError(f"{path}: month coverage is incorrect")
        if values.shape != (len(expected_dates), len(FEATURE_NAMES)):
            raise ValueError(f"{path}: tensor shape is incorrect: {values.shape}")
        sequences[split] = values

    train_values, valid_values = sequences["train"], sequences["valid"]
    if not np.isfinite(train_values).all():
        raise ValueError(f"city {city_id}: training tensor contains non-finite values")
    if not np.isfinite(valid_values[:, :13]).all():
        raise ValueError(f"city {city_id}: validation structural tensor is incomplete")
    if not np.isfinite(valid_values[:, FEATURE_INDEX["Amod2"]]).all():
        raise ValueError(f"city {city_id}: validation UHI sequence is incomplete")
    for column in DYNAMIC_FIELDS.values():
        channel = FEATURE_INDEX[column]
        if not np.isfinite(valid_values[:, channel]).all():
            raise ValueError(f"city {city_id}: unexpected missing values in {column}")

    ordered = list(load_city_dataset(data_dir, city_id, "train")) + list(
        load_city_dataset(data_dir, city_id, "valid")
    )
    expected_targets = expected_months(2004, 1, 204)
    actual_targets = [sample_date(row) for row in ordered]
    if actual_targets != expected_targets:
        raise ValueError(f"city {city_id}: target-month coverage is incorrect")
    for current, following in zip(ordered, ordered[1:]):
        current_x = np.asarray(current["x"], dtype=float).ravel()
        following_x = np.asarray(following["x"], dtype=float).ravel()
        current_y = float(np.asarray(current["y"], dtype=float).ravel()[0])
        if len(current_x) != 12 or len(following_x) != 12:
            raise ValueError(f"city {city_id}: historical UHI window must contain 12 values")
        if not np.allclose(current_x[1:], following_x[:-1], atol=1e-5, rtol=0.0):
            raise ValueError(f"city {city_id}: adjacent historical windows are misaligned")
        if not np.isclose(current_y, following_x[-1], atol=1e-5, rtol=0.0):
            raise ValueError(f"city {city_id}: target/history alignment is incorrect")


def validate_manifests(data_dir: Path, expected_cities: int) -> set[int]:
    metrics_dir = data_dir / "city_metrics"
    all_cities = pd.read_csv(metrics_dir / "city_metrics_all.csv")
    train = pd.read_csv(metrics_dir / "city_metrics_train.csv")
    test = pd.read_csv(metrics_dir / "city_metrics_test.csv")
    calibration = pd.read_csv(metrics_dir / "calibration_cities.csv")
    assignments = pd.read_csv(metrics_dir / "region_assignments_600.csv")
    all_ids = set(all_cities["id"].astype(int))
    train_ids = set(train["id"].astype(int))
    test_ids = set(test["id"].astype(int))
    calibration_ids = set(calibration["city_id"].astype(int))
    if len(all_cities) != expected_cities or len(all_ids) != expected_cities:
        raise ValueError(f"city_metrics_all.csv must contain {expected_cities} unique cities")
    if len(train_ids) != 500 or len(test_ids) != 100 or len(calibration_ids) != 100:
        raise ValueError("expected 500 train, 100 test, and 100 calibration cities")
    if train_ids & test_ids or (train_ids | test_ids) & calibration_ids:
        raise ValueError("train, test, and calibration city cohorts overlap")
    if not (train_ids | test_ids | calibration_ids).issubset(all_ids):
        raise ValueError("a selected city is absent from city_metrics_all.csv")
    if len(assignments) != 600 or set(assignments["id"].astype(int)) != train_ids | test_ids:
        raise ValueError("region_assignments_600.csv does not match the modeling cohort")

    fold_root = data_dir / "regional_folds"
    for region in REGION_NAMES:
        validate_fold(
            region,
            fold_root / f"fold_{region}_train.csv",
            fold_root / f"fold_{region}_test.csv",
            metrics_dir / "region_assignments_600.csv",
        )
    return all_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--expected-cities", type=int, default=1020)
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Load every city dataset and verify all 204 target-month alignments.",
    )
    args = parser.parse_args()

    all_ids = validate_manifests(args.data_dir, args.expected_cities)
    train_ids = discover_city_ids(args.data_dir / "train")
    valid_ids = discover_city_ids(args.data_dir / "valid")
    if train_ids != all_ids or valid_ids != all_ids:
        raise ValueError("train/valid city directories do not match city_metrics_all.csv")

    unwanted = [
        path
        for split in ("train", "valid")
        for path in (args.data_dir / split).rglob("*")
        if path.is_file()
        and (path.name.startswith("cache-") or path.name.startswith("tmp"))
    ]
    if unwanted:
        raise ValueError(f"generated cache/temp files must not be released: {unwanted[:5]}")

    if args.deep:
        for index, city_id in enumerate(sorted(all_ids), start=1):
            validate_city_dataset(args.data_dir, city_id)
            if index % 100 == 0 or index == len(all_ids):
                print(f"validated city datasets: {index}/{len(all_ids)}")
    print(
        f"Processed data are valid: {len(all_ids)} cities, 500/100 modeling split, "
        "100 calibration cities, and six regional folds."
    )


if __name__ == "__main__":
    main()
