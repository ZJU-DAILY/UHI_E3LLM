#!/usr/bin/env python3
"""Create an inference view for the fixed independent calibration cities."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        raise RuntimeError(f"Refusing to replace non-symlink path: {path}")
    path.symlink_to(target.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_cities", type=int, default=100)
    args = parser.parse_args()

    selected = pd.read_csv(args.data_dir / "city_metrics/calibration_cities.csv")
    if "city_id" not in selected:
        raise ValueError("calibration_cities.csv must contain city_id")
    selected_ids = selected["city_id"].astype(int).tolist()
    if len(selected_ids) != args.expected_cities or len(set(selected_ids)) != args.expected_cities:
        raise ValueError(f"Expected {args.expected_cities} unique calibration city IDs")

    train_ids = set(
        pd.read_csv(args.data_dir / "city_metrics/city_metrics_train.csv")["id"].astype(int)
    )
    test_ids = set(
        pd.read_csv(args.data_dir / "city_metrics/city_metrics_test.csv")["id"].astype(int)
    )
    overlap = set(selected_ids).intersection(train_ids.union(test_ids))
    if overlap:
        raise ValueError(f"Calibration/modeling overlap: {sorted(overlap)}")

    all_metrics = pd.read_csv(args.data_dir / "city_metrics/city_metrics_all.csv")
    calibration_metrics = all_metrics[all_metrics["id"].isin(selected_ids)].copy()
    observed_ids = set(calibration_metrics["id"].astype(int))
    if observed_ids != set(selected_ids):
        missing = sorted(set(selected_ids).difference(observed_ids))
        unexpected = sorted(observed_ids.difference(selected_ids))
        raise ValueError(
            f"Calibration city metadata mismatch: missing={missing}, unexpected={unexpected}"
        )
    order = {city_id: index for index, city_id in enumerate(selected_ids)}
    calibration_metrics["_order"] = calibration_metrics["id"].map(order)
    calibration_metrics.sort_values("_order", inplace=True)
    calibration_metrics.drop(columns="_order", inplace=True)

    metrics_dir = args.output_dir / "city_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    calibration_metrics.to_csv(metrics_dir / "city_metrics_test.csv", index=False)
    pd.read_csv(args.data_dir / "city_metrics/city_metrics_train.csv").to_csv(
        metrics_dir / "city_metrics_train.csv", index=False
    )
    replace_symlink(args.output_dir / "train", args.data_dir / "train")
    replace_symlink(args.output_dir / "valid", args.data_dir / "valid")
    print(f"Prepared {args.expected_cities} independent calibration cities in {args.output_dir}")


if __name__ == "__main__":
    main()
