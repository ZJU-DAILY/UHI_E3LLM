#!/usr/bin/env python3
"""Average aligned per-city predictions from multiple checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["city_idx", "target_year", "target_month", "target_year_month"]


def locate_city_results(root: Path) -> Path:
    matches = sorted(root.rglob("city_results"))
    matches = [path for path in matches if any(path.glob("city_*_predictions.csv"))]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one city_results directory below {root}; found {matches}")
    return matches[0]


def load_checkpoint(root: Path, checkpoint_index: int) -> pd.DataFrame:
    city_dir = locate_city_results(root)
    frames = [pd.read_csv(path) for path in sorted(city_dir.glob("city_*_predictions.csv"))]
    data = pd.concat(frames, ignore_index=True)
    required = set(KEYS + ["true_uhi", "predicted_uhi"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{city_dir} is missing columns: {sorted(missing)}")
    selected = data[KEYS + ["true_uhi", "predicted_uhi"]]
    if selected.duplicated(KEYS).any():
        raise ValueError(f"{city_dir} contains duplicate prediction keys.")
    return selected.rename(
        columns={"predicted_uhi": f"prediction_{checkpoint_index}"}
    )


def prediction_keys(data: pd.DataFrame) -> set[tuple]:
    return set(data[KEYS].itertuples(index=False, name=None))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cities", type=int, default=0)
    parser.add_argument("--expected-samples-per-city", type=int, default=36)
    args = parser.parse_args()
    if args.expected_cities < 0:
        parser.error("--expected-cities must be non-negative")
    if args.expected_samples_per_city <= 0:
        parser.error("--expected-samples-per-city must be positive")

    merged = load_checkpoint(args.input[0], 0)
    expected_keys = prediction_keys(merged)
    for index, root in enumerate(args.input[1:], start=1):
        current = load_checkpoint(root, index)
        current_keys = prediction_keys(current)
        if current_keys != expected_keys:
            missing = len(expected_keys - current_keys)
            unexpected = len(current_keys - expected_keys)
            raise RuntimeError(
                f"Prediction keys differ for {root}: missing={missing}, "
                f"unexpected={unexpected}."
            )
        merged = merged.merge(current, on=KEYS + ["true_uhi"], how="inner", validate="one_to_one")

    prediction_columns = [column for column in merged if column.startswith("prediction_")]
    city_count = merged["city_idx"].nunique()
    expected_cities = args.expected_cities or city_count
    expected_rows = expected_cities * args.expected_samples_per_city
    per_city_counts = merged.groupby("city_idx").size()
    wrong_city_counts = per_city_counts[
        per_city_counts != args.expected_samples_per_city
    ].to_dict()
    if (
        city_count != expected_cities
        or len(merged) != expected_rows
        or wrong_city_counts
    ):
        raise RuntimeError(
            f"Expected {expected_cities} cities and {expected_rows} aligned targets; "
            f"found {city_count} cities and {len(merged)} rows. "
            f"Wrong per-city counts: {wrong_city_counts}."
        )
    merged["predicted_uhi"] = merged[prediction_columns].mean(axis=1)
    error = merged["predicted_uhi"] - merged["true_uhi"]
    merged["absolute_error"] = error.abs()
    merged["squared_error"] = error.pow(2)

    args.output.mkdir(parents=True, exist_ok=True)
    city_dir = args.output / "city_results"
    city_dir.mkdir(exist_ok=True)
    for city_id, city in merged.groupby("city_idx", sort=True):
        city.to_csv(city_dir / f"city_{city_id}_predictions.csv", index=False)

    mse = float(merged["squared_error"].mean())
    summary = pd.DataFrame(
        [
            {
                "checkpoints": len(prediction_columns),
                "cities": merged["city_idx"].nunique(),
                "samples": len(merged),
                "mae": merged["absolute_error"].mean(),
                "mse": mse,
                "rmse": np.sqrt(mse),
            }
        ]
    )
    merged.to_csv(args.output / "ensemble_predictions.csv", index=False)
    summary.to_csv(args.output / "ensemble_metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
