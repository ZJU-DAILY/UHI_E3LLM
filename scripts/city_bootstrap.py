#!/usr/bin/env python3
"""City-level bootstrap confidence intervals for MAE, MSE, and RMSE."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


def read_predictions(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)
    files = sorted(path.rglob("city_*_predictions.csv"))
    if not files:
        files = sorted(path.rglob("city_*.csv"))
    if not files:
        raise FileNotFoundError(f"No city prediction files below {path}")
    frames = []
    for file in files:
        frame = pd.read_csv(file)
        if not {"city_idx", "city_id", "city"}.intersection(frame.columns):
            match = re.search(r"city_(\d+)", file.stem)
            if match:
                frame.insert(0, "city_idx", int(match.group(1)))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_cities", type=int, default=100)
    parser.add_argument("--expected_samples_per_city", type=int, default=36)
    args = parser.parse_args()

    data = read_predictions(args.predictions)
    if "city_idx" not in data.columns and "city_id" in data.columns:
        data = data.rename(columns={"city_id": "city_idx"})
    data = data.rename(
        columns={"city": "city_idx", "true": "true_uhi", "pred": "predicted_uhi"}
    )
    required = {"city_idx", "true_uhi", "predicted_uhi"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    city_ids = np.array(sorted(data["city_idx"].unique()))
    city_counts = data.groupby("city_idx").size()
    if (
        len(city_ids) != args.expected_cities
        or not city_counts.eq(args.expected_samples_per_city).all()
    ):
        bad = city_counts[city_counts.ne(args.expected_samples_per_city)].to_dict()
        raise ValueError(
            f"Expected {args.expected_cities} cities and "
            f"{args.expected_samples_per_city} samples per city; found "
            f"{len(city_ids)} cities. Wrong counts: {bad}"
        )
    by_city = {city: data[data["city_idx"] == city] for city in city_ids}
    rng = np.random.default_rng(args.seed)

    rows = []
    for replicate in range(args.replicates):
        sampled = rng.choice(city_ids, size=len(city_ids), replace=True)
        current = pd.concat([by_city[city] for city in sampled], ignore_index=True)
        error = current["predicted_uhi"] - current["true_uhi"]
        mse = float(np.square(error).mean())
        rows.append(
            {
                "replicate": replicate,
                "mae": float(np.abs(error).mean()),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
            }
        )
    samples = pd.DataFrame(rows)
    summary_rows = []
    for metric in ["mae", "mse", "rmse"]:
        summary_rows.append(
            {
                "metric": metric.upper(),
                "bootstrap_mean": samples[metric].mean(),
                "ci90_lower": samples[metric].quantile(0.05),
                "ci90_upper": samples[metric].quantile(0.95),
                "replicates": args.replicates,
                "seed": args.seed,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.output_dir / "bootstrap_replicates.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "bootstrap_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
