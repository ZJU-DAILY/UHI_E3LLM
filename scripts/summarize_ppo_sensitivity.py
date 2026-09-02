#!/usr/bin/env python3
"""Collect independently evaluated PPO-lambda metrics into one table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambdas", nargs="+", type=int, default=[0, 1, 5, 10, 20])
    parser.add_argument("--expected_cities", type=int, default=500)
    parser.add_argument("--expected_samples_per_city", type=int, default=36)
    args = parser.parse_args()
    if args.expected_cities <= 0 or args.expected_samples_per_city <= 0:
        parser.error("expected city and per-city sample counts must be positive")

    rows = []
    reference_city_ids = None
    for penalty_lambda in args.lambdas:
        run_dir = args.root / f"ppo_lambda_{penalty_lambda}"
        prediction_file = run_dir / "ensemble" / "ensemble_predictions.csv"
        if not prediction_file.is_file():
            raise RuntimeError(
                f"Missing two-checkpoint ensemble predictions for lambda={penalty_lambda}: "
                f"{prediction_file}"
            )
        predictions = pd.read_csv(prediction_file)
        required = {
            "city_idx",
            "target_year",
            "target_month",
            "true_uhi",
            "predicted_uhi",
        }
        missing = required.difference(predictions.columns)
        if missing:
            raise RuntimeError(
                f"lambda={penalty_lambda} ensemble is missing {sorted(missing)}"
            )
        for column in required:
            predictions[column] = pd.to_numeric(predictions[column], errors="raise")
        numeric = predictions[list(required)].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"lambda={penalty_lambda} contains non-finite predictions")
        keys = ["city_idx", "target_year", "target_month"]
        if predictions.duplicated(keys).any():
            raise RuntimeError(f"lambda={penalty_lambda} contains duplicate city-month keys")
        city_counts = predictions.groupby("city_idx").size()
        if (
            len(city_counts) != args.expected_cities
            or not city_counts.eq(args.expected_samples_per_city).all()
        ):
            raise RuntimeError(f"Incomplete lambda={penalty_lambda} ensemble")
        city_ids = set(city_counts.index.astype(int))
        if reference_city_ids is None:
            reference_city_ids = city_ids
        elif city_ids != reference_city_ids:
            raise RuntimeError(
                f"lambda={penalty_lambda} evaluates a different city set"
            )
        error = predictions["predicted_uhi"] - predictions["true_uhi"]
        mae = float(error.abs().mean())
        mse = float(error.pow(2).mean())
        rows.append(
            {
                "lambda": penalty_lambda,
                "cities": len(city_counts),
                "samples": len(predictions),
                "mae": mae,
                "mse": mse,
                "rmse": mse**0.5,
                "source": str(prediction_file),
            }
        )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
