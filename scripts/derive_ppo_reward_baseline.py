#!/usr/bin/env python3
"""Derive the PPO reward baseline e_bar from post-SFT training inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"city_idx", "samples", "mae", "mse", "rmse"}

SPLIT_SPECS = {
    "train": {
        "statistic": "sample-weighted training MAE",
        "role": "post-fusion SFT training targets before PPO",
        "target_start": "2004-01",
        "target_end": "2017-12",
        "default_samples_per_city": 168,
    },
    "validation": {
        "statistic": "sample-weighted validation MAE",
        "role": "post-fusion SFT validation before PPO",
        "target_start": "2018-01",
        "target_end": "2020-12",
        "default_samples_per_city": 36,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_inference(
    inference_csv: Path,
    *,
    split: str = "train",
    expected_cities: int,
    expected_samples_per_city: int,
) -> dict[str, object]:
    if split not in SPLIT_SPECS:
        raise ValueError(f"split must be one of {sorted(SPLIT_SPECS)}; got {split}")
    if expected_cities <= 0 or expected_samples_per_city <= 0:
        raise ValueError("expected city and sample counts must be positive")
    if not inference_csv.is_file():
        raise FileNotFoundError(f"baseline inference CSV not found: {inference_csv}")

    frame = pd.read_csv(inference_csv)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"baseline inference CSV is missing columns: {missing}")
    if len(frame) != expected_cities:
        raise ValueError(
            f"expected {expected_cities} {split} cities, found {len(frame)}"
        )
    if frame["city_idx"].isna().any() or frame["city_idx"].duplicated().any():
        raise ValueError(f"{split} city IDs must be non-null and unique")

    samples = pd.to_numeric(frame["samples"], errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(samples)) or not np.all(samples == np.floor(samples)):
        raise ValueError(f"{split} sample counts must be finite integers")
    if not np.all(samples == expected_samples_per_city):
        bad = frame.loc[
            samples != expected_samples_per_city, ["city_idx", "samples"]
        ]
        raise ValueError(
            f"each {split} city must have {expected_samples_per_city} samples; "
            f"mismatches: {bad.head(10).to_dict(orient='records')}"
        )

    metrics = {}
    for column in ("mae", "mse", "rmse"):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{split} {column} values must be finite and non-negative")
        metrics[column] = values

    if not np.allclose(
        np.square(metrics["rmse"]), metrics["mse"], rtol=1e-5, atol=1e-8
    ):
        raise ValueError("per-city RMSE values are inconsistent with MSE")

    total_samples = int(samples.sum())
    expected_samples = expected_cities * expected_samples_per_city
    if total_samples != expected_samples:
        raise ValueError(
            f"expected {expected_samples} {split} predictions, found {total_samples}"
        )

    spec = SPLIT_SPECS[split]
    baseline_mae = float(np.average(metrics["mae"], weights=samples))
    baseline_mse = float(np.average(metrics["mse"], weights=samples))
    baseline_rmse = math.sqrt(baseline_mse)
    return {
        "schema_version": 2,
        "statistic": spec["statistic"],
        "reward_symbol": "e_bar",
        # The key name y_bar is kept for workflow compatibility; the value is
        # the manuscript reference statistic e_bar.
        "y_bar": baseline_mae,
        "metrics": {
            "mae": baseline_mae,
            "mse": baseline_mse,
            "rmse": baseline_rmse,
        },
        "split": {
            "name": split,
            "role": spec["role"],
            "cities": expected_cities,
            "samples_per_city": expected_samples_per_city,
            "samples": total_samples,
            "target_start": spec["target_start"],
            "target_end": spec["target_end"],
            "held_out_test_cities_used": False,
        },
        "source_inference_csv": str(inference_csv.resolve()),
        "source_inference_sha256": _sha256(inference_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_SPECS),
        default="train",
        help=(
            "Dataset split scored by the post-SFT model. The manuscript reward "
            "statistic e_bar is the training MAE."
        ),
    )
    parser.add_argument("--expected-cities", type=int, default=500)
    parser.add_argument(
        "--expected-samples-per-city",
        type=int,
        default=None,
        help="Defaults to 168 for the train split and 36 for validation.",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"reward-baseline report already exists: {args.output}; pass --overwrite"
        )
    expected_samples_per_city = (
        args.expected_samples_per_city
        if args.expected_samples_per_city is not None
        else SPLIT_SPECS[args.split]["default_samples_per_city"]
    )
    report = summarize_inference(
        args.inference_csv,
        split=args.split,
        expected_cities=args.expected_cities,
        expected_samples_per_city=expected_samples_per_city,
    )
    report["model"] = args.model or None
    report["dataset_path"] = (
        str(args.dataset_path.resolve()) if args.dataset_path else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        f"Derived PPO reward baseline e_bar={report['y_bar']:.12g} from "
        f"{report['split']['samples']} {args.split} predictions."
    )


if __name__ == "__main__":
    main()
