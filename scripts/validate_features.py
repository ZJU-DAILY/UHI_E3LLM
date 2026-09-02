#!/usr/bin/env python3
"""Validate every processed feature across all cities and samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "baseline"))

from common import flatten_sample  # noqa: E402
from metrics import target_month_from_sample  # noqa: E402
from utils import format_numeric_plain_prompt, format_prompt  # noqa: E402


FIELD_COUNTS = {
    "year_info": 1,
    "month_info": 1,
    "x": 12,
    "area_info": 1,
    "built_height_info": 1,
    "built_surface_info": 1,
    "urbanization_level_info": 1,
    "compactness_info": 1,
    "urban_land_use_info": 5,
    "location_info": 2,
    "pop_info": 1,
    "cloud_info": 12,
    "solar_info": 12,
    "precipitation_info": 12,
    "snow_info": 12,
    "evi_info": 12,
    "ndvi_info": 12,
    "pm25_info": 12,
    "temperature_info": 12,
    "vapour_info": 12,
    "wind_info": 12,
    "y": 1,
    "city_idx": 1,
}

PROMPT_LABELS = [
    "[UHI Data:]",
    "[Area:]",
    "[Built Height:]",
    "[Built Surface Area:]",
    "[Urbanization Level:]",
    "[City Compactness:]",
    "[Urban Land Use:]",
    "[Location:]",
    "[Population Density:]",
    "[Average Cloud Cover:]",
    "[Solar Radiation:]",
    "[Precipitation Flux:]",
    "[Snow Thickness:]",
    "[Enhanced Vegetation Index (EVI):]",
    "[Normalized Difference Vegetation Index (NDVI):]",
    "[PM2.5 Concentration:]",
    "[Average Temperature:]",
    "[Average Vapour Pressure:]",
    "[Average Wind Speed:]",
    "[Result:]",
]


def flatten(value):
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(flatten(item))
        return output
    if hasattr(value, "item"):
        value = value.item()
    return [value]


def empty_stats():
    return {
        field: {"count": 0, "minimum": math.inf, "maximum": -math.inf, "nonfinite": 0}
        for field in FIELD_COUNTS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_cities", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=100)
    args = parser.parse_args()

    all_path = args.data_dir / "city_metrics/city_metrics_all.csv"
    if not all_path.exists():
        all_path = args.data_dir / "city_metrics/city_metrics.csv"
    all_metrics = pd.read_csv(all_path)
    city_ids = all_metrics["id"].astype(int).tolist()
    if args.max_cities:
        city_ids = city_ids[: args.max_cities]

    stats = empty_stats()
    errors = []
    split_rows = {"train": 0, "valid": 0}
    prompts_checked = 0
    numeric_prompts_checked = 0
    flattened_rows_checked = 0
    valid_target_months = {}

    for city_index, city_id in enumerate(city_ids, start=1):
        city_valid_targets = set()
        for split, expected_rows in [("train", 168), ("valid", 36)]:
            path = args.data_dir / split / f"UHI_city_{city_id}"
            if not path.is_dir():
                errors.append(f"missing dataset: {path}")
                continue
            dataset = load_from_disk(str(path))
            if len(dataset) != expected_rows:
                errors.append(
                    f"city {city_id} {split}: expected {expected_rows} rows, found {len(dataset)}"
                )
            split_rows[split] += len(dataset)

            for row_index, row in enumerate(dataset):
                missing = set(FIELD_COUNTS).difference(row)
                if missing:
                    errors.append(f"city {city_id} {split} row {row_index}: missing {sorted(missing)}")
                    continue
                for field, expected_count in FIELD_COUNTS.items():
                    values = flatten(row[field])
                    if len(values) != expected_count:
                        errors.append(
                            f"city {city_id} {split} row {row_index} {field}: "
                            f"expected {expected_count} values, found {len(values)}"
                        )
                        continue
                    numeric = np.asarray(values, dtype=float)
                    finite = np.isfinite(numeric)
                    current = stats[field]
                    current["count"] += len(numeric)
                    current["nonfinite"] += int((~finite).sum())
                    if finite.any():
                        current["minimum"] = min(current["minimum"], float(numeric[finite].min()))
                        current["maximum"] = max(current["maximum"], float(numeric[finite].max()))

                features, _ = flatten_sample(row)
                if features.shape != (147,):
                    errors.append(
                        f"city {city_id} {split} row {row_index}: flattened shape {features.shape}"
                    )
                flattened_rows_checked += 1

                try:
                    format_numeric_plain_prompt(row)
                    numeric_prompts_checked += 1
                except Exception as exc:
                    errors.append(
                        f"city {city_id} {split} row {row_index}: numeric prompt: {exc}"
                    )

                if row_index == 0:
                    try:
                        prompt = format_prompt(row)
                        missing_labels = [label for label in PROMPT_LABELS if label not in prompt]
                        if missing_labels:
                            errors.append(
                                f"city {city_id} {split}: prompt missing labels {missing_labels}"
                            )
                        prompts_checked += 1
                    except Exception as exc:
                        errors.append(f"city {city_id} {split}: text prompt: {exc}")

                if split == "valid":
                    city_valid_targets.add(target_month_from_sample(row))

        expected_targets = {
            (year, month)
            for year in range(2018, 2021)
            for month in range(1, 13)
        }
        if city_valid_targets != expected_targets:
            errors.append(
                f"city {city_id}: valid target months differ from 2018-01/2020-12"
            )
        valid_target_months[str(city_id)] = len(city_valid_targets)

        if args.progress_every and city_index % args.progress_every == 0:
            print(f"validated {city_index}/{len(city_ids)} cities", flush=True)

    for field, current in stats.items():
        if current["minimum"] == math.inf:
            current["minimum"] = None
            current["maximum"] = None

    summary = {
        "data_dir": str(args.data_dir.resolve()),
        "cities_checked": len(city_ids),
        "split_rows": split_rows,
        "expected_total_rows": len(city_ids) * (168 + 36),
        "flattened_rows_checked": flattened_rows_checked,
        "flattened_feature_count": 147,
        "text_prompts_checked": prompts_checked,
        "numeric_prompts_checked": numeric_prompts_checked,
        "all_valid_cities_have_36_target_months": all(
            count == 36 for count in valid_target_months.values()
        ),
        "field_statistics": stats,
        "error_count": len(errors),
        "errors": errors[:200],
        "manual_review_required": [
            "built_surface_info physical unit",
            "pop_info count-versus-density semantics and unit",
            "compactness_info definition and unit",
            "source-product provenance for every field",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in [
        "cities_checked", "split_rows", "flattened_rows_checked",
        "text_prompts_checked", "numeric_prompts_checked", "error_count"
    ]}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
