#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets, load_from_disk
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sequence_dataset import save_city_series, samples_to_series


UNUSED_CITY_METRICS = {
    "Log Area", "Anisometry", "Shape Index (MSI)",
    "Edge Density (ED)", "Geodesic Area",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--city-limit", type=int, default=0)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if input_root == output_root:
        raise ValueError("input and output dataset roots must differ")
    if (output_root / "train").exists() or (output_root / "valid").exists():
        raise FileExistsError(f"output already contains train/valid data: {output_root}")

    metrics = pd.read_csv(
        input_root / "city_metrics" / "city_metrics_all.csv", usecols=["id"]
    )
    city_ids = metrics["id"].astype(int).tolist()
    if len(city_ids) != 1020 or len(set(city_ids)) != 1020:
        raise ValueError(f"expected 1020 unique cities, found {len(city_ids)}")
    if args.city_limit:
        city_ids = city_ids[: args.city_limit]

    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("city_metrics", "regional_folds", "explanation_samples"):
        source = input_root / name
        if source.exists():
            shutil.copytree(source, output_root / name)
            if name == "city_metrics":
                for path in (output_root / name).rglob("*.csv"):
                    columns = pd.read_csv(path, nrows=0).columns
                    if not UNUSED_CITY_METRICS.intersection(columns):
                        continue
                    frame = pd.read_csv(
                        path, dtype=str, keep_default_na=False,
                        usecols=lambda column: column not in UNUSED_CITY_METRICS,
                    )
                    temporary = path.with_name(path.name + ".tmp")
                    frame.to_csv(temporary, index=False)
                    temporary.replace(path)

    for city_id in tqdm(city_ids, desc="Converting city sequences"):
        parts = [
            load_from_disk(str(input_root / split / f"UHI_city_{city_id}"))
            for split in ("train", "valid")
        ]
        samples = concatenate_datasets(parts)
        if len(samples) != 204:
            raise ValueError(f"city {city_id}: expected 204 targets, found {len(samples)}")
        year_month, values = samples_to_series(samples, city_id)
        save_city_series(output_root / "train" / f"UHI_city_{city_id}", year_month[:180], values[:180])
        save_city_series(output_root / "valid" / f"UHI_city_{city_id}", year_month[180:], values[180:])

    print(f"Converted {len(city_ids)} cities to {output_root}")


if __name__ == "__main__":
    main()
