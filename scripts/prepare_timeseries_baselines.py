#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sequence_dataset import FEATURE_NAMES, load_city_matrix


FEATURE_COLUMNS = list(FEATURE_NAMES)

EXPECTED_DATES = pd.period_range("2003-01", "2020-12", freq="M")
EXPECTED_TARGETS = pd.period_range("2004-01", "2020-12", freq="M")
SCHEMA_FILENAME = "multivariate_24.schema"


def reconstruct_from_processed(data_dir: Path, city_id: int) -> pd.DataFrame:
    year_month, values = load_city_matrix(data_dir, city_id)
    expected = np.asarray(
        [period.year * 100 + period.month for period in EXPECTED_DATES],
        dtype=np.int32,
    )
    if not np.array_equal(year_month, expected):
        raise RuntimeError(f"City {city_id} does not contain 2003-2020 monthly data")
    frame = pd.DataFrame(values, index=EXPECTED_DATES, columns=FEATURE_COLUMNS)
    frame.loc[:, FEATURE_COLUMNS[13:]] = frame.loc[:, FEATURE_COLUMNS[13:]].ffill()
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        missing = frame.columns[frame.isna().any()].tolist()
        raise RuntimeError(f"City {city_id} has incomplete multivariate data: {missing}")
    result = frame.reset_index(names="date")
    result["date"] = result["date"].astype(str)
    if len(result) != 216:
        raise RuntimeError(f"Failed to reconstruct 2003-2020 series for city {city_id}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_city_metrics", type=Path)
    parser.add_argument("--test_city_metrics", type=Path)
    args = parser.parse_args()
    train_metrics = args.train_city_metrics or args.data_dir / "city_metrics/city_metrics_train.csv"
    test_metrics = args.test_city_metrics or args.data_dir / "city_metrics/city_metrics_test.csv"
    train_ids = pd.read_csv(train_metrics)["id"].astype(int).tolist()
    test_ids = pd.read_csv(test_metrics)["id"].astype(int).tolist()
    overlap = set(train_ids).intersection(test_ids)
    if overlap:
        raise ValueError(f"Train/test city manifests overlap: {sorted(overlap)[:5]}")

    for split in ["all", "train", "valid"]:
        (args.output_dir / split).mkdir(parents=True, exist_ok=True)
    schema_path = args.output_dir / SCHEMA_FILENAME
    schema_path.unlink(missing_ok=True)

    frames = {
        city_id: reconstruct_from_processed(args.data_dir, city_id)
        for city_id in sorted(set(train_ids).union(test_ids))
    }

    for city_id, data in frames.items():
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values("date", inplace=True)
        data.reset_index(drop=True, inplace=True)
        if data["date"].duplicated().any():
            raise ValueError(f"Duplicate monthly timestamps for city {city_id}")
        expected_dates = pd.date_range("2003-01-01", "2020-12-01", freq="MS")
        if len(data) != len(expected_dates) or not data["date"].equals(
            pd.Series(expected_dates, name="date")
        ):
            raise ValueError(
                f"City {city_id} must contain every month from 2003-01 through 2020-12"
            )
        if data["Amod2"].isna().any():
            raise ValueError(f"Missing UHI values for city {city_id}")
        filename = f"UHI_city_{city_id}.csv"
        data.to_csv(args.output_dir / "all" / filename, index=False)
        data[data["date"].dt.year <= 2017].to_csv(
            args.output_dir / "train" / filename, index=False
        )
        data[data["date"].dt.year.between(2018, 2020)].to_csv(
            args.output_dir / "valid" / filename, index=False
        )
    schema_path.write_text(
        ",".join(["date", *FEATURE_COLUMNS]) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {len(train_ids)} training and {len(test_ids)} test city series "
        f"with {len(FEATURE_COLUMNS)} channels in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
