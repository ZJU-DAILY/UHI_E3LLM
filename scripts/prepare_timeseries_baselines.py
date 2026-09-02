#!/usr/bin/env python3
"""Create aligned multivariate CSV inputs for the time-series baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from datasets import concatenate_datasets, load_from_disk


FEATURE_COLUMNS = [
    "area",
    "built_height",
    "built_surface_density",
    "urbanization_level",
    "longitude",
    "latitude",
    "compactness",
    "population_density",
    "open_spaces",
    "water_surfaces",
    "road_surfaces",
    "residential",
    "non_residential",
    "cloud_cover",
    "solar_radiation",
    "precipitation_flux",
    "snow_thickness",
    "evi",
    "ndvi",
    "pm25",
    "temperature",
    "vapour_pressure",
    "wind_speed",
    "Amod2",
]

DYNAMIC_FIELDS = {
    "cloud_info": "cloud_cover",
    "solar_info": "solar_radiation",
    "precipitation_info": "precipitation_flux",
    "snow_info": "snow_thickness",
    "evi_info": "evi",
    "ndvi_info": "ndvi",
    "pm25_info": "pm25",
    "temperature_info": "temperature",
    "vapour_info": "vapour_pressure",
    "wind_info": "wind_speed",
}

EXPECTED_DATES = pd.period_range("2003-01", "2020-12", freq="M")
EXPECTED_TARGETS = pd.period_range("2004-01", "2020-12", freq="M")
SCHEMA_FILENAME = "multivariate_24.schema"


def _value(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value.item() if hasattr(value, "item") else value


def _series(value, expected: int) -> list[float]:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    result = [float(_value(item)) for item in value]
    if len(result) != expected:
        raise ValueError(f"Expected {expected} values, found {len(result)}")
    return result


def _target_period(row) -> pd.Period:
    return pd.Period(
        year=int(_value(row["year_info"])),
        month=int(_value(row["month_info"])),
        freq="M",
    )


def _assign(frame: pd.DataFrame, period: pd.Period, column: str, value: float) -> None:
    current = frame.at[period, column]
    if pd.notna(current) and not np.isclose(current, value, rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"Conflicting {column} value for {period}: {current} versus {value}"
        )
    frame.at[period, column] = value


def _structural_values(row) -> dict[str, float]:
    location = _series(row["location_info"], 2)
    land_use = _series(row["urban_land_use_info"], 5)
    return {
        "area": float(_value(row["area_info"])),
        "built_height": float(_value(row["built_height_info"])),
        "built_surface_density": float(_value(row["built_surface_info"])) * 100.0,
        "urbanization_level": float(_value(row["urbanization_level_info"])),
        "longitude": location[0],
        "latitude": location[1],
        "compactness": float(_value(row["compactness_info"])) / 1000.0,
        "population_density": float(_value(row["pop_info"])) * 100.0,
        "open_spaces": land_use[0],
        "water_surfaces": land_use[1],
        "road_surfaces": land_use[2],
        "residential": land_use[3],
        "non_residential": land_use[4],
    }


def reconstruct_from_processed(data_dir: Path, city_id: int) -> pd.DataFrame:
    train = load_from_disk(str(data_dir / "train" / f"UHI_city_{city_id}"))
    valid = load_from_disk(str(data_dir / "valid" / f"UHI_city_{city_id}"))
    dataset = concatenate_datasets([train, valid])
    rows = sorted(
        dataset,
        key=_target_period,
    )
    targets = [_target_period(row) for row in rows]
    if targets != list(EXPECTED_TARGETS):
        raise RuntimeError(f"City {city_id} does not contain all 2004-2020 targets")

    frame = pd.DataFrame(index=EXPECTED_DATES, columns=FEATURE_COLUMNS, dtype=float)
    for row in rows:
        target = _target_period(row)
        history_periods = pd.period_range(target - 12, target - 1, freq="M")
        for period, value in zip(history_periods, _series(row["x"], 12)):
            _assign(frame, period, "Amod2", value)
        _assign(frame, target, "Amod2", float(_value(row["y"])))

        for source, column in DYNAMIC_FIELDS.items():
            for period, value in zip(history_periods, _series(row[source], 12)):
                _assign(frame, period, column, value)
        for column, value in _structural_values(row).items():
            _assign(frame, target, column, value)

    structural_columns = FEATURE_COLUMNS[:13]
    frame.loc[frame.index < EXPECTED_TARGETS[0], structural_columns] = (
        frame.loc[EXPECTED_TARGETS[0], structural_columns].to_numpy()
    )
    for column in DYNAMIC_FIELDS.values():
        missing = frame.index[frame[column].isna()].tolist()
        if missing not in ([], [EXPECTED_DATES[-1]]):
            raise RuntimeError(f"City {city_id} has missing {column} months: {missing}")
        frame[column] = frame[column].ffill()

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
