#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sequence_dataset import FEATURE_INDEX, read_city_series, save_city_series


SOURCE_COLUMNS = {
    "cloud_cover": ("ClimateData/climate_2020_12.csv", "Cloud_Cover_Mean"),
    "solar_radiation": ("ClimateData/climate_2020_12.csv", "Solar_Radiation_Flux"),
    "precipitation_flux": ("ClimateData/climate_2020_12.csv", "Precipitation_Flux"),
    "snow_thickness": ("ClimateData/climate_2020_12.csv", "Snow_Thickness_Mean"),
    "temperature": ("ClimateData/climate_2020_12.csv", "Temperature_Air_2m_Mean_24h"),
    "vapour_pressure": ("ClimateData/climate_2020_12.csv", "Vapour_Pressure_Mean"),
    "wind_speed": ("ClimateData/climate_2020_12.csv", "Wind_Speed_10m_Mean"),
    "evi": ("MOD13A3_EVI_NDVI/evi_ndvi_2020_12.csv", "EVI"),
    "ndvi": ("MOD13A3_EVI_NDVI/evi_ndvi_2020_12.csv", "NDVI"),
    "pm25": ("PM25/pm25_2020_12.csv", "b1"),
}


def load_month_values(
    raw_root: Path, expected_ids: set[int], month: str
) -> dict[str, dict[int, float]]:
    frames = {}
    for path, column in sorted(set(SOURCE_COLUMNS.values())):
        path = path.replace("2020_12", month)
        frame = pd.read_csv(raw_root / path, usecols=["id", column])
        if "id" not in frame or column not in frame:
            raise ValueError(f"{path}: missing id or {column}")
        ids = frame["id"].astype(int)
        if ids.duplicated().any() or set(ids) != expected_ids:
            raise ValueError(f"{path}: city IDs do not match the dataset")
        frames[(path, column)] = pd.Series(
            pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float32),
            index=ids.to_numpy(),
        )
    return {
        feature: frames[(source[0].replace("2020_12", month), source[1])].to_dict()
        for feature, source in SOURCE_COLUMNS.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--expected-cities", type=int, default=1020)
    parser.add_argument("--city-limit", type=int, default=0)
    args = parser.parse_args()

    metrics = pd.read_csv(
        args.data_root / "city_metrics" / "city_metrics_all.csv", usecols=["id"]
    )
    city_ids = sorted(metrics["id"].astype(int).tolist())
    if len(city_ids) != args.expected_cities or len(set(city_ids)) != len(city_ids):
        raise ValueError(f"expected {args.expected_cities} unique city IDs")
    expected_ids = set(city_ids)
    if args.city_limit:
        city_ids = city_ids[: args.city_limit]
    previous = load_month_values(args.raw_data, expected_ids, "2020_11")
    terminal = load_month_values(args.raw_data, expected_ids, "2020_12")
    for city_id in tqdm(city_ids, desc="Completing terminal month"):
        path = args.data_root / "valid" / f"UHI_city_{city_id}"
        year_month, values = read_city_series(path)
        if len(year_month) != 36 or year_month[-1] != 202012:
            raise ValueError(f"city {city_id}: validation sequence must end at 202012")
        for feature, city_values in previous.items():
            if not np.isclose(
                values[-2, FEATURE_INDEX[feature]],
                city_values[city_id],
                rtol=0.0,
                atol=1e-5,
            ):
                raise ValueError(f"city {city_id}: raw data mismatch for {feature} in 2020-11")
        for feature, city_values in terminal.items():
            values[-1, FEATURE_INDEX[feature]] = city_values[city_id]
        if not np.isfinite(values).all():
            raise ValueError(f"city {city_id}: terminal completion left missing values")
        save_city_series(path, year_month, values)


if __name__ == "__main__":
    main()
