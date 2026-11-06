#!/usr/bin/env python3

from __future__ import annotations

import argparse
from bisect import bisect_right
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sequence_dataset import (
    FEATURE_INDEX,
    load_city_dataset,
    save_city_series,
    samples_to_series,
)


GHS_EPOCHS = (2000, 2005, 2010, 2015, 2020)
FIRST_YEAR = 2003
MONTHS_TOTAL = 216
FIRST_TARGET_INDEX = 12

LAND_USE_COLUMNS = [
    "Open Spaces",
    "Water Surfaces",
    "Road Surfaces",
    "Residential",
    "Non-Residential",
]
LAND_USE_CLASS_GROUPS = {
    "Open Spaces": ["1", "2", "3"],
    "Water Surfaces": ["4"],
    "Road Surfaces": ["5"],
    "Residential": ["11", "12", "13", "14", "15"],
    "Non-Residential": ["21", "22", "23", "24", "25"],
}
CLIMATE_COLUMNS = {
    "wind_info": "Wind_Speed_10m_Mean",
    "temperature_info": "Temperature_Air_2m_Mean_24h",
    "cloud_info": "Cloud_Cover_Mean",
    "precipitation_info": "Precipitation_Flux",
    "snow_info": "Snow_Thickness_Mean",
    "vapour_info": "Vapour_Pressure_Mean",
    "solar_info": "Solar_Radiation_Flux",
}
VEGETATION_COLUMNS = {"ndvi_info": "NDVI", "evi_info": "EVI"}
PM25_COLUMNS = {"pm25_info": "b1"}


def date_from_index(index: int) -> tuple[int, int]:
    return FIRST_YEAR + index // 12, index % 12 + 1


def epoch_index(target_year: int) -> int:
    index = bisect_right(GHS_EPOCHS, target_year) - 1
    if index < 0:
        raise ValueError(f"no GHS epoch at or before {target_year}")
    return index


def read_indexed(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    usecols = None if columns is None else ["id", *columns]
    frame = pd.read_csv(path, usecols=usecols)
    if "id" not in frame.columns:
        raise ValueError(f"{path}: missing id column")
    frame = frame.drop(
        columns=[c for c in frame.columns if c.startswith("Unnamed")]
    )
    return frame.set_index("id", verify_integrity=True)


def load_monthly_series(
    directory: Path, pattern: str, *, columns: list[str] | None = None
) -> list[pd.DataFrame]:
    regex = re.compile(pattern)
    dated = {}
    for path in directory.iterdir():
        match = regex.fullmatch(path.name)
        if match:
            dated[(int(match.group(1)), int(match.group(2)))] = path
    expected = [date_from_index(index) for index in range(MONTHS_TOTAL)]
    missing = [date for date in expected if date not in dated]
    if missing:
        raise ValueError(f"{directory}: missing monthly files, e.g. {missing[:3]}")
    return [read_indexed(dated[date], columns) for date in expected]


def load_epoch_series(directory: Path) -> list[pd.DataFrame]:
    frames = []
    for epoch in GHS_EPOCHS:
        candidates = [
            path
            for path in directory.iterdir()
            if path.suffix == ".csv" and str(epoch) in path.name
        ]
        if len(candidates) != 1:
            raise ValueError(f"{directory}: expected one CSV for epoch {epoch}")
        frames.append(read_indexed(candidates[0]))
    return frames


def land_use_table(path: Path) -> pd.DataFrame:
    frame = read_indexed(path)
    missing = [c for c in LAND_USE_COLUMNS if c not in frame.columns]
    if missing:
        for group, classes in LAND_USE_CLASS_GROUPS.items():
            absent = [c for c in classes if c not in frame.columns]
            if absent:
                raise ValueError(
                    f"{path}: cannot aggregate land use; missing classes {absent}"
                )
            frame[group] = frame[classes].sum(axis=1)
    return frame[LAND_USE_COLUMNS]


def raw_uhi_values(raw_data: Path, city_id: int) -> np.ndarray:
    path = raw_data / "Extracted_City_Data" / f"UHI_city_{city_id}.csv"
    frame = pd.read_csv(path)
    column = "Amod2" if "Amod2" in frame.columns else "AMod2"
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    if len(values) != MONTHS_TOTAL:
        raise ValueError(
            f"{path}: expected {MONTHS_TOTAL} monthly values, found {len(values)}"
        )
    return values


class RawSources:
    def __init__(self, raw_data: Path, city_metrics: Path):
        self.raw_data = raw_data
        self.city_info = read_indexed(
            city_metrics, ["centroid_x", "centroid_y", "Area", "Compactness (CI)"]
        )
        self.climate = load_monthly_series(
            raw_data / "ClimateData", r"climate_(\d{4})_(\d{2})\.csv",
            columns=list(CLIMATE_COLUMNS.values()),
        )
        self.vegetation = load_monthly_series(
            raw_data / "MOD13A3_EVI_NDVI", r"evi_ndvi_(\d{4})_(\d{2})\.csv",
            columns=list(VEGETATION_COLUMNS.values()),
        )
        self.pm25 = load_monthly_series(
            raw_data / "PM25", r"pm25_(\d{4})_(\d{2})\.csv",
            columns=list(PM25_COLUMNS.values()),
        )
        self.pop = load_epoch_series(raw_data / "GHS_POP")
        self.built_surface = load_epoch_series(raw_data / "GHS_BUILT_S")
        self.urbanization = load_epoch_series(raw_data / "GHS_SMOD")
        self.built_height = read_indexed(
            raw_data / "GHS_BUILT_H" / "ghs_built_h.csv"
        )
        self.land_use = land_use_table(
            raw_data / "GHS_BUILT_C" / "ghs_built_c.csv"
        )

    def monthly_window(
        self, frames: list[pd.DataFrame], columns: dict[str, str],
        city_id: int, target_index: int,
    ) -> dict[str, np.ndarray]:
        window = {field: [] for field in columns}
        for frame in frames[target_index - 12 : target_index]:
            for field, column in columns.items():
                window[field].append(np.float32(frame.loc[city_id, column]))
        return {
            field: np.asarray(values, dtype=np.float32)
            for field, values in window.items()
        }

    def build_sample(
        self, city_id: int, target_index: int, uhi: np.ndarray
    ) -> dict[str, np.ndarray]:
        target_year, target_month = date_from_index(target_index)
        epoch = epoch_index(target_year)
        sample = {
            "year_info": np.array([target_year]),
            "month_info": np.array([target_month]),
            "location_info": np.array(
                [
                    self.city_info.loc[
                        city_id, ["centroid_x", "centroid_y"]
                    ].to_numpy(dtype=np.float32)
                ]
            ),
            "area_info": np.array([float(self.city_info.loc[city_id, "Area"])]),
            "pop_info": np.array(
                [np.float32(self.pop[epoch].loc[city_id].iloc[0])],
                dtype=np.float32,
            ),
            "built_surface_info": np.array(
                [np.float32(self.built_surface[epoch].loc[city_id].iloc[0])],
                dtype=np.float32,
            ),
            "built_height_info": np.array(
                [np.float32(self.built_height.loc[city_id].iloc[0])],
                dtype=np.float32,
            ),
            "urbanization_level_info": np.array(
                [np.float32(self.urbanization[epoch].loc[city_id].iloc[0])],
                dtype=np.float32,
            ),
            "urban_land_use_info": np.array(
                [self.land_use.loc[city_id].to_numpy(dtype=np.float64)]
            ),
            "compactness_info": np.array(
                [float(self.city_info.loc[city_id, "Compactness (CI)"])]
            ),
        }
        for field, values in self.monthly_window(
            self.climate, CLIMATE_COLUMNS, city_id, target_index
        ).items():
            sample[field] = np.array([values])
        for field, values in self.monthly_window(
            self.vegetation, VEGETATION_COLUMNS, city_id, target_index
        ).items():
            sample[field] = np.array([values])
        for field, values in self.monthly_window(
            self.pm25, PM25_COLUMNS, city_id, target_index
        ).items():
            sample[field] = np.array([values])
        sample["x"] = np.array(
            [uhi[target_index - 12 : target_index].astype(np.float32)]
        )
        sample["y"] = np.array([float(uhi[target_index])])
        sample["city_idx"] = np.array([city_id])
        return sample

    def complete_terminal_month(self, city_id: int, values: np.ndarray) -> None:
        terminal = {
            "cloud_cover": self.climate[-1].loc[city_id, CLIMATE_COLUMNS["cloud_info"]],
            "solar_radiation": self.climate[-1].loc[
                city_id, CLIMATE_COLUMNS["solar_info"]
            ],
            "precipitation_flux": self.climate[-1].loc[
                city_id, CLIMATE_COLUMNS["precipitation_info"]
            ],
            "snow_thickness": self.climate[-1].loc[
                city_id, CLIMATE_COLUMNS["snow_info"]
            ],
            "temperature": self.climate[-1].loc[
                city_id, CLIMATE_COLUMNS["temperature_info"]
            ],
            "vapour_pressure": self.climate[-1].loc[
                city_id, CLIMATE_COLUMNS["vapour_info"]
            ],
            "wind_speed": self.climate[-1].loc[city_id, CLIMATE_COLUMNS["wind_info"]],
            "evi": self.vegetation[-1].loc[city_id, VEGETATION_COLUMNS["evi_info"]],
            "ndvi": self.vegetation[-1].loc[
                city_id, VEGETATION_COLUMNS["ndvi_info"]
            ],
            "pm25": self.pm25[-1].loc[city_id, PM25_COLUMNS["pm25_info"]],
        }
        for feature, value in terminal.items():
            values[-1, FEATURE_INDEX[feature]] = np.float32(value)


def sample_date(sample: dict) -> tuple[int, int]:
    return int(np.asarray(sample["year_info"]).ravel()[0]), int(
        np.asarray(sample["month_info"]).ravel()[0]
    )


def verify_city(city_id: int, rows: list[dict], reference_root: Path) -> None:
    reference = {}
    for split in ("train", "valid"):
        dataset_path = reference_root / split / f"UHI_city_{city_id}"
        stored = load_from_disk(str(dataset_path))
        if {"year_month", "values"}.issubset(stored.column_names):
            dataset = load_city_dataset(reference_root, city_id, split)
        else:
            dataset = stored
        for row in dataset:
            reference[sample_date(row)] = row
    if len(reference) != len(rows):
        raise AssertionError(
            f"city {city_id}: {len(rows)} generated vs {len(reference)} reference samples"
        )
    for row in rows:
        date = sample_date(row)
        expected = reference[date]
        for field, value in row.items():
            generated = np.asarray(value, dtype=np.float64).ravel()
            stored = np.asarray(expected[field], dtype=np.float64).ravel()
            if generated.shape != stored.shape or not np.allclose(
                generated, stored, rtol=0.0, atol=1e-5
            ):
                raise AssertionError(
                    f"city {city_id} {date} field {field}: "
                    f"generated {generated[:3]} != stored {stored[:3]}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument(
        "--city-metrics",
        type=Path,
        default=Path("data/city_metrics/city_metrics_all.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--expected-cities", type=int, default=1020)
    parser.add_argument("--city-limit", type=int, default=0)
    parser.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        help="Compare generated samples with an existing dataset root instead of writing.",
    )
    args = parser.parse_args()

    sources = RawSources(args.raw_data, args.city_metrics)
    city_ids = sorted(int(v) for v in sources.city_info.index)
    if len(city_ids) != args.expected_cities:
        raise ValueError(
            f"expected {args.expected_cities} cities in {args.city_metrics}, "
            f"found {len(city_ids)}"
        )
    if args.city_limit:
        city_ids = city_ids[: args.city_limit]

    for city_id in tqdm(city_ids, desc="Building city datasets"):
        uhi = raw_uhi_values(args.raw_data, city_id)
        rows = [
            sources.build_sample(city_id, target_index, uhi)
            for target_index in range(FIRST_TARGET_INDEX, MONTHS_TOTAL)
        ]
        if args.verify_against is not None:
            verify_city(city_id, rows, args.verify_against)
            continue

        year_month, values = samples_to_series(rows, city_id)
        sources.complete_terminal_month(city_id, values)
        output_train = args.output / "train" / f"UHI_city_{city_id}"
        output_valid = args.output / "valid" / f"UHI_city_{city_id}"
        if output_train.exists() or output_valid.exists():
            raise FileExistsError(f"output already exists for city {city_id}")
        save_city_series(output_train, year_month[:180], values[:180])
        save_city_series(output_valid, year_month[180:], values[180:])

    if args.verify_against is not None:
        print(f"Verified {len(city_ids)} cities against {args.verify_against}.")
    else:
        print(f"Wrote {len(city_ids)} city datasets below {args.output}.")


if __name__ == "__main__":
    main()
