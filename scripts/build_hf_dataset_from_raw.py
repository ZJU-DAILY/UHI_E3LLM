#!/usr/bin/env python3
"""Build the processed per-city Hugging Face datasets from the raw factor CSVs.

This is the missing conversion step between the Google Earth Engine
preprocessing outputs (scripts/Preprocess_new.ipynb) and the training/
inference dataset layout under ``data/train`` and ``data/valid``.

Expected raw layout (one directory, ``--raw-data``):

    Extracted_City_Data/UHI_city_<id>.csv   216 monthly rows, column Amod2
    ClimateData/climate_<YYYY>_<MM>.csv     216 files, ERA5 monthly means
    MOD13A3_EVI_NDVI/evi_ndvi_<YYYY>_<MM>.csv
    PM25/pm25_<YYYY>_<MM>.csv
    GHS_POP/GHS_POP_<2005|2010|2015|2020>.csv
    GHS_BUILT_S/GHS_BUILT_S_<2005|2010|2015|2020>.csv
    GHS_SMOD/ghs_smod_<2005|2010|2015|2020>.csv
    GHS_BUILT_H/ghs_built_h_2018.csv
    GHS_BUILT_C/ghs_built_c_2018.csv

Static geometry attributes (centroid, area, compactness) come from the
versioned ``data/city_metrics/city_metrics_all.csv``.

Sample construction reproduces the manuscript alignment: for every target
month index ``t`` in ``12..215`` (2004-01 through 2020-12), the historical
window is ``[t-12, t)``, the label is the UHI value at ``t``, and the ten
monthly factors use the same 12-month window. The 5-year GHS epochs are
assigned with the frozen rule ``min((target_year - 2003 + 3) // 5, 3)`` over
the epochs ``[2005, 2010, 2015, 2020]``. Targets through 2017-12 form the
168-sample training split; 2018-01 through 2020-12 form the 36-sample
evaluation split.

With ``--verify-against`` the script does not write anything; it instead
checks every generated sample against an existing dataset directory
field-by-field, which is how the released implementation was validated
against the archived manuscript datasets.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk
from tqdm import tqdm


GHS_EPOCHS = (2005, 2010, 2015, 2020)
FIRST_YEAR = 2003
MONTHS_TOTAL = 216
FIRST_TARGET_INDEX = 12
TRAIN_LAST_INDEX = 179  # 2017-12

LAND_USE_COLUMNS = [
    "Open Spaces",
    "Water Surfaces",
    "Road Surfaces",
    "Residential",
    "Non-Residential",
]
# GHS-BUILT-C class aggregation (class 0 "No data" is excluded).
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


def month_index(year: int, month: int) -> int:
    return (year - FIRST_YEAR) * 12 + month - 1


def date_from_index(index: int) -> tuple[int, int]:
    return FIRST_YEAR + index // 12, index % 12 + 1


def epoch_index(target_year: int) -> int:
    return min((target_year - FIRST_YEAR + 3) // 5, len(GHS_EPOCHS) - 1)


def read_indexed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "id" not in frame.columns:
        raise ValueError(f"{path}: missing id column")
    frame = frame.drop(
        columns=[c for c in frame.columns if c.startswith("Unnamed")]
    )
    return frame.set_index("id", verify_integrity=True)


def load_monthly_series(directory: Path, pattern: str) -> list[pd.DataFrame]:
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
    return [read_indexed(dated[date]) for date in expected]


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
        self.city_info = read_indexed(city_metrics)
        self.climate = load_monthly_series(
            raw_data / "ClimateData", r"climate_(\d{4})_(\d{2})\.csv"
        )
        self.vegetation = load_monthly_series(
            raw_data / "MOD13A3_EVI_NDVI", r"evi_ndvi_(\d{4})_(\d{2})\.csv"
        )
        self.pm25 = load_monthly_series(
            raw_data / "PM25", r"pm25_(\d{4})_(\d{2})\.csv"
        )
        self.pop = load_epoch_series(raw_data / "GHS_POP")
        self.built_surface = load_epoch_series(raw_data / "GHS_BUILT_S")
        self.urbanization = load_epoch_series(raw_data / "GHS_SMOD")
        self.built_height = read_indexed(
            raw_data / "GHS_BUILT_H" / "ghs_built_h_2018.csv"
        )
        self.land_use = land_use_table(
            raw_data / "GHS_BUILT_C" / "ghs_built_c_2018.csv"
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


def sample_date(sample: dict) -> tuple[int, int]:
    return int(np.asarray(sample["year_info"]).ravel()[0]), int(
        np.asarray(sample["month_info"]).ravel()[0]
    )


def verify_city(city_id: int, rows: list[dict], reference_root: Path) -> None:
    reference = {}
    for split in ("train", "valid"):
        dataset = load_from_disk(str(reference_root / split / f"UHI_city_{city_id}"))
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
    parser.add_argument("--seed", type=int, default=42)
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

    rng = random.Random(args.seed)
    for city_id in tqdm(city_ids, desc="Building city datasets"):
        uhi = raw_uhi_values(args.raw_data, city_id)
        rows = [
            sources.build_sample(city_id, target_index, uhi)
            for target_index in range(FIRST_TARGET_INDEX, MONTHS_TOTAL)
        ]
        if args.verify_against is not None:
            verify_city(city_id, rows, args.verify_against)
            continue

        train_rows = [
            row
            for row in rows
            if month_index(*sample_date(row)) <= TRAIN_LAST_INDEX
        ]
        valid_rows = [
            row
            for row in rows
            if month_index(*sample_date(row)) > TRAIN_LAST_INDEX
        ]
        if len(train_rows) != 168 or len(valid_rows) != 36:
            raise ValueError(f"city {city_id}: expected 168+36 samples")
        rng.shuffle(train_rows)
        rng.shuffle(valid_rows)
        output_train = args.output / "train" / f"UHI_city_{city_id}"
        output_valid = args.output / "valid" / f"UHI_city_{city_id}"
        if output_train.exists() or output_valid.exists():
            raise FileExistsError(f"output already exists for city {city_id}")
        Dataset.from_list(train_rows).save_to_disk(str(output_train))
        Dataset.from_list(valid_rows).save_to_disk(str(output_valid))

    if args.verify_against is not None:
        print(f"Verified {len(city_ids)} cities against {args.verify_against}.")
    else:
        print(f"Wrote {len(city_ids)} city datasets below {args.output}.")


if __name__ == "__main__":
    main()
