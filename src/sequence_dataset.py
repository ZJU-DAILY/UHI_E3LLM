from __future__ import annotations

from pathlib import Path

import numpy as np
from datasets import Array2D, Dataset, Features, Sequence, Value, load_from_disk


FEATURE_NAMES = (
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
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
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
FIRST_YEAR = 2003
FIRST_TARGET_INDEX = 12
TRAIN_MONTHS = 180
TOTAL_MONTHS = 216


def _flat(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _scalar(value) -> float:
    return float(_flat(value)[0])


def target_index(sample) -> int:
    year = int(_scalar(sample["year_info"]))
    month = int(_scalar(sample["month_info"]))
    return (year - FIRST_YEAR) * 12 + month - 1


def samples_to_series(samples, city_id: int) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(samples, key=target_index)
    expected_targets = list(range(FIRST_TARGET_INDEX, TOTAL_MONTHS))
    indices = [target_index(sample) for sample in ordered]
    if indices != expected_targets:
        raise ValueError(f"city {city_id}: target months are incomplete or duplicated")

    values = np.full((TOTAL_MONTHS, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    for sample in ordered:
        target = target_index(sample)
        history = slice(target - 12, target)

        def assign_history(column: str, source, field: str) -> None:
            incoming = _flat(source)
            if incoming.size != 12:
                raise ValueError(f"city {city_id} {field}: expected 12 values")
            channel = FEATURE_INDEX[column]
            current = values[history, channel]
            overlap = np.isfinite(current)
            if overlap.any() and not np.allclose(
                current[overlap], incoming[overlap], rtol=0.0, atol=1e-5
            ):
                raise ValueError(
                    f"city {city_id} {field}: inconsistent overlapping months"
                )
            current[~overlap] = incoming[~overlap]
            values[history, channel] = current

        assign_history("Amod2", sample["x"], "x")
        for source, column in DYNAMIC_FIELDS.items():
            assign_history(column, sample[source], source)

        values[target, FEATURE_INDEX["Amod2"]] = _scalar(sample["y"])
        location = _flat(sample["location_info"])
        land_use = _flat(sample["urban_land_use_info"])
        if location.size != 2 or land_use.size != 5:
            raise ValueError(f"city {city_id}: invalid structural feature dimensions")
        structural = (
            _scalar(sample["area_info"]),
            _scalar(sample["built_height_info"]),
            _scalar(sample["built_surface_info"]) * 100.0,
            _scalar(sample["urbanization_level_info"]),
            location[0],
            location[1],
            _scalar(sample["compactness_info"]),
            _scalar(sample["pop_info"]) * 100.0,
            *land_use.tolist(),
        )
        values[target, :13] = structural

    values[:FIRST_TARGET_INDEX, :13] = values[FIRST_TARGET_INDEX, :13]
    missing_uhi = np.flatnonzero(~np.isfinite(values[:, FEATURE_INDEX["Amod2"]]))
    if missing_uhi.size:
        raise ValueError(f"city {city_id}: missing UHI months {missing_uhi.tolist()}")
    for column in DYNAMIC_FIELDS.values():
        channel = FEATURE_INDEX[column]
        missing = np.flatnonzero(~np.isfinite(values[:, channel]))
        if not np.array_equal(missing, np.array([TOTAL_MONTHS - 1])):
            raise ValueError(
                f"city {city_id}: unexpected missing months for {column}: {missing.tolist()}"
            )
    if not np.isfinite(values[:, :13]).all():
        raise ValueError(f"city {city_id}: missing structural values")

    year_month = np.asarray(
        [
            (FIRST_YEAR + index // 12) * 100 + index % 12 + 1
            for index in range(TOTAL_MONTHS)
        ],
        dtype=np.int32,
    )
    return year_month, values


def save_city_series(path: Path, year_month, values) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    years = np.asarray(year_month, dtype=np.int32)
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (len(years), len(FEATURE_NAMES)):
        raise ValueError(f"invalid sequence matrix shape: {matrix.shape}")
    if years.size == 0 or np.any(np.diff(years) <= 0):
        raise ValueError("year_month must be strictly increasing")
    features = Features(
        {
            "year_month": Sequence(Value("int32"), length=len(years)),
            "values": Array2D(
                shape=(len(years), len(FEATURE_NAMES)), dtype="float64"
            ),
        }
    )
    dataset = Dataset.from_dict(
        {"year_month": [years.tolist()], "values": [matrix]},
        features=features,
    )
    dataset.save_to_disk(str(path))


def read_city_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    dataset = load_from_disk(str(path))
    if len(dataset) != 1 or set(dataset.column_names) != {"year_month", "values"}:
        raise ValueError(f"{path}: expected one time-series tensor record")
    row = dataset[0]
    year_month = np.asarray(row["year_month"], dtype=np.int32)
    values = np.asarray(row["values"], dtype=np.float64)
    if values.shape != (len(year_month), len(FEATURE_NAMES)):
        raise ValueError(f"{path}: invalid sequence tensor shape {values.shape}")
    return year_month, values


def read_city_samples(
    data_root: Path, city_id: int, split: str
) -> Dataset:
    data_root = Path(data_root)
    train_dates, train_values = read_city_series(
        data_root / "train" / f"UHI_city_{city_id}"
    )
    if split == "train":
        dates, values = train_dates, train_values
        targets = range(FIRST_TARGET_INDEX, TRAIN_MONTHS)
    elif split in {"valid", "all"}:
        valid_dates, valid_values = read_city_series(
            data_root / "valid" / f"UHI_city_{city_id}"
        )
        dates = np.concatenate((train_dates, valid_dates))
        values = np.concatenate((train_values, valid_values), axis=0)
        targets = (
            range(TRAIN_MONTHS, TOTAL_MONTHS)
            if split == "valid"
            else range(FIRST_TARGET_INDEX, TOTAL_MONTHS)
        )
    else:
        raise ValueError(f"unsupported split: {split}")

    rows = []
    dynamic_indices = {
        field: FEATURE_INDEX[column] for field, column in DYNAMIC_FIELDS.items()
    }
    for target in targets:
        if target >= len(values) or target < 12:
            raise ValueError(f"city {city_id}: target index {target} is out of range")
        year_month = int(dates[target])
        year, month = divmod(year_month, 100)
        history = values[target - 12 : target]
        static = values[target, :13]
        rows.append(
            {
                "year_info": [year],
                "month_info": [month],
                "location_info": [[static[4], static[5]]],
                "area_info": [static[0]],
                "pop_info": [static[7] / 100.0],
                "built_surface_info": [static[2] / 100.0],
                "built_height_info": [static[1]],
                "urbanization_level_info": [static[3]],
                "urban_land_use_info": [static[8:13].tolist()],
                "compactness_info": [static[6]],
                **{
                    field: [history[:, index].tolist()]
                    for field, index in dynamic_indices.items()
                },
                "x": [
                    history[:, FEATURE_INDEX["Amod2"]]
                    .astype(np.float32)
                    .tolist()
                ],
                "y": [values[target, FEATURE_INDEX["Amod2"]]],
                "city_idx": [city_id],
            }
        )
    return Dataset.from_list(rows)


def load_city_dataset(data_root: Path, city_id: int, split: str) -> Dataset:
    return read_city_samples(data_root, city_id, split)


def load_city_matrix(data_root: Path, city_id: int) -> tuple[np.ndarray, np.ndarray]:
    dates, train_values = read_city_series(
        Path(data_root) / "train" / f"UHI_city_{city_id}"
    )
    valid_dates, valid_values = read_city_series(
        Path(data_root) / "valid" / f"UHI_city_{city_id}"
    )
    dates = np.concatenate((dates, valid_dates))
    values = np.concatenate((train_values, valid_values), axis=0)
    expected_dates = np.asarray(
        [
            (FIRST_YEAR + index // 12) * 100 + index % 12 + 1
            for index in range(TOTAL_MONTHS)
        ],
        dtype=np.int32,
    )
    if not np.array_equal(dates, expected_dates):
        raise ValueError(f"city {city_id}: expected monthly sequence 2003-01 to 2020-12")
    if not np.isfinite(values[:, :13]).all():
        raise ValueError(f"city {city_id}: missing structural features")
    if not np.isfinite(values[:, FEATURE_INDEX["Amod2"]]).all():
        raise ValueError(f"city {city_id}: missing UHI values")
    if not np.isfinite(values[:, 13:23]).all():
        raise ValueError(f"city {city_id}: invalid environmental feature coverage")
    return dates, values
