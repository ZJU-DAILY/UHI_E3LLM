"""Shared feature extraction and data loading for the tabular baselines.

`flatten_sample` defines the 147-dimensional feature vector used by every
tabular baseline (12 historical UHI values, 6 structural scalars, 5 land-use
shares, longitude/latitude, 10 monthly factors x 12 months, year, month).
`load_data` reproduces the manuscript split logic: the 500 training cities
come from `data/train/` and the 100 held-out test cities from `data/valid/`
restricted to the 2018-2020 targets (3,600 samples).
"""

import os

import numpy as np
import pandas as pd
from datasets import load_from_disk


def flatten_sample(sample):
    """Flatten a dataset sample into a 1D feature vector + target.

    All values are stored as list[0] — scalar or list-of-12.
    """
    features = []

    # Historical UHI (12 months)
    features.extend([float(v) for v in sample["x"][0]])

    # Structure factors (scalars)
    for k in [
        "area_info",
        "built_height_info",
        "built_surface_info",
        "urbanization_level_info",
        "compactness_info",
        "pop_info",
    ]:
        features.append(float(sample[k][0]))

    # Urban land use (5 values)
    features.extend([float(v) for v in sample["urban_land_use_info"][0]])

    # Location (lon, lat)
    features.extend([float(v) for v in sample["location_info"][0]])

    # Liquidity factors (12 monthly values each)
    for k in [
        "cloud_info",
        "solar_info",
        "precipitation_info",
        "snow_info",
        "evi_info",
        "ndvi_info",
        "pm25_info",
        "temperature_info",
        "vapour_info",
        "wind_info",
    ]:
        features.extend([float(v) for v in sample[k][0]])

    # Time features
    features.append(float(sample["year_info"][0]))
    features.append(float(sample["month_info"][0]))

    target = float(sample["y"][0])
    return np.array(features, dtype=np.float32), target


def load_data(dataset_path, mode="train"):
    """Load flattened data from processed HuggingFace datasets.

    Matches the manuscript split logic in Inference.py:
      - train: loads train/ only (500 cities from city_metrics_train.csv)
      - test: loads valid/ only (2018-2020 targets for 100 held-out cities)
      - test_full: loads train/ and valid/ (2004-2020 for held-out cities)
    """
    csv_file = os.path.join(
        dataset_path,
        "city_metrics",
        f"city_metrics_{'test' if mode in {'test', 'test_full'} else 'train'}.csv",
    )
    df = pd.read_csv(csv_file)
    city_ids = df["id"].tolist()

    X_list, y_list, city_list = [], [], []

    for cid in city_ids:
        split_dirs = ["train", "valid"] if mode == "test_full" else [
            "valid" if mode == "test" else "train"
        ]
        for split_dir in split_dirs:
            ds = load_from_disk(
                os.path.join(dataset_path, split_dir, f"UHI_city_{cid}")
            )
            for sample in ds:
                try:
                    feats, target = flatten_sample(sample)
                    X_list.append(feats)
                    y_list.append(target)
                    city_list.append(cid)
                except Exception:
                    continue

    if mode == "test" and len(city_ids) == 100 and len(y_list) != 3600:
        raise RuntimeError(
            f"Expected 3,600 test targets from 100 cities; found {len(y_list)}"
        )
    if mode == "test_full" and len(city_ids) == 100 and len(y_list) != 20400:
        raise RuntimeError(
            f"Expected 20,400 full-period test targets from 100 cities; found {len(y_list)}"
        )

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(city_list),
    )
