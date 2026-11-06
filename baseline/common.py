import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sequence_dataset import load_city_dataset


def flatten_sample(sample):
    features = []

    features.extend([float(v) for v in sample["x"][0]])

    for k in [
        "area_info",
        "built_height_info",
        "built_surface_info",
        "urbanization_level_info",
        "compactness_info",
        "pop_info",
    ]:
        value = float(sample[k][0])
        features.append(value * 1000.0 if k == "compactness_info" else value)

    features.extend([float(v) for v in sample["urban_land_use_info"][0]])

    features.extend([float(v) for v in sample["location_info"][0]])

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

    features.append(float(sample["year_info"][0]))
    features.append(float(sample["month_info"][0]))

    target = float(sample["y"][0])
    return np.array(features, dtype=np.float32), target


def load_data(dataset_path, mode="train"):
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
            ds = load_city_dataset(dataset_path, int(cid), split_dir)
            for sample in ds:
                feats, target = flatten_sample(sample)
                X_list.append(feats)
                y_list.append(target)
                city_list.append(cid)

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
