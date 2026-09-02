#!/usr/bin/env python3
"""Build regional assignments and leave-one-region-out manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from assign_city_regions import assign_regions


REGION_FILENAMES = {
    "Africa": "Africa",
    "East Asia & Oceania": "EastAsiaOceania",
    "Europe": "Europe",
    "Latin America": "LatinAmerica",
    "North America": "NorthAmerica",
    "West Asia": "SCWAsia",
}


def write_regional_folds(assignments: pd.DataFrame, output_dir: Path) -> None:
    """Write six held-out-region manifests and their exact complements."""
    if len(assignments) != 600 or assignments["id"].nunique() != 600:
        raise ValueError("regional assignments must contain 600 unique cities")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_ids = set(assignments["id"].astype(int))
    observed_test_ids: set[int] = set()
    for region, filename in REGION_FILENAMES.items():
        test_ids = set(
            assignments.loc[assignments["region"] == region, "id"].astype(int)
        )
        train_ids = all_ids - test_ids
        if not test_ids or train_ids & test_ids or train_ids | test_ids != all_ids:
            raise RuntimeError(f"invalid leave-one-region-out partition: {region}")
        pd.DataFrame({"id": sorted(train_ids)}).to_csv(
            output_dir / f"fold_{filename}_train.csv", index=False
        )
        pd.DataFrame({"id": sorted(test_ids)}).to_csv(
            output_dir / f"fold_{filename}_test.csv", index=False
        )
        observed_test_ids.update(test_ids)
    if observed_test_ids != all_ids:
        raise RuntimeError("regional test folds do not cover every modeling city once")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    metrics_dir = args.data_dir / "city_metrics"

    all_cities = pd.read_csv(metrics_dir / "city_metrics_all.csv")
    train = pd.read_csv(metrics_dir / "city_metrics_train.csv")
    test = pd.read_csv(metrics_dir / "city_metrics_test.csv")

    train_ids = set(train["id"].astype(int))
    test_ids = set(test["id"].astype(int))
    if train_ids.intersection(test_ids):
        raise RuntimeError("Training and test city lists overlap")
    if len(train_ids) != 500 or len(test_ids) != 100:
        raise RuntimeError("Expected 500 training and 100 test cities")
    all_ids = set(all_cities["id"].astype(int))
    if not (train_ids | test_ids).issubset(all_ids):
        raise RuntimeError("Modeling manifests contain IDs absent from city_metrics_all.csv")

    modeling = pd.concat(
        [train.assign(split="train"), test.assign(split="test")],
        ignore_index=True,
    )
    if modeling["id"].nunique() != 600:
        raise RuntimeError("Expected 600 unique modeling cities")
    for column in ("city_name", "country"):
        if column not in modeling.columns:
            modeling[column] = ""
        else:
            modeling[column] = modeling[column].fillna("").astype(str)
    regions = assign_regions(modeling)

    output_columns = [
        "id",
        "city_name",
        "country",
        "split",
        "region",
        "centroid_x",
        "centroid_y",
        "Area",
    ]
    assignments = regions[output_columns].sort_values(
        ["region", "split", "id"]
    )
    assignments.to_csv(metrics_dir / "region_assignments_600.csv", index=False)
    modeling.sort_values(["split", "id"]).to_csv(
        metrics_dir / "modeling_split_600.csv", index=False
    )
    write_regional_folds(assignments, args.data_dir / "regional_folds")
    counts = regions["region"].value_counts().sort_index()
    print(
        "Wrote modeling_split_600.csv, coordinate-derived "
        "region_assignments_600.csv, and 12 regional fold manifests"
    )
    print(counts.to_string())


if __name__ == "__main__":
    main()
