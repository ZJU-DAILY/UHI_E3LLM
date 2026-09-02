#!/usr/bin/env python3
"""Validate one frozen leave-one-region-out city fold and print its sizes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REGION_NAMES = {
    "Africa": "Africa",
    "EastAsiaOceania": "East Asia & Oceania",
    "Europe": "Europe",
    "LatinAmerica": "Latin America",
    "NorthAmerica": "North America",
    "SCWAsia": "West Asia",
}


def load_ids(path: Path, role: str) -> set[int]:
    frame = pd.read_csv(path)
    if "id" not in frame:
        raise ValueError(f"{role} manifest lacks an id column: {path}")
    ids = pd.to_numeric(frame["id"], errors="raise")
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError(f"{role} city IDs must be non-null and unique: {path}")
    return set(ids.astype(int))


def validate_fold(
    region: str,
    train_path: Path,
    test_path: Path,
    assignments_path: Path,
) -> tuple[int, int]:
    if region not in REGION_NAMES:
        raise ValueError(f"unknown region key: {region}")
    train_ids = load_ids(train_path, "train")
    test_ids = load_ids(test_path, "test")
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"train/test overlap for {region}: {sorted(overlap)[:10]}")

    assignments = pd.read_csv(assignments_path)
    required = {"id", "region"}
    if missing := required.difference(assignments.columns):
        raise ValueError(f"assignment manifest lacks columns: {sorted(missing)}")
    if len(assignments) != 600 or assignments["id"].nunique() != 600:
        raise ValueError("regional assignment manifest must contain 600 unique cities")
    assignment_ids = set(assignments["id"].astype(int))
    if train_ids | test_ids != assignment_ids:
        raise ValueError(f"{region}: train/test union differs from the 600-city manifest")

    expected_test = set(
        assignments.loc[assignments["region"] == REGION_NAMES[region], "id"].astype(int)
    )
    if test_ids != expected_test:
        raise ValueError(f"{region}: test IDs differ from the coordinate-derived region")
    if train_ids != assignment_ids - expected_test:
        raise ValueError(f"{region}: train IDs are not the exact regional complement")
    return len(train_ids), len(test_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=sorted(REGION_NAMES))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    args = parser.parse_args()
    train_count, test_count = validate_fold(
        args.region, args.train, args.test, args.assignments
    )
    print(train_count, test_count)


if __name__ == "__main__":
    main()
