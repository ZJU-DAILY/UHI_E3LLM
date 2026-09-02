#!/usr/bin/env python3
"""Assign cities to the six manuscript macro-regions from centroid coordinates.

The continuous longitude/latitude cuts reproduce the deterministic partition
used for Fig. 2(g) and the leave-one-region-out experiments. Political borders
are not used: every city is assigned from its city-center coordinate alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WESTERN_HEMISPHERE_CUT = -30.0
AMERICAS_LATITUDE_CUT = 10.0
AFRICA_EAST_CUT = 55.0
AFRICA_NORTH_CUT = 35.0
EUROPE_EAST_CUT = 40.0
EAST_ASIA_WEST_CUT = 93.0

REGION_ORDER = (
    "North America",
    "Latin America",
    "Africa",
    "Europe",
    "East Asia & Oceania",
    "West Asia",
)

EXPECTED_1020_COUNTS = {
    "North America": 296,
    "Latin America": 56,
    "Africa": 83,
    "Europe": 179,
    "East Asia & Oceania": 299,
    "West Asia": 107,
}


def classify_region(longitude: float, latitude: float) -> str:
    """Assign one valid WGS84 coordinate to exactly one macro-region."""
    longitude = float(longitude)
    latitude = float(latitude)
    if not np.isfinite(longitude) or not np.isfinite(latitude):
        raise ValueError("longitude and latitude must be finite")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError(f"longitude outside [-180, 180]: {longitude}")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"latitude outside [-90, 90]: {latitude}")

    if longitude < WESTERN_HEMISPHERE_CUT:
        return "North America" if latitude >= AMERICAS_LATITUDE_CUT else "Latin America"
    if longitude < AFRICA_EAST_CUT and latitude < AFRICA_NORTH_CUT:
        return "Africa"
    if longitude < EUROPE_EAST_CUT and latitude >= AFRICA_NORTH_CUT:
        return "Europe"
    if longitude >= EAST_ASIA_WEST_CUT:
        return "East Asia & Oceania"
    return "West Asia"


def assign_regions(
    frame: pd.DataFrame,
    longitude_column: str = "centroid_x",
    latitude_column: str = "centroid_y",
    region_column: str = "region",
) -> pd.DataFrame:
    """Return a copy of ``frame`` with deterministic macro-region labels."""
    missing = [
        column
        for column in (longitude_column, latitude_column)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"missing coordinate columns: {', '.join(missing)}")

    result = frame.copy()
    result[region_column] = [
        classify_region(longitude, latitude)
        for longitude, latitude in zip(
            result[longitude_column], result[latitude_column]
        )
    ]
    return result


def validate_expected_counts(
    frame: pd.DataFrame,
    expected: dict[str, int],
    region_column: str = "region",
) -> None:
    """Raise when observed region counts differ from a frozen partition."""
    observed = frame[region_column].value_counts().to_dict()
    if observed != expected:
        details = ", ".join(
            f"{region}={observed.get(region, 0)} (expected {count})"
            for region, count in expected.items()
        )
        raise ValueError(f"region count validation failed: {details}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--longitude-column", default="centroid_x")
    parser.add_argument("--latitude-column", default="centroid_y")
    parser.add_argument("--region-column", default="region")
    parser.add_argument(
        "--validate-1020",
        action="store_true",
        help="Require the frozen region counts for all 1,020 candidate cities.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    assigned = assign_regions(
        frame,
        longitude_column=args.longitude_column,
        latitude_column=args.latitude_column,
        region_column=args.region_column,
    )
    if args.validate_1020:
        if len(assigned) != 1020:
            raise ValueError(f"expected 1,020 cities, found {len(assigned):,}")
        validate_expected_counts(assigned, EXPECTED_1020_COUNTS, args.region_column)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(assigned):,} cities)")
    print(assigned[args.region_column].value_counts().reindex(REGION_ORDER).to_string())


if __name__ == "__main__":
    main()
