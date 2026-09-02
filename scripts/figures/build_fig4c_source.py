#!/usr/bin/env python3
"""Generate the five-city Fig. 4c source table from raw factor CSVs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DATASET_SPECS = {
    "Climate Data": {
        "directory": "ClimateData",
        "pattern": r"climate_(\d{4})_(\d{2})\.csv",
        "columns": {
            "Wind_Speed_10m_Mean": "Average Wind Speed",
            "Temperature_Air_2m_Mean_24h": "Average Temperature",
            "Cloud_Cover_Mean": "Average Cloud Cover",
            "Precipitation_Flux": "Precipitation Flux",
            "Snow_Thickness_Mean": "Snow Thickness",
            "Vapour_Pressure_Mean": "Average Vapour Pressure",
            "Solar_Radiation_Flux": "Solar Radiation",
        },
    },
    "Built Surface Area": {
        "directory": "GHS_BUILT_S",
        "pattern": r"GHS_BUILT_S_(\d{4})\.csv",
        "columns": {"built_surface": "Built Surface Area"},
    },
    "Population Density": {
        "directory": "GHS_POP",
        "pattern": r"GHS_POP_(\d{4})\.csv",
        "columns": {"population_count": "Population Density"},
    },
    "Urbanization Level": {
        "directory": "GHS_SMOD",
        "pattern": r"ghs_smod_(\d{4})\.csv",
        "columns": {"smod_code": "Urbanization Level"},
    },
    "Vegetation Cover": {
        "directory": "MOD13A3_EVI_NDVI",
        "pattern": r"evi_ndvi_(\d{4})_(\d{2})\.csv",
        "columns": {"NDVI": "NDVI", "EVI": "EVI"},
    },
    "PM2.5 Concentration": {
        "directory": "PM25",
        "pattern": r"pm25_(\d{4})_(\d{2})\.csv",
        "columns": {"b1": "PM2.5 Concentration"},
    },
}


def collect_records(
    raw_data: Path, city_ids: set[int] | None = None
) -> pd.DataFrame:
    records = []
    for group, spec in DATASET_SPECS.items():
        directory = raw_data / spec["directory"]
        pattern = re.compile(spec["pattern"])
        if not directory.is_dir():
            raise FileNotFoundError(f"missing raw factor directory: {directory}")
        matched = 0
        for path in sorted(directory.glob("*.csv")):
            match = pattern.fullmatch(path.name)
            if not match:
                continue
            matched += 1
            year = int(match.group(1))
            month = int(match.group(2)) if match.lastindex == 2 else 1
            usecols = list(spec["columns"])
            if city_ids is not None:
                usecols.append("id")
            frame = pd.read_csv(path, usecols=usecols)
            if city_ids is not None:
                frame = frame[frame["id"].isin(city_ids)]
                if frame.empty:
                    raise ValueError(
                        f"none of the requested city IDs occur in {path}"
                    )
            for source_column, prompt_label in spec["columns"].items():
                values = pd.to_numeric(frame[source_column], errors="coerce")
                records.append(
                    {
                        "date": f"{year:04d}-{month:02d}-01",
                        "group": group,
                        "factor": prompt_label,
                        "raw_mean": values.mean(),
                    }
                )
        if matched == 0:
            raise FileNotFoundError(
                f"no files matching {spec['pattern']} below {directory}"
            )
    table = pd.DataFrame.from_records(records)
    table["normalized_value"] = table.groupby("factor")["raw_mean"].transform(
        lambda values: (
            (values - values.min()) / (values.max() - values.min())
            if values.max() != values.min()
            else 0.0
        )
    )
    return table.sort_values(["group", "factor", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/figures/data/fig4c_factor_evolution_5cities.csv"
        ),
    )
    parser.add_argument(
        "--city-ids",
        help="Optional comma-separated city IDs; the default uses every row.",
    )
    args = parser.parse_args()
    city_ids = None
    if args.city_ids:
        city_ids = {int(value.strip()) for value in args.city_ids.split(",")}
    table = collect_records(args.raw_data, city_ids=city_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {len(table)} factor-date rows to {args.output}")


if __name__ == "__main__":
    main()
