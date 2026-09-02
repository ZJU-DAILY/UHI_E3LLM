#!/usr/bin/env python3
"""Aggregate the five July-2018 LIME outputs into the Fig. 5c source table."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


CASE_CITIES = [
    (1, 864, "Norton Shores, United States"),
    (2, 677, "Solna, Sweden"),
    (3, 222, "Wenzhou, China"),
    (4, 661, "Madrid, Spain"),
    (5, 413, "Delhi, India"),
]
DROP_FEATURES = {"year_info", "month_info"}
LOCATION_FEATURES = {"longitude", "latitude"}
MONTH_SUFFIX = re.compile(r"_(?:[1-9]|1[0-2])$")


def read_grouped_weights(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    grouped: dict[str, float] = defaultdict(float)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line == "Top Features:" or ":" not in line:
            continue
        feature, raw_weight = line.rsplit(":", 1)
        feature = feature.strip()
        if feature in DROP_FEATURES or feature.startswith("past_uhi_"):
            continue
        feature = MONTH_SUFFIX.sub("", feature)
        if feature in LOCATION_FEATURES:
            feature = "location"
        grouped[feature] += float(raw_weight.strip())
    if len(grouped) < 10:
        raise ValueError(f"{path}: fewer than ten usable grouped LIME features")
    scale = sum(abs(weight) for weight in grouped.values())
    if scale == 0:
        raise ValueError(f"{path}: all grouped LIME weights are zero")
    return {feature: weight / scale for feature, weight in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lime-root", type=Path, default=Path("outputs/lime_explanation"))
    parser.add_argument(
        "--sample-index",
        type=int,
        default=6,
        help="Zero-based target index; 6 is July 2018 for the 2018-2020 test split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/data/fig5c_lime_importance.csv"),
    )
    args = parser.parse_args()
    if args.sample_index < 0:
        raise ValueError("sample index must be non-negative")

    records = []
    for panel_order, city_id, display_city in CASE_CITIES:
        source = args.lime_root / f"City{city_id}" / f"explanation_{args.sample_index}.txt"
        weights = read_grouped_weights(source)
        selected = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)[:10]
        for rank, (feature, weight) in enumerate(selected, start=1):
            records.append(
                {
                    "panel_order": panel_order,
                    "display_city": display_city,
                    "source_city_id": city_id,
                    "rank": rank,
                    "feature": feature,
                    "lime_importance": weight,
                }
            )

    output = pd.DataFrame.from_records(records)
    counts = output.groupby("panel_order").size()
    if len(counts) != 5 or not counts.eq(10).all():
        raise RuntimeError("Fig. 5c output must contain five panels with ten rows each")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
