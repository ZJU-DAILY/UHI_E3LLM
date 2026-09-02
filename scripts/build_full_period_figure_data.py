#!/usr/bin/env python3
"""Build the validated 2004-2020 city-year metrics consumed by Fig. 3b."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "figures"))

from figure_data import METHOD_ORDER, city_year_metrics, collect, validate_collection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-period-output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "full_period",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "figures"
            / "data"
            / "method_city_year_metrics_2004_2020.csv"
        ),
    )
    args = parser.parse_args()

    samples = collect(args.full_period_output_root)
    validate_collection(
        samples,
        expected_cities=100,
        expected_samples_per_city=204,
        methods=METHOD_ORDER,
    )
    result = city_year_metrics(samples)
    counts = result.groupby("method").size()
    if len(result) != 8 * 100 * 17 or not counts.eq(100 * 17).all():
        raise RuntimeError(
            "Expected 13,600 method/city/year rows "
            f"(8 methods x 100 cities x 17 years); observed {len(result)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"[ok] Fig. 3b source: {args.output}")


if __name__ == "__main__":
    main()
