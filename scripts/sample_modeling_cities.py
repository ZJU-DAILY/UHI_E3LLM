#!/usr/bin/env python3
"""Stratified selection of modeling, test, and calibration cities.

The 600 modeling cities are drawn from the 1,020 eligible candidates by
stratified random sampling across the five city-size categories, and the 500
training / 100 held-out test cities are then allocated within every stratum.
An additional 100 calibration cities are sampled, with the same stratification,
from candidates outside the 600-city modeling cohort. Per-stratum quotas are
proportional to the available stratum sizes and rounded with the
largest-remainder method (ties broken toward the larger stratum, then by
stratum order).

By default a fresh random seed is generated for every new split and printed to
the console. Pass ``--seed`` explicitly when a reproducible new split is
required.

Outputs (schema identical to ``city_metrics_all.csv`` plus ``split`` for the
600-city table):

    city_metrics_train.csv   500 training cities
    city_metrics_test.csv    100 held-out test cities
    modeling_split_600.csv   all 600 modeling cities with their split
    calibration_cities.csv   100 independent calibration city IDs

Run ``build_city_manifests.py`` afterwards to derive every regional label and
the twelve leave-one-region-out train/test manifests.
"""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path

import numpy as np
import pandas as pd


AREA_BINS = [100, 150, 450, 800, 1500, np.inf]
AREA_LABELS = ["Very small", "Small", "Medium", "Large", "Very large"]


def largest_remainder_quotas(sizes: pd.Series, total: int) -> pd.Series:
    exact = sizes * total / sizes.sum()
    quotas = np.floor(exact).astype(int)
    remainder = exact - quotas
    order = sorted(
        range(len(sizes)),
        key=lambda i: (-remainder.iloc[i], -sizes.iloc[i], i),
    )
    for i in order[: total - int(quotas.sum())]:
        quotas.iloc[i] += 1
    if int(quotas.sum()) != total or (quotas > sizes).any():
        raise ValueError("infeasible stratum quotas")
    return quotas


def assign_strata(frame: pd.DataFrame) -> pd.Series:
    strata = pd.cut(
        frame["Area"],
        bins=AREA_BINS,
        labels=AREA_LABELS,
        right=False,
        include_lowest=True,
    )
    if strata.isna().any():
        bad = frame.loc[strata.isna(), ["id", "Area"]]
        raise ValueError(f"cities outside the area strata: {bad.to_dict('records')}")
    return strata


def stratified_sample(
    frame: pd.DataFrame, quotas: pd.Series, rng: np.random.Generator
) -> pd.DataFrame:
    parts = []
    for label in AREA_LABELS:
        pool = frame[frame["stratum"] == label].sort_values("id")
        take = int(quotas[label])
        chosen = rng.choice(len(pool), size=take, replace=False)
        parts.append(pool.iloc[np.sort(chosen)])
    return pd.concat(parts)


def calibration_manifest(calibration: pd.DataFrame) -> pd.DataFrame:
    """Return the machine-readable calibration manifest used by inference."""
    ordered = calibration.sort_values("id").reset_index(drop=True)
    return pd.DataFrame(
        {
            "id": np.arange(1, len(ordered) + 1),
            "city_id": ordered["id"].astype(int),
            "longitude": ordered["centroid_x"],
            "latitude": ordered["centroid_y"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--city-metrics",
        type=Path,
        default=Path("data/city_metrics/city_metrics_all.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/city_metrics")
    )
    parser.add_argument("--modeling-cities", type=int, default=600)
    parser.add_argument("--train-cities", type=int, default=500)
    parser.add_argument("--calibration-cities", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed. If omitted, generate and print a fresh seed."
        ),
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the stratified allocation and checks without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Required to replace existing manifest files.",
    )
    args = parser.parse_args()

    candidates = pd.read_csv(args.city_metrics)
    if candidates["id"].duplicated().any():
        raise ValueError("candidate city IDs must be unique")
    candidates = candidates.copy()
    candidates["stratum"] = assign_strata(candidates)

    sizes = candidates["stratum"].value_counts().reindex(AREA_LABELS)
    modeling_quotas = largest_remainder_quotas(sizes, args.modeling_cities)
    train_quotas = largest_remainder_quotas(modeling_quotas, args.train_cities)

    seed = args.seed if args.seed is not None else secrets.randbits(63)
    rng = np.random.default_rng(seed)
    modeling = stratified_sample(candidates, modeling_quotas, rng)
    train = stratified_sample(modeling, train_quotas, rng)
    test = modeling[~modeling["id"].isin(train["id"])]

    remaining = candidates[~candidates["id"].isin(modeling["id"])]
    remaining_sizes = remaining["stratum"].value_counts().reindex(AREA_LABELS)
    calibration_quotas = largest_remainder_quotas(
        remaining_sizes, args.calibration_cities
    )
    calibration = stratified_sample(remaining, calibration_quotas, rng)

    summary = pd.DataFrame(
        {
            "eligible_1020": sizes,
            "modeling_600": modeling_quotas,
            "train_500": train_quotas,
            "test_100": modeling_quotas - train_quotas,
            "calibration_100": calibration_quotas,
        }
    )
    print(f"seed: {seed}")
    print(summary.to_string())
    print(
        f"totals: modeling={len(modeling)}, train={len(train)}, "
        f"test={len(test)}, calibration={len(calibration)}"
    )
    if (
        len(train) != args.train_cities
        or len(modeling) != args.modeling_cities
        or len(calibration) != args.calibration_cities
    ):
        raise RuntimeError("allocation failed")
    train_ids = set(train["id"].astype(int))
    test_ids = set(test["id"].astype(int))
    calibration_ids = set(calibration["id"].astype(int))
    if train_ids & test_ids or (train_ids | test_ids) & calibration_ids:
        raise RuntimeError("training, test, and calibration cohorts must be disjoint")
    if args.preview:
        print("Preview only; no files written.")
        return

    outputs = {
        "city_metrics_train.csv": train.drop(columns=["stratum"]),
        "city_metrics_test.csv": test.drop(columns=["stratum"]),
        "calibration_cities.csv": calibration_manifest(calibration),
    }
    modeling_out = modeling.drop(columns=["stratum"]).copy()
    modeling_out["split"] = np.where(
        modeling_out["id"].isin(train["id"]), "train", "test"
    )
    outputs["modeling_split_600.csv"] = modeling_out
    for name, frame in outputs.items():
        path = args.output_dir / name
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
        print(f"wrote {args.output_dir / name} ({len(frame)} cities)")
if __name__ == "__main__":
    main()
