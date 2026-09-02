#!/usr/bin/env python3
"""Compute pairwise cosine similarity between LoRA update matrices."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lora_fusion import pairwise_delta_cosine  # noqa: E402


OUTPUT_FILES = (
    "pairwise_lora_similarity.csv",
    "lora_threshold_summary.csv",
    "lora_similarity_report.json",
)


def validate_thresholds(thresholds: list[float]) -> list[float]:
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    if any(
        not math.isfinite(value) or not -1.0 <= value <= 1.0
        for value in thresholds
    ):
        raise ValueError("All thresholds must be finite and in [-1, 1].")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Thresholds must be unique.")
    if thresholds != sorted(thresholds):
        raise ValueError("Thresholds must be in strictly increasing order.")
    return thresholds


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    existing = [
        output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()
    ]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing similarity outputs: {paths}. "
            "Pass --overwrite to replace them."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.4, 0.5, 0.6, 0.7, 0.8],
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(args.adapters) < 2:
        raise ValueError("At least two adapters are required.")
    thresholds = validate_thresholds(args.thresholds)
    prepare_output_dir(args.output_dir, args.overwrite)

    adapters = {}
    for index, path in enumerate(args.adapters):
        path = path.resolve()
        for required_name in ("adapter_config.json", "adapter_model.safetensors"):
            required_path = path / required_name
            if not required_path.is_file():
                raise FileNotFoundError(required_path)
        name = path.name or f"adapter{index}"
        if name in adapters:
            raise ValueError(f"Duplicate adapter name: {name}")
        adapters[name] = path
    _, rows = pairwise_delta_cosine(adapters)
    for row in rows:
        similarity = float(row["cosine_similarity"])
        row.update(
            {
                f"eligible_at_{threshold:g}": similarity >= threshold
                for threshold in thresholds
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "pairwise_lora_similarity.csv", index=False)

    threshold_rows = []
    for threshold in thresholds:
        eligible = result[f"eligible_at_{threshold:g}"]
        threshold_rows.append(
            {
                "threshold": threshold,
                "pairs": len(result),
                "eligible_pairs": int(eligible.sum()),
                "all_pairs_eligible": bool(eligible.all()),
                "minimum_similarity": result["cosine_similarity"].min(),
            }
        )
    pd.DataFrame(threshold_rows).to_csv(
        args.output_dir / "lora_threshold_summary.csv", index=False
    )
    report = {
        "analysis": "pairwise_lora_update_cosine_threshold_eligibility",
        "adapters": {name: str(path) for name, path in adapters.items()},
        "thresholds": thresholds,
        "pair_count": len(result),
        "minimum_similarity": float(result["cosine_similarity"].min()),
        "maximum_similarity": float(result["cosine_similarity"].max()),
        "artifacts": list(OUTPUT_FILES[:2]),
    }
    (args.output_dir / "lora_similarity_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
