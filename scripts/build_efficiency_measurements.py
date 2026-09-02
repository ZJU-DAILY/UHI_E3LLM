#!/usr/bin/env python3
"""Aggregate workflow-generated raw timing records for Fig. 7c/7d."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DIRECT_KEYS = {
    ("training", "UHI-E3LLM", "SFT"),
    ("training", "UHI-E3LLM", "PPO"),
    ("training", "ARMD", "train"),
    ("training", "FilterNet", "train"),
    ("training", "XGBoost", "train"),
    ("training", "RandomForest", "train"),
    ("training", "LightGBM", "train"),
    ("training", "MLP", "train"),
}
INFERENCE_METHODS = {
    "UHI-E3LLM", "ARMD", "FilterNet", "ST-MTM",
    "XGBoost", "RandomForest", "LightGBM", "MLP",
}


def load_records(raw_dir: Path) -> dict[tuple[str, str, str], dict]:
    records: dict[tuple[str, str, str], dict] = {}
    for path in sorted(raw_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        required = {"kind", "method", "stage", "value", "unit", "hardware", "note"}
        missing = required.difference(record)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        key = (record["kind"], record["method"], record["stage"])
        if key in records:
            raise ValueError(f"duplicate raw efficiency record for {key}")
        value = float(record["value"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{path}: value must be finite and positive")
        records[key] = record
    return records


def summed_record(records: dict, keys: list[tuple[str, str, str]], stage: str) -> dict:
    missing = [key for key in keys if key not in records]
    if missing:
        raise ValueError(f"missing efficiency records needed for {stage}: {missing}")
    selected = [records[key] for key in keys]
    if {record["unit"] for record in selected} != {"seconds"}:
        raise ValueError(f"{stage} inputs must use seconds")
    hardware = selected[0]["hardware"]
    if any(record["hardware"] != hardware for record in selected[1:]):
        hardware = "mixed; see raw records"
    return {
        "kind": "training",
        "method": "UHI-E3LLM" if stage == "Total" else "ST-MTM",
        "stage": stage,
        "value": sum(float(record["value"]) for record in selected),
        "unit": "seconds",
        "hardware": hardware,
        "note": " + ".join(f"{record['method']} {record['stage']}" for record in selected),
    }


def aggregate(raw_dir: Path) -> pd.DataFrame:
    records = load_records(raw_dir)
    required = set(DIRECT_KEYS)
    required.update(("inference", method, "inference") for method in INFERENCE_METHODS)
    required.update({
        ("training", "ST-MTM", "pretrain"),
        ("training", "ST-MTM", "finetune"),
    })
    missing = sorted(required.difference(records))
    if missing:
        raise ValueError(
            "efficiency workflow is incomplete; missing raw records: "
            + ", ".join(map(str, missing))
        )

    rows = [records[key] for key in sorted(DIRECT_KEYS)]
    rows.append(summed_record(
        records,
        [
            ("training", "ST-MTM", "pretrain"),
            ("training", "ST-MTM", "finetune"),
        ],
        "train",
    ))
    rows.append(summed_record(
        records,
        [
            ("training", "UHI-E3LLM", "SFT"),
            ("training", "UHI-E3LLM", "PPO"),
        ],
        "Total",
    ))
    rows.extend(
        records[("inference", method, "inference")]
        for method in sorted(INFERENCE_METHODS)
    )
    columns = ["kind", "method", "stage", "value", "unit", "hardware", "note"]
    return pd.DataFrame([{column: row[column] for column in columns} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = aggregate(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {len(table)} efficiency rows to {args.output}")


if __name__ == "__main__":
    main()
