#!/usr/bin/env python3
"""Run inference with the tabular models fitted by the main evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from common import load_data
from tabular_baselines import (
    TorchMLP,
    area_group_metrics,
    metrics,
    per_city_metrics,
    predict_torch,
    prediction_frame,
)


MODEL_FILES = {
    "xgb": "xgb_n1000_depth16_lr001.json",
    "lgb": "lgb_gbdt_n500_lr0023_leaves255.txt",
    "rf": "rf_n500_bootstrap_false.joblib",
    "mlp": "mlp_384_192_96_epoch2800.pt",
}


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required fitted baseline artifact is missing: {path}. "
            "Run scripts/run_main_evaluation.sh first."
        )
    return path


def load_predictions(model_name: str, model_dir: Path, features, args):
    model_path = require_file(model_dir / MODEL_FILES[model_name])
    if model_name == "xgb":
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        model.set_params(device=args.xgb_device)
        return model.predict(features)
    if model_name == "lgb":
        return lgb.Booster(model_file=str(model_path)).predict(features)
    if model_name == "rf":
        return joblib.load(model_path).predict(features)

    device = torch.device(args.torch_device)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    model = TorchMLP(int(checkpoint["in_features"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    return predict_torch(model, tensor, args.eval_batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=project_root / "outputs" / "baselines" / "tabular" / "models",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgb", "lgb", "rf", "mlp"],
        choices=sorted(MODEL_FILES),
    )
    parser.add_argument("--xgb_device", default="cuda")
    parser.add_argument("--torch_device", default="cuda:0")
    parser.add_argument("--eval_batch_size", type=int, default=8192)
    parser.add_argument("--expected_cities", type=int, default=100)
    parser.add_argument("--expected_samples_per_city", type=int, default=36)
    args = parser.parse_args()

    scaler = joblib.load(require_file(args.model_dir / "standard_scaler.joblib"))
    features, targets, cities = load_data(args.dataset_path, mode="test")
    city_counts = pd.Series(cities).value_counts()
    if (
        len(city_counts) != args.expected_cities
        or not city_counts.eq(args.expected_samples_per_city).all()
    ):
        raise RuntimeError(
            f"Expected {args.expected_cities} cities x "
            f"{args.expected_samples_per_city} samples; found {len(city_counts)} "
            f"cities and {len(targets)} samples."
        )
    years = features[:, -2].astype(int)
    months = features[:, -1].astype(int)
    features_scaled = scaler.transform(features).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    per_city_dir = args.output_dir / "per_city"
    area_dir = args.output_dir / "area_groups"
    for directory in (prediction_dir, per_city_dir, area_dir):
        directory.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_name in args.models:
        predictions = np.asarray(
            load_predictions(model_name, args.model_dir, features_scaled, args),
            dtype=float,
        )
        if predictions.shape != targets.shape or not np.isfinite(predictions).all():
            raise RuntimeError(
                f"{model_name} returned invalid prediction shape/values: "
                f"shape={predictions.shape}, expected={targets.shape}."
            )
        frame = prediction_frame(cities, targets, predictions)
        frame.insert(1, "target_year", years)
        frame.insert(2, "target_month", months)
        frame.to_csv(prediction_dir / f"{model_name}_test_predictions.csv", index=False)

        city_frame = per_city_metrics(frame)
        city_frame.insert(0, "model", model_name)
        city_frame.to_csv(per_city_dir / f"{model_name}_test_city_metrics.csv", index=False)
        area_frame = area_group_metrics(
            frame, args.dataset_path / "city_metrics" / "city_metrics_test.csv"
        )
        area_frame.insert(0, "model", model_name)
        area_frame.to_csv(area_dir / f"{model_name}_area_group_metrics.csv", index=False)
        summaries.append({"model": model_name, "samples": len(frame), **metrics(targets, predictions)})

    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    metadata = {
        "mode": "inference_only",
        "dataset_path": str(args.dataset_path.resolve()),
        "source_model_dir": str(args.model_dir.resolve()),
        "models": args.models,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
