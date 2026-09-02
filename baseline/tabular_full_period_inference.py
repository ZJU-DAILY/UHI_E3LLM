#!/usr/bin/env python3
"""Evaluate fitted tabular baselines on the 2004-2020 test-city records."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from common import load_data
from tabular_baselines import TorchMLP, predict_torch


MODEL_FILES = {
    "xgb": "xgb_n1000_depth16_lr001.json",
    "lgb": "lgb_gbdt_n500_lr0023_leaves255.txt",
    "rf": "rf_n500_bootstrap_false.joblib",
    "mlp": "mlp_384_192_96_epoch2800.pt",
}


def load_predictions(
    model_name: str,
    model_dir: Path,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model_path = model_dir / MODEL_FILES[model_name]
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing fitted {model_name} model: {model_path}")

    if model_name == "xgb":
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return np.asarray(model.predict(features), dtype=float)
    if model_name == "lgb":
        return np.asarray(lgb.Booster(model_file=str(model_path)).predict(features))
    if model_name == "rf":
        return np.asarray(joblib.load(model_path).predict(features), dtype=float)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = TorchMLP(int(checkpoint["in_features"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    return np.asarray(predict_torch(model, tensor, batch_size), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset-path", type=Path, default=project_root / "data")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root / "outputs" / "baselines" / "tabular" / "models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "full_period" / "baselines" / "tabular",
    )
    parser.add_argument(
        "--models", nargs="+", choices=sorted(MODEL_FILES), default=sorted(MODEL_FILES)
    )
    parser.add_argument("--torch-device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    scaler_path = args.model_dir / "standard_scaler.joblib"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Missing fitted feature scaler: {scaler_path}")
    features, targets, cities = load_data(args.dataset_path, mode="test_full")
    years = features[:, -2].astype(int)
    months = features[:, -1].astype(int)
    scaled = joblib.load(scaler_path).transform(features).astype(np.float32)
    device = torch.device(args.torch_device)

    predictions_dir = args.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        predicted = load_predictions(
            model_name, args.model_dir, scaled, device, args.batch_size
        )
        if predicted.shape != targets.shape or not np.isfinite(predicted).all():
            raise RuntimeError(
                f"{model_name}: invalid prediction shape or non-finite values"
            )
        frame = pd.DataFrame(
            {
                "city": cities.astype(int),
                "target_year": years,
                "target_month": months,
                "true": targets,
                "pred": predicted,
            }
        )
        counts = frame.groupby("city").size()
        if len(frame) != 20400 or len(counts) != 100 or not counts.eq(204).all():
            raise RuntimeError(
                f"{model_name}: expected 100 cities x 204 months; "
                f"observed {len(frame)} rows across {len(counts)} cities"
            )
        error = frame["pred"] - frame["true"]
        frame["error"] = error
        frame["abs_error"] = error.abs()
        frame["sq_error"] = error.pow(2)
        output = predictions_dir / f"{model_name}_test_predictions.csv"
        frame.sort_values(["city", "target_year", "target_month"]).to_csv(
            output, index=False
        )
        print(f"[ok] {model_name}: {output}")


if __name__ == "__main__":
    main()
