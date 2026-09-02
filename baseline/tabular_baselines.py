#!/usr/bin/env python3
"""Run fixed-parameter ML baselines on the original 100 test cities.

This script uses the final post-review hyperparameters while also saving
predictions, per-city metrics, area-group metrics, scalers, and model files.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn

from common import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from efficiency import (  # noqa: E402
    benchmark_model_call,
    detected_hardware,
    synchronize_cuda,
    write_measurement,
)


AREA_BINS = [100, 150, 450, 800, 1500, np.inf]
AREA_LABELS = ["Very small", "Small", "Medium", "Large", "Very large"]


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def prediction_frame(cities, y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return pd.DataFrame(
        {
            "city": np.asarray(cities),
            "true": y_true,
            "pred": y_pred,
            "error": y_pred - y_true,
            "abs_error": np.abs(y_pred - y_true),
            "sq_error": (y_pred - y_true) ** 2,
        }
    )


def per_city_metrics(pred_df):
    rows = []
    for city, group in pred_df.groupby("city", sort=True):
        m = metrics(group["true"], group["pred"])
        rows.append({"city": int(city), "samples": int(len(group)), **m})
    return pd.DataFrame(rows)


def area_group_metrics(pred_df, city_metrics_path):
    city_info = pd.read_csv(city_metrics_path)[["id", "Area"]]
    df = pred_df.merge(city_info, left_on="city", right_on="id", how="left")
    df["area_group"] = pd.cut(
        df["Area"],
        bins=AREA_BINS,
        labels=AREA_LABELS,
        right=False,
        include_lowest=True,
    )
    rows = []
    for group_name, group in df.groupby("area_group", observed=False):
        if group.empty:
            rows.append(
                {
                    "area_group": str(group_name),
                    "cities": 0,
                    "samples": 0,
                    "mae": np.nan,
                    "mse": np.nan,
                    "rmse": np.nan,
                }
            )
            continue
        m = metrics(group["true"], group["pred"])
        rows.append(
            {
                "area_group": str(group_name),
                "cities": int(group["city"].nunique()),
                "samples": int(len(group)),
                **m,
            }
        )
    return pd.DataFrame(rows)


def fit_xgb(X_train, y_train, args):
    model = xgb.XGBRegressor(
        booster="gbtree",
        objective="reg:squarederror",
        n_estimators=1000,
        max_depth=16,
        learning_rate=0.001,
        subsample=1.0,
        colsample_bytree=1.0,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        device=args.xgb_device,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def fit_lgb(X_train, y_train, args):
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "n_estimators": 500,
        "learning_rate": 0.0023,
        "num_leaves": 255,
        "max_depth": 16,
        "min_child_samples": 5,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "min_split_gain": 0.0,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
        "verbose": -1,
    }
    if args.lgb_device == "gpu":
        params.update(
            {
                "device_type": "gpu",
                "gpu_platform_id": args.lgb_gpu_platform_id,
                "gpu_device_id": args.lgb_gpu_device_id,
            }
        )
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model


def fit_rf(X_train, y_train, args):
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=False,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=0,
    )
    model.fit(X_train, y_train)
    return model


class TorchMLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        layers = []
        last = in_features
        for width in (384, 192, 96):
            layers.append(nn.Linear(last, width))
            layers.append(nn.ReLU())
            last = width
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict_torch(model, X_t, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, X_t.shape[0], batch_size):
            preds.append(model(X_t[start : start + batch_size]).detach().cpu().numpy())
    return np.concatenate(preds)


def fit_mlp(X_train, y_train, args):
    device = torch.device(args.torch_device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    model = TorchMLP(X_train.shape[1]).to(device)
    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    loss_fn = nn.MSELoss()
    batch_size = min(2048, X_train_t.shape[0])

    for epoch in range(1, 2801):
        model.train()
        perm = torch.randperm(X_train_t.shape[0], device=device)
        total = 0.0
        seen = 0
        for start in range(0, X_train_t.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            xb = X_train_t.index_select(0, idx)
            yb = y_train_t.index_select(0, idx)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            seen += xb.shape[0]
        if args.progress_interval and epoch % args.progress_interval == 0:
            pred_train = predict_torch(model, X_train_t, args.eval_batch_size)
            train_mae = mean_absolute_error(y_train, pred_train)
            print(
                f"    MLP epoch={epoch}/2800 mse={total / max(seen, 1):.6f} "
                f"train_mae={train_mae:.6f}",
                flush=True,
            )

    return model


def predict_model(model_name, model, X, args):
    if model_name != "mlp":
        return model.predict(X)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=args.torch_device)
    return predict_torch(model, X_t, args.eval_batch_size)


def write_efficiency_records(model_name, model, X_test, elapsed, args):
    if not args.efficiency_record_dir:
        return
    display_names = {
        "xgb": "XGBoost",
        "lgb": "LightGBM",
        "rf": "RandomForest",
        "mlp": "MLP",
    }
    method = display_names[model_name]
    record_dir = Path(args.efficiency_record_dir)
    training_device = "cpu"
    if model_name == "mlp":
        training_device = args.torch_device
    elif model_name == "xgb":
        training_device = args.xgb_device
    elif model_name == "lgb":
        training_device = args.lgb_device
    write_measurement(
        record_dir / f"training_{model_name}.json",
        kind="training",
        method=method,
        stage="train",
        value=elapsed,
        unit="seconds",
        hardware=detected_hardware(training_device),
        note="model fit/optimization only; excludes data loading, scaling, prediction, metrics, and file I/O",
        metadata={"training_samples": int(args.train_sample_count)},
    )

    requested_batch_size = (
        args.efficiency_gpu_batch_size
        if model_name == "mlp"
        else args.efficiency_batch_size
    )
    batch_size = min(requested_batch_size, len(X_test))
    X_batch = X_test[:batch_size]
    if model_name == "mlp":
        X_batch_t = torch.as_tensor(
            X_batch, dtype=torch.float32, device=args.torch_device
        )
        call = lambda: model(X_batch_t)
        hardware = detected_hardware(args.torch_device)
    else:
        old_n_jobs = model.get_params().get("n_jobs")
        model.set_params(n_jobs=1)
        old_device = model.get_params().get("device")
        if model_name == "xgb":
            model.set_params(device="cpu")
        call = lambda: model.predict(X_batch)
        hardware = detected_hardware("cpu")
    try:
        latency = benchmark_model_call(
            call,
            sample_count=batch_size,
            warmup_calls=args.efficiency_warmup_calls,
            measured_calls=args.efficiency_measured_calls,
        )
    finally:
        if model_name != "mlp":
            model.set_params(n_jobs=old_n_jobs)
            if model_name == "xgb":
                model.set_params(device=old_device)
    write_measurement(
        record_dir / f"inference_{model_name}.json",
        kind="inference",
        method=method,
        stage="inference",
        value=latency,
        unit="ms_per_sample",
        hardware=hardware,
        note="model execution only; excludes loading, scaling, metrics, and file I/O",
        metadata={
            "batch_size": batch_size,
            "warmup_calls": args.efficiency_warmup_calls,
            "measured_calls": args.efficiency_measured_calls,
            "timing_protocol": "independent_repetitions_mean",
            "cpu_threads": 1 if model_name != "mlp" else None,
        },
    )


def save_model(model_name, model, scaler, out_dir, args):
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "standard_scaler.joblib")
    if model_name == "xgb":
        model.save_model(model_dir / "xgb_n1000_depth16_lr001.json")
    elif model_name == "lgb":
        model.booster_.save_model(str(model_dir / "lgb_gbdt_n500_lr0023_leaves255.txt"))
    elif model_name == "rf":
        joblib.dump(model, model_dir / "rf_n500_bootstrap_false.joblib")
    elif model_name == "mlp":
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_features": args.in_features,
                "hidden_layers": [384, 192, 96],
                "epochs": 2800,
                "learning_rate": 1e-4,
                "batch_size": 2048,
            },
            model_dir / "mlp_384_192_96_epoch2800.pt",
        )


def run_model(model_name, X_train, y_train, city_train, X_test, y_test, city_test, scaler, args,
              train_dates=None, test_dates=None):
    print(f"=== {model_name} ===", flush=True)
    set_seeds(args.seed)
    synchronize_cuda()
    t0 = time.perf_counter()
    if model_name == "xgb":
        model = fit_xgb(X_train, y_train, args)
    elif model_name == "lgb":
        model = fit_lgb(X_train, y_train, args)
    elif model_name == "rf":
        model = fit_rf(X_train, y_train, args)
    elif model_name == "mlp":
        model = fit_mlp(X_train, y_train, args)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    synchronize_cuda()
    elapsed = time.perf_counter() - t0
    write_efficiency_records(model_name, model, X_test, elapsed, args)
    pred_train = predict_model(model_name, model, X_train, args)
    pred_test = predict_model(model_name, model, X_test, args)

    train_df = prediction_frame(city_train, y_train, pred_train)
    test_df = prediction_frame(city_test, y_test, pred_test)
    for frame, dates in ((train_df, train_dates), (test_df, test_dates)):
        if dates is not None:
            years, months = dates
            frame.insert(1, "target_year", np.asarray(years, dtype=int))
            frame.insert(2, "target_month", np.asarray(months, dtype=int))
    train_m = metrics(y_train, pred_train)
    test_m = metrics(y_test, pred_test)
    city_df = per_city_metrics(test_df)
    area_df = area_group_metrics(
        test_df,
        Path(args.dataset_path) / "city_metrics" / "city_metrics_test.csv",
    )

    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_city").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "area_groups").mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.output_dir / "predictions" / f"{model_name}_train_predictions.csv", index=False)
    test_df.to_csv(args.output_dir / "predictions" / f"{model_name}_test_predictions.csv", index=False)
    city_df.insert(0, "model", model_name)
    city_df.to_csv(args.output_dir / "per_city" / f"{model_name}_test_city_metrics.csv", index=False)
    area_df.insert(0, "model", model_name)
    area_df.to_csv(args.output_dir / "area_groups" / f"{model_name}_area_group_metrics.csv", index=False)
    save_model(model_name, model, scaler, args.output_dir, args)

    row = {
        "model": model_name,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "train_mae": train_m["mae"],
        "train_mse": train_m["mse"],
        "train_rmse": train_m["rmse"],
        "test_mae": test_m["mae"],
        "test_mse": test_m["mse"],
        "test_rmse": test_m["rmse"],
        "elapsed_sec": elapsed,
    }
    print(
        f"  done: train_mae={row['train_mae']:.6f} test_mae={row['test_mae']:.6f} "
        f"test_mse={row['test_mse']:.6f} test_rmse={row['test_rmse']:.6f} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset_path", default=str(project_root / "data"))
    parser.add_argument(
        "--output_dir",
        default=str(project_root / "outputs" / "baselines" / "tabular"),
    )
    parser.add_argument("--models", nargs="+", default=["xgb", "lgb", "rf", "mlp"], choices=["xgb", "lgb", "rf", "mlp"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=64)
    parser.add_argument("--xgb_device", default="cuda")
    parser.add_argument("--lgb_device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--lgb_gpu_platform_id", type=int, default=0)
    parser.add_argument("--lgb_gpu_device_id", type=int, default=0)
    parser.add_argument("--torch_device", default="cuda:0")
    parser.add_argument("--eval_batch_size", type=int, default=8192)
    parser.add_argument("--progress_interval", type=int, default=100)
    parser.add_argument("--efficiency_record_dir", type=str, default="")
    parser.add_argument("--efficiency_batch_size", type=int, default=128)
    parser.add_argument("--efficiency_gpu_batch_size", type=int, default=36)
    parser.add_argument("--efficiency_warmup_calls", type=int, default=5)
    parser.add_argument("--efficiency_measured_calls", type=int, default=5)
    args = parser.parse_args()
    if args.efficiency_batch_size <= 0 or args.efficiency_gpu_batch_size <= 0:
        parser.error("efficiency batch sizes must be positive")
    if args.efficiency_warmup_calls < 0 or args.efficiency_measured_calls <= 0:
        parser.error("efficiency warm-up/measured calls are invalid")
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Fixed-parameter ML baselines on original 100 test cities", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"models={args.models}", flush=True)

    set_seeds(args.seed)
    X_train, y_train, city_train = load_data(args.dataset_path, mode="train")
    X_test, y_test, city_test = load_data(args.dataset_path, mode="test")
    # flatten_sample appends year and month as the final two (unscaled) features.
    train_dates = (X_train[:, -2].astype(int), X_train[:, -1].astype(int))
    test_dates = (X_test[:, -2].astype(int), X_test[:, -1].astype(int))
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    args.in_features = int(X_train_s.shape[1])
    args.train_sample_count = int(len(y_train))
    print(f"Train: {X_train_s.shape}; Test: {X_test_s.shape}", flush=True)

    metadata = {
        "dataset_path": args.dataset_path,
        "seed": args.seed,
        "models": args.models,
        "xgb_device": args.xgb_device,
        "lgb_device": args.lgb_device,
        "lgb_gpu_platform_id": args.lgb_gpu_platform_id,
        "lgb_gpu_device_id": args.lgb_gpu_device_id,
        "torch_device": args.torch_device,
        "n_jobs": args.n_jobs,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    rows = []
    area_parts = []
    city_parts = []
    for model_name in args.models:
        row = run_model(
            model_name,
            X_train_s,
            y_train,
            city_train,
            X_test_s,
            y_test,
            city_test,
            scaler,
            args,
            train_dates=train_dates,
            test_dates=test_dates,
        )
        rows.append(row)
        area_parts.append(pd.read_csv(args.output_dir / "area_groups" / f"{model_name}_area_group_metrics.csv"))
        city_parts.append(pd.read_csv(args.output_dir / "per_city" / f"{model_name}_test_city_metrics.csv"))
        pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)
        pd.concat(area_parts, ignore_index=True).to_csv(args.output_dir / "area_group_summary.csv", index=False)
        pd.concat(city_parts, ignore_index=True).to_csv(args.output_dir / "per_city_summary.csv", index=False)

    print("All done.", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
