#!/usr/bin/env python3
"""Run fixed-parameter ML baselines on leave-one-region-out splits."""

import argparse
import os
import random
import shutil
import time
from pathlib import Path

DEFAULT_TMP = str(Path(__file__).resolve().parents[1] / "outputs" / ".tmp")
os.makedirs(DEFAULT_TMP, exist_ok=True)
os.environ.setdefault("TMPDIR", DEFAULT_TMP)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(DEFAULT_TMP, "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

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


REGIONS = [
    "Africa",
    "EastAsiaOceania",
    "Europe",
    "LatinAmerica",
    "NorthAmerica",
    "SCWAsia",
]


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_region_dataset(region, dataset_root, fold_root, data_dir):
    region_dir = Path(dataset_root) / region
    metrics_dir = region_dir / "city_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_link = region_dir / "train"
    valid_link = region_dir / "valid"
    if not train_link.exists():
        train_link.symlink_to(Path(data_dir).resolve() / "train")
    if not valid_link.exists():
        valid_link.symlink_to(Path(data_dir).resolve() / "valid")

    for split in ["train", "test"]:
        dst = metrics_dir / f"city_metrics_{split}.csv"
        if dst.exists():
            continue
        src = Path(fold_root) / f"fold_{region}_{split}.csv"
        if not src.exists():
            raise FileNotFoundError(f"Missing fold file: {src}")
        shutil.copy2(src, dst)
        print(f"[{region}] copied {src} -> {dst}", flush=True)

    return region_dir


def metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def per_city_metrics(y_true, y_pred, cities):
    df = pd.DataFrame({"city": cities, "true": y_true, "pred": y_pred})

    def city_row(group):
        m = metrics(group["true"], group["pred"])
        return pd.Series({"samples": len(group), **m})

    return df.groupby("city", sort=True).apply(city_row).reset_index()


def write_predictions(path, cities, y_true, y_pred, target_year, target_month):
    pd.DataFrame(
        {
            "city": cities,
            "target_year": target_year,
            "target_month": target_month,
            "true": y_true,
            "pred": y_pred,
            "error": np.asarray(y_pred) - np.asarray(y_true),
            "abs_error": np.abs(np.asarray(y_pred) - np.asarray(y_true)),
        }
    ).to_csv(path, index=False)


def fit_xgb_model(X_train, y_train, X_test, args):
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
    return model.predict(X_train), model.predict(X_test)


def fit_lgb_model(X_train, y_train, X_test, args):
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "n_estimators": 500,
        "learning_rate": 0.0023,
        "num_leaves": 255,
        "max_depth": 16,
        "min_child_samples": 5,
        "subsample": 1.0,
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
    return model.predict(X_train), model.predict(X_test)


def fit_rf_model(X_train, y_train, X_test, args):
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
    return model.predict(X_train), model.predict(X_test)


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
            preds.append(model(X_t[start:start + batch_size]).detach().cpu().numpy())
    return np.concatenate(preds)


def fit_mlp_model(X_train, y_train, X_test, args):
    device = torch.device(args.torch_device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    model = TorchMLP(X_train.shape[1]).to(device)
    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    loss_fn = nn.MSELoss()
    batch_size = min(2048, X_train_t.shape[0])

    for epoch in range(1, 2801):
        model.train()
        perm = torch.randperm(X_train_t.shape[0], device=device)
        total = 0.0
        seen = 0
        for start in range(0, X_train_t.shape[0], batch_size):
            idx = perm[start:start + batch_size]
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
                f"    MLP epoch={epoch}/2800 mse={total / max(seen, 1):.6f} train_mae={train_mae:.6f}",
                flush=True,
            )

    return (
        predict_torch(model, X_train_t, args.eval_batch_size),
        predict_torch(model, X_test_t, args.eval_batch_size),
    )


def fit_model(model_name, X_train, y_train, X_test, args):
    if model_name == "xgb":
        return fit_xgb_model(X_train, y_train, X_test, args)
    if model_name == "lgb":
        return fit_lgb_model(X_train, y_train, X_test, args)
    if model_name == "rf":
        return fit_rf_model(X_train, y_train, X_test, args)
    if model_name == "mlp":
        return fit_mlp_model(X_train, y_train, X_test, args)
    raise ValueError(f"Unknown model: {model_name}")


def run_one(model_name, region, args):
    print(f"=== {model_name} / {region} ===", flush=True)
    region_dir = ensure_region_dataset(
        region, args.dataset_root, args.fold_root, args.data_dir
    )

    X_train, y_train, city_train = load_data(str(region_dir), mode="train")
    X_test, y_test, city_test = load_data(str(region_dir), mode="test")
    train_dates = (X_train[:, -2].astype(int), X_train[:, -1].astype(int))
    test_dates = (X_test[:, -2].astype(int), X_test[:, -1].astype(int))
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    print(f"  Train: {X_train_s.shape}; Test: {X_test_s.shape}", flush=True)

    t0 = time.time()
    pred_train, pred_test = fit_model(model_name, X_train_s, y_train, X_test_s, args)
    elapsed = time.time() - t0

    train_m = metrics(y_train, pred_train)
    test_m = metrics(y_test, pred_test)
    city_test_df = per_city_metrics(y_test, pred_test, city_test)
    city_test_mean = {
        "city_test_mae_mean": float(city_test_df["mae"].mean()),
        "city_test_mse_mean": float(city_test_df["mse"].mean()),
        "city_test_rmse_mean": float(city_test_df["rmse"].mean()),
    }

    pred_dir = Path(args.output_dir) / "predictions" / model_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(
        pred_dir / f"{region}_train_predictions.csv",
        city_train,
        y_train,
        pred_train,
        *train_dates,
    )
    write_predictions(
        pred_dir / f"{region}_test_predictions.csv",
        city_test,
        y_test,
        pred_test,
        *test_dates,
    )

    city_dir = Path(args.output_dir) / "per_city" / model_name
    city_dir.mkdir(parents=True, exist_ok=True)
    city_test_df.insert(0, "model", model_name)
    city_test_df.insert(1, "region", region)
    city_test_df.to_csv(city_dir / f"{region}_test_city_metrics.csv", index=False)

    row = {
        "model": model_name,
        "region": region,
        "train_cities": int(len(np.unique(city_train))),
        "test_cities": int(len(np.unique(city_test))),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "train_mae": train_m["mae"],
        "train_mse": train_m["mse"],
        "train_rmse": train_m["rmse"],
        "test_mae": test_m["mae"],
        "test_mse": test_m["mse"],
        "test_rmse": test_m["rmse"],
        **city_test_mean,
        "elapsed_sec": elapsed,
        "status": "ok",
        "error": "",
    }
    print(
        f"  done: train_mae={row['train_mae']:.6f} test_mae={row['test_mae']:.6f} "
        f"test_rmse={row['test_rmse']:.6f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return row


def write_model_summary(model_name, rows, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"summary_{model_name}.csv"
    current = pd.DataFrame(rows)
    if summary_path.exists() and "region" in current.columns:
        previous = pd.read_csv(summary_path)
        replaced_regions = set(current["region"].dropna())
        previous = previous.loc[~previous["region"].isin(replaced_regions)]
        df = pd.concat([previous, current], ignore_index=True)
    else:
        df = current
    df = df.sort_values("region").reset_index(drop=True)
    df.to_csv(summary_path, index=False)
    per_city_files = [
        Path(output_dir) / "per_city" / model_name / f"{row['region']}_test_city_metrics.csv"
        for row in df.to_dict("records")
        if row.get("status") == "ok"
    ]
    existing = [pd.read_csv(path) for path in per_city_files if path.exists()]
    if existing:
        pd.concat(existing, ignore_index=True).to_csv(
            out_dir / f"per_city_{model_name}.csv", index=False
        )


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="+", default=REGIONS, choices=REGIONS)
    parser.add_argument("--models", nargs="+", default=["xgb", "lgb", "rf", "mlp"], choices=["xgb", "lgb", "rf", "mlp"])
    parser.add_argument("--data_dir", default=str(project_root / "data"))
    parser.add_argument(
        "--dataset_root",
        default=str(project_root / "outputs" / "region_datasets"),
    )
    parser.add_argument(
        "--fold_root", default=str(project_root / "data" / "regional_folds")
    )
    parser.add_argument(
        "--output_dir",
        default=str(project_root / "outputs" / "baselines" / "regional"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=32)
    parser.add_argument("--xgb_device", default="cuda")
    parser.add_argument("--lgb_device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--lgb_gpu_platform_id", type=int, default=0)
    parser.add_argument("--lgb_gpu_device_id", type=int, default=0)
    parser.add_argument("--torch_device", default="cuda:0")
    parser.add_argument("--eval_batch_size", type=int, default=8192)
    parser.add_argument("--progress_interval", type=int, default=100)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TMPDIR", str(Path(args.output_dir) / "tmp"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Fixed-parameter CV ML baselines", flush=True)
    print(f"models={args.models}", flush=True)
    print(f"regions={args.regions}", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"TMPDIR={os.environ.get('TMPDIR')}", flush=True)
    print("", flush=True)

    if args.dry_run:
        for region in args.regions:
            region_dir = ensure_region_dataset(
                region, args.dataset_root, args.fold_root, args.data_dir
            )
            print(f"would use {region_dir}", flush=True)
        return

    for model_name in args.models:
        set_seeds(args.seed)
        rows = []
        for region in args.regions:
            try:
                rows.append(run_one(model_name, region, args))
            except Exception as exc:
                row = {
                    "model": model_name,
                    "region": region,
                    "status": "failed",
                    "error": repr(exc),
                }
                rows.append(row)
                print(f"  FAILED {model_name}/{region}: {exc!r}", flush=True)
            write_model_summary(model_name, rows, args.output_dir)


if __name__ == "__main__":
    main()
