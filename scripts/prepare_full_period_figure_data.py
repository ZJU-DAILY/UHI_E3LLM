#!/usr/bin/env python3
"""Prepare the eight-method 2004-2020 metrics required by Fig. 3b."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "UHI-E3LLM", "ARMD", "FilterNet", "ST-MTM",
    "XGBoost", "RandomForest", "LightGBM", "MLP",
}


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[run] cwd={cwd} command={shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Required fitted model is missing: {path}\n"
            "Run scripts/run_main_evaluation.sh before generating the figures."
        )


def validate_existing(path: Path) -> bool:
    if not path.is_file():
        return False
    table = pd.read_csv(path)
    required = {"method", "city_idx", "target_year", "samples", "mae", "mse", "rmse"}
    if missing := required.difference(table.columns):
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    counts = table.groupby("method").size()
    years = set(pd.to_numeric(table["target_year"], errors="raise").astype(int))
    if (
        set(table["method"]) != METHODS
        or len(table) != 13600
        or not counts.eq(1700).all()
        or years != set(range(2004, 2021))
        or not table["samples"].eq(12).all()
    ):
        raise ValueError(
            f"{path}: expected 13,600 rows "
            "(8 methods x 100 cities x 17 years; 12 samples per city-year)"
        )
    print(f"[reuse] validated Fig. 3b source: {path}")
    return True


def fitted_artifacts(
    output_root: Path, filternet_epochs: int, stmtm_epochs: int
) -> dict[str, Path]:
    filternet_setting = (
        f"UHI500City1passes_{filternet_epochs}epochs_PaiFilter_custom_"
        "sl12_pl1_embed12_hidden64_bs1024_lr2e-05_drop0.0_manuscript_mv24"
    )
    stmtm_setting = (
        "finetune_STMTM_Exchange_M_sl12_ll0_pl1_dm16_df64_nh4_dh64_el1_"
        f"fc1_dp0.2_hdp0.2_ep{stmtm_epochs}_bs1024_lr1.6e-05_wd0.05_"
        "ks200_sl4_pt0.2_topk3_tau0.1_alp0.5_iter_manuscript_cp1_cities500"
    )
    models = output_root / "baselines/tabular/models"
    return {
        "scaler": models / "standard_scaler.joblib",
        "xgboost": models / "xgb_n1000_depth16_lr001.json",
        "lightgbm": models / "lgb_gbdt_n500_lr0023_leaves255.txt",
        "random_forest": models / "rf_n500_bootstrap_false.joblib",
        "mlp": models / "mlp_384_192_96_epoch2800.pt",
        "armd": output_root / "baselines/armd/checkpoints_mv24_12/checkpoint-final.pt",
        "filternet": (
            output_root / "baselines/filternet/checkpoints"
            / filternet_setting / "checkpoint.pth"
        ),
        "st_mtm": (
            output_root / "baselines/st_mtm/checkpoints"
            / stmtm_setting / "checkpoint.pth"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/figures/data/method_city_year_metrics_2004_2020.csv",
    )
    parser.add_argument("--gpu-id", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", "deepseek-8b_merged__Global500"),
    )
    parser.add_argument("--filternet-epochs", type=int, default=5)
    parser.add_argument("--stmtm-finetune-epochs", type=int, default=5)
    args = parser.parse_args()

    if validate_existing(args.output):
        return

    data_dir = args.data_dir.resolve()
    output_root = args.output_root.resolve()
    full_root = output_root / "full_period"
    timeseries_data = output_root / "baselines/timeseries_data"
    train_cities = data_dir / "city_metrics/city_metrics_train.csv"
    test_cities = data_dir / "city_metrics/city_metrics_test.csv"
    distilled_root = ROOT / "model_adapters/explanation_distilled" / args.model
    artifacts = fitted_artifacts(
        output_root, args.filternet_epochs, args.stmtm_finetune_epochs
    )
    for path in (*artifacts.values(), train_cities, test_cities):
        require_file(path)
    for epoch in (0, 1):
        require_file(
            ROOT / "model" / f"{args.model}_PPO_epoch{epoch}_merged" / "config.json"
        )
        require_file(distilled_root / f"epoch{epoch}/adapter_config.json")

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
            "PYTHON": sys.executable,
            "DATA_DIR": str(data_dir),
            "OUTPUT_ROOT": str(output_root),
            "GPU_ID": str(args.gpu_id),
        }
    )
    ensemble_env = env.copy()
    ensemble_env.update(
        {
            "DISTILLED_OUTPUT_DIR": str(full_root / "main_ensemble"),
            "BASE_MODEL": args.model,
            "DISTILLED_ADAPTER_ROOT": str(distilled_root),
            "DISTILLED_MODE": "test_full",
            "DISTILLED_EXPECTED_CITIES": "100",
            "DISTILLED_EXPECTED_SAMPLES_PER_CITY": "204",
            "DISTILLED_LOG_PREFIX": "full_period",
        }
    )
    run(
        ["bash", str(ROOT / "scripts/run_distilled_ensemble_inference.sh")],
        ROOT,
        ensemble_env,
    )

    run(
        [
            sys.executable,
            str(ROOT / "baseline/tabular_full_period_inference.py"),
            "--dataset-path", str(data_dir),
            "--model-dir", str(output_root / "baselines/tabular/models"),
            "--output-dir", str(full_root / "baselines/tabular"),
            "--torch-device", "cuda:0",
        ],
        ROOT,
        env,
    )

    if not (
        (timeseries_data / "all").is_dir()
        and (timeseries_data / "multivariate_24.schema").is_file()
    ):
        run(
            [
                sys.executable,
                str(ROOT / "scripts/prepare_timeseries_baselines.py"),
                "--data_dir", str(data_dir),
                "--output_dir", str(timeseries_data),
                "--train_city_metrics", str(train_cities),
                "--test_city_metrics", str(test_cities),
            ],
            ROOT,
            env,
        )

    run(
        [
            sys.executable, "main_new.py", "--config_path", "Config/amod2.yaml",
            "--save_dir", str(full_root / "baselines/armd"), "--gpu", "0",
            "--skip_train", "--load_ckpt_path", str(artifacts["armd"]),
            "--train_city_metrics_path", str(train_cities),
            "--test_city_metrics_path", str(test_cities),
            "--uhi_train_dir", str(timeseries_data / "train"),
            "--uhi_eval_dir", str(timeseries_data / "all"),
            "--base_lr", "2e-5", "--batch_size", "64", "--weight_decay", "0.01",
            "--timesteps", "12", "--sampling_timesteps", "12",
            "--ema_decay", "0.995", "--ema_update_interval", "10",
            "--warmup_lr", "2e-5", "--warmup_steps", "500",
            "--feature_size", "24",
            "--evaluation_horizon", "1", "--evaluation_start", "200401",
            "--evaluation_end", "202012",
            "--sample_errors_dir", str(full_root / "baselines/armd/sample_errors"),
            "--overwrite_outputs",
        ],
        ROOT / "baseline/armd",
        env,
    )

    run(
        [
            sys.executable, "run.py", "--task_name", "long_term_forecast",
            "--is_training", "0", "--model_id", "UHI500", "--model", "PaiFilter",
            "--data", "custom", "--features", "M", "--target", "Amod2", "--freq", "m",
            "--seq_len", "12", "--label_len", "0", "--pred_len", "1",
            "--enc_in", "24", "--dec_in", "24", "--c_out", "24",
            "--embed_size", "12", "--hidden_size", "64", "--batch_size", "1024",
            "--learning_rate", "2e-5", "--weight_decay", "0.05", "--dropout", "0",
            "--train_epochs", str(args.filternet_epochs), "--city_passes", "1",
            "--train_city_metrics_path", str(train_cities),
            "--test_city_metrics_path", str(test_cities),
            "--uhi_data_dir", str(timeseries_data / "all"),
            "--checkpoints", str(output_root / "baselines/filternet/checkpoints"),
            "--load_checkpoint_path", str(artifacts["filternet"]),
            "--results_dir", str(full_root / "baselines/filternet"),
            "--sample_errors_dir", str(full_root / "baselines/filternet/sample_errors"),
            "--evaluation_start", "200401", "--evaluation_end", "202012",
            "--output_tag", "full_period", "--evaluate_target_only",
            "--overwrite_outputs",
        ],
        ROOT / "baseline/filternet",
        env,
    )

    run(
        [
            sys.executable, "run.py", "--task_name", "finetune", "--is_training", "0",
            "--model_id", "UHI500", "--model", "STMTM", "--data", "Exchange",
            "--features", "M", "--target", "Amod2", "--freq", "m",
            "--seq_len", "12", "--label_len", "0", "--pred_len", "1",
            "--enc_in", "24", "--dec_in", "24", "--c_out", "24",
            "--d_model", "16", "--d_ff", "64", "--d_hidden", "64",
            "--n_heads", "4", "--e_layers", "1", "--dropout", "0.2",
            "--head_dropout", "0.2", "--fc_dropout", "0", "--batch_size", "1024",
            "--num_workers", "0", "--learning_rate", "1.6e-5", "--weight_decay", "0.05",
            "--kernel_size", "200", "--seg_len", "4", "--p_tmask", "0.2",
            "--topk", "3", "--tau", "0.1", "--alpha", "0.5",
            "--component_mode", "weighted", "--trend_weight", "0.98",
            "--head_init", "xavier", "--train_epochs", str(args.stmtm_finetune_epochs),
            "--city_passes", "1", "--train_city_metrics_path", str(train_cities),
            "--test_city_metrics_path", str(test_cities),
            "--uhi_data_dir", str(timeseries_data / "all"),
            "--checkpoints", str(output_root / "baselines/st_mtm/checkpoints"),
            "--pretrain_checkpoints", str(output_root / "baselines/st_mtm/pretrain_checkpoints"),
            "--load_checkpoints", str(artifacts["st_mtm"]),
            "--results_dir", str(full_root / "baselines/st_mtm"),
            "--sample_errors_dir", str(full_root / "baselines/st_mtm/sample_errors"),
            "--evaluation_start", "200401", "--evaluation_end", "202012",
            "--output_tag", "full_period", "--target_only_metrics",
            "--overwrite_outputs",
        ],
        ROOT / "baseline/st_mtm",
        env,
    )

    run(
        [
            sys.executable,
            str(ROOT / "scripts/build_full_period_figure_data.py"),
            "--full-period-output-root", str(full_root),
            "--output", str(args.output),
        ],
        ROOT,
        env,
    )
    validate_existing(args.output)


if __name__ == "__main__":
    main()
