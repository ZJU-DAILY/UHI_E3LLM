import argparse
import importlib.metadata
import json
import logging
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from accelerate.utils import gather_object
from datasets import concatenate_datasets, load_from_disk
from peft import LoraConfig
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import PPODecorators

from efficiency import write_measurement
from output_parsing import parse_prediction
from parallel_rollout import ParallelRolloutPool
from reward import compute_ppo_reward
from utils import (
    format_numeric_plain_prompt_PPO,
    format_prompt_PPO,
    format_task_only_prompt_PPO,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _all_ranks_tensors_finite(tensors):
    tensors = [tensor for tensor in tensors if tensor is not None]
    if not tensors:
        return True
    flag = torch.ones((), dtype=torch.int32, device=tensors[0].device)
    for tensor in tensors:
        flag.mul_(torch.isfinite(tensor.detach()).all().to(dtype=torch.int32))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _sanitize_train_stats(train_stats):
    return {
        key: (
            torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.is_tensor(value) and value.is_floating_point()
            else value
        )
        for key, value in train_stats.items()
    }


def _load_local_ppo_weights(model, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    value_head_path = checkpoint_dir / "pytorch_model.bin"
    if not adapter_path.is_file() or not value_head_path.is_file():
        raise FileNotFoundError(
            f"incomplete PPO checkpoint under {checkpoint_dir}"
        )

    saved_adapter = load_safetensors(str(adapter_path), device="cpu")
    adapter_parameters = {}
    for name, parameter in model.pretrained_model.named_parameters():
        if "lora_" not in name:
            continue
        checkpoint_name = name.replace(".default.", ".")
        if checkpoint_name in adapter_parameters:
            raise ValueError(f"duplicate normalized LoRA key: {checkpoint_name}")
        adapter_parameters[checkpoint_name] = parameter

    missing = sorted(set(adapter_parameters) - set(saved_adapter))
    unexpected = sorted(set(saved_adapter) - set(adapter_parameters))
    if missing or unexpected:
        raise ValueError(
            "PPO adapter keys do not match the initialized LoRA model: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    with torch.no_grad():
        for name, parameter in adapter_parameters.items():
            value = saved_adapter[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"PPO adapter shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    value_head_state = torch.load(
        value_head_path, map_location="cpu", weights_only=True
    )
    model.post_init(dict(value_head_state))
    return len(saved_adapter)


class FiniteSafePPOTrainer(PPOTrainer):
    """Synchronously skip PPO optimizer updates with non-finite values."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_params = [
            parameter
            for parameter_group in self.optimizer.param_groups
            for parameter in parameter_group["params"]
        ]
        self.non_finite_loss_skips = 0
        self.non_finite_gradient_skips = 0

    def _skip_non_finite_update(self, train_stats, reason, device):
        self.optimizer.zero_grad()
        if reason == "loss":
            self.non_finite_loss_skips += 1
        else:
            self.non_finite_gradient_skips += 1
        if self.accelerator.is_main_process:
            logger.warning(
                "Skipping PPO optimizer update on all ranks due to non-finite %s",
                reason,
            )
        train_stats = _sanitize_train_stats(train_stats)
        train_stats["safety/skipped_non_finite_loss"] = torch.tensor(
            float(reason == "loss"), device=device
        )
        train_stats["safety/skipped_non_finite_gradient"] = torch.tensor(
            float(reason == "gradient"), device=device
        )
        return train_stats

    @PPODecorators.empty_device_cache()
    def train_minibatch(
        self,
        old_logprobs,
        values,
        logprobs,
        logits,
        vpreds,
        mask,
        advantages,
        returns,
    ):
        self.model.train()
        loss_p, loss_v, train_stats = self.loss(
            old_logprobs,
            values,
            logits,
            vpreds,
            logprobs,
            mask,
            advantages,
            returns,
        )
        loss = loss_p + loss_v
        if not _all_ranks_tensors_finite([loss]):
            return self._skip_non_finite_update(
                train_stats, "loss", loss.device
            )

        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.model_params, self.config.max_grad_norm
            )
        gradients = [
            parameter.grad
            for parameter in self.model_params
            if parameter.grad is not None
        ]
        if not _all_ranks_tensors_finite(gradients):
            return self._skip_non_finite_update(
                train_stats, "gradient", loss.device
            )

        self.optimizer.step()
        self.optimizer.zero_grad()
        train_stats["safety/skipped_non_finite_loss"] = torch.zeros(
            (), device=loss.device
        )
        train_stats["safety/skipped_non_finite_gradient"] = torch.zeros(
            (), device=loss.device
        )
        return train_stats


def _resolve_project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_model_dir(model):
    path = Path(model).expanduser()
    if path.is_absolute():
        model_dir = path
    elif (PROJECT_ROOT / path).is_dir():
        model_dir = PROJECT_ROOT / path
    else:
        model_dir = PROJECT_ROOT / "model" / path
    if not model_dir.is_dir():
        raise FileNotFoundError(f"PPO base model directory not found: {model_dir}")
    return model_dir.resolve()


def _read_city_ids(metrics_path):
    if not metrics_path.is_file():
        raise FileNotFoundError(f"training-city manifest not found: {metrics_path}")
    frame = pd.read_csv(metrics_path)
    if "id" not in frame.columns:
        raise ValueError(f"training-city manifest has no 'id' column: {metrics_path}")
    numeric_ids = pd.to_numeric(frame["id"], errors="raise")
    if numeric_ids.isna().any() or not np.all(numeric_ids == np.floor(numeric_ids)):
        raise ValueError("all training city IDs must be finite integers")
    city_ids = numeric_ids.astype(int).tolist()
    if not city_ids:
        raise ValueError("training-city manifest is empty")
    duplicate_ids = sorted(
        city_id for city_id, count in Counter(city_ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"duplicate training city IDs: {duplicate_ids[:10]}")
    return city_ids


def _load_training_dataset(
    dataset_dir, expected_train_cities=0, expected_samples_per_city=0
):
    metrics_path = dataset_dir / "city_metrics" / "city_metrics_train.csv"
    city_ids = _read_city_ids(metrics_path)
    if expected_train_cities and len(city_ids) != expected_train_cities:
        raise ValueError(
            f"expected {expected_train_cities} training cities, found {len(city_ids)} "
            f"in {metrics_path}"
        )

    city_datasets = []
    missing = []
    bad_sizes = []
    for city_id in tqdm(city_ids, desc="Loading city datasets"):
        city_path = dataset_dir / "train" / f"UHI_city_{city_id}"
        if not city_path.is_dir():
            missing.append(str(city_path))
            continue
        city_dataset = load_from_disk(str(city_path))
        if (
            expected_samples_per_city
            and len(city_dataset) != expected_samples_per_city
        ):
            bad_sizes.append((city_id, len(city_dataset)))
        city_datasets.append(city_dataset)

    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"missing {len(missing)} training city datasets; first entries:\n{preview}"
        )
    if bad_sizes:
        preview = ", ".join(f"{city_id}:{size}" for city_id, size in bad_sizes[:10])
        raise ValueError(
            f"expected {expected_samples_per_city} samples per city; "
            f"found mismatches (city_id:rows): {preview}"
        )
    return concatenate_datasets(city_datasets), city_ids, metrics_path


def _checkpoint_directories(save_root, train_epochs, start_epoch):
    if train_epochs == 1:
        return [save_root]
    return [
        save_root / f"epoch{start_epoch + offset}"
        for offset in range(train_epochs)
    ]


def _remove_path(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _prepare_checkpoint_directories(checkpoint_dirs, overwrite, resume_dir=None):
    resume_resolved = resume_dir.resolve() if resume_dir else None
    existing = []
    for checkpoint_dir in checkpoint_dirs:
        if resume_resolved and checkpoint_dir.resolve() == resume_resolved:
            raise ValueError(
                f"refusing to overwrite the resume source checkpoint: {checkpoint_dir}"
            )
        if os.path.lexists(checkpoint_dir):
            existing.append(checkpoint_dir)
    if existing and not overwrite:
        paths = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "PPO checkpoint output already exists. Choose a new PPO_OUTPUT or "
            f"pass --overwrite:\n{paths}"
        )
    for path in existing:
        _remove_path(path)


def _package_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _save_ppo_checkpoint_locally(ppo_trainer, save_dir):
    """Save a PPO checkpoint without invoking Hugging Face Hub APIs."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
    model.save_pretrained(str(save_dir))
    ppo_trainer.tokenizer.save_pretrained(str(save_dir))
    _write_json(save_dir / "config.json", ppo_trainer.config.to_dict())


def _save_latest_training_state(accelerator, state_dir, metadata):
    """Atomically replace the rolling Accelerate training-state checkpoint."""
    state_dir = Path(state_dir)
    temporary_dir = state_dir.with_name(f".{state_dir.name}.tmp")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if os.path.lexists(temporary_dir):
            _remove_path(temporary_dir)
        temporary_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(temporary_dir))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _write_json(temporary_dir / "ppo_training_state.json", metadata)
        if os.path.lexists(state_dir):
            _remove_path(state_dir)
        temporary_dir.replace(state_dir)
    accelerator.wait_for_everyone()


def _load_training_resume_metadata(state_dir, checkpoint_dir, world_size):
    state_dir = Path(state_dir).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    metadata_path = state_dir / "ppo_training_state.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"PPO training-state metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported PPO training-state schema")
    recorded_checkpoint = Path(metadata.get("adapter_checkpoint", "")).resolve()
    if recorded_checkpoint != checkpoint_dir:
        raise ValueError(
            "training state and adapter checkpoint do not match: "
            f"{recorded_checkpoint} != {checkpoint_dir}"
        )
    required_files = [state_dir / "model.safetensors", state_dir / "optimizer.bin"]
    required_files.extend(
        state_dir / f"random_states_{rank}.pkl" for rank in range(world_size)
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "incomplete PPO training state; missing:\n" + "\n".join(missing)
        )
    completed_steps = metadata.get("ppo_steps_completed_in_epoch")
    if not isinstance(completed_steps, int) or completed_steps < 0:
        raise ValueError("invalid ppo_steps_completed_in_epoch in training state")
    return state_dir, metadata


def _validate_resume_run_config(checkpoint_dir, args, world_size):
    config_path = Path(checkpoint_dir) / "ppo_run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"PPO resume run config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    expected = {
        "per_process_batch_size": args.batch_size,
        "per_process_mini_batch_size": args.mini_batch_size,
        "distributed_processes": world_size,
        "ppo_epochs_per_batch": args.ppo_epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "max_prompt_tokens": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "input_format": args.input_format,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    reward = config.get("reward", {})
    expected_reward = {
        "y_bar": args.reward_baseline,
        "lambda": args.penalty_lambda,
        "valid_min": args.valid_min,
        "valid_max": args.valid_max,
        "max_decimal_places": args.max_decimal_places,
    }
    for key, value in expected_reward.items():
        if reward.get(key) != value:
            mismatches[f"reward.{key}"] = (reward.get(key), value)
    if mismatches:
        details = ", ".join(
            f"{key}: saved={saved!r}, requested={requested!r}"
            for key, (saved, requested) in sorted(mismatches.items())
        )
        raise ValueError(f"PPO resume configuration mismatch: {details}")
    return config


def _validate_reward_baseline_report(path, reward_baseline):
    if not path.is_file():
        raise FileNotFoundError(f"reward-baseline report not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    report_baseline = report.get("y_bar")
    if not isinstance(report_baseline, (int, float)) or not np.isfinite(
        report_baseline
    ):
        raise ValueError("reward-baseline report has no finite numeric y_bar")
    if not np.isclose(float(report_baseline), reward_baseline, rtol=0, atol=1e-12):
        raise ValueError(
            f"--reward_baseline={reward_baseline} does not match report "
            f"y_bar={report_baseline}"
        )
    if report.get("statistic") != "sample-weighted training MAE":
        raise ValueError(
            "reward-baseline report must use the sample-weighted training MAE "
            "(the manuscript reference statistic e_bar)"
        )
    metrics = report.get("metrics", {})
    if not np.isclose(
        float(metrics.get("mae", float("nan"))),
        float(report_baseline),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("reward-baseline report MAE does not match y_bar")
    split = report.get("split", {})
    if split.get("held_out_test_cities_used") is not False:
        raise ValueError(
            "reward-baseline report must explicitly confirm that test cities were not used"
        )
    return report


def _build_parser():
    parser = argparse.ArgumentParser(
        description="PPO refinement for the UHI-E3LLM numerical predictor."
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=None,
        help="PPO update mini-batch size. Defaults to --batch_size.",
    )
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=4,
        help="Number of PPO optimization passes within each generated batch.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument(
        "--max_length",
        type=int,
        default=1300,
        help="Maximum prompt length in tokens; generated tokens are additional.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--model", type=str, default="deepseek-8b_merged")
    parser.add_argument("--target_modules", nargs="*", default=["q_proj", "v_proj"])
    parser.add_argument(
        "--quant", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dataset_path", type=str, default=str(PROJECT_ROOT / "data")
    )
    parser.add_argument(
        "--reward_baseline",
        "--y_bar",
        dest="reward_baseline",
        type=float,
        required=True,
        help=(
            "Post-SFT training MAE (the reference statistic e_bar in the "
            "manuscript reward). The main wrapper derives this value from the "
            "500 training cities' 2004-2017 targets before launching PPO."
        ),
    )
    parser.add_argument(
        "--penalty_lambda",
        type=float,
        default=10.0,
        help="Invalid-output coefficient lambda in the manuscript reward.",
    )
    parser.add_argument(
        "--reward_baseline_report",
        type=str,
        default="",
        help="JSON provenance report used to derive --reward_baseline.",
    )
    parser.add_argument(
        "--valid_min",
        type=float,
        default=-5.0,
        help="Inclusive lower bound for a valid UHI prediction.",
    )
    parser.add_argument(
        "--valid_max",
        type=float,
        default=5.0,
        help="Inclusive upper bound for a valid UHI prediction.",
    )
    parser.add_argument(
        "--max_decimal_places",
        type=int,
        default=3,
        help="Maximum decimal precision retained by the output parser.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Zero reproduces the TRL 0.11.2 main-run default.",
    )
    parser.add_argument(
        "--expected_train_cities",
        type=int,
        default=0,
        help="Fail unless the training manifest has this many cities; zero disables the check.",
    )
    parser.add_argument(
        "--expected_samples_per_city",
        type=int,
        default=0,
        help="Fail unless each city has this many PPO samples; zero disables the check.",
    )
    parser.add_argument(
        "--save_path",
        "--output_dir",
        dest="save_path",
        type=str,
        default="",
        help=(
            "Checkpoint destination. A one-epoch run saves directly to this path; "
            "a multi-epoch run saves epochN subdirectories."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the checkpoint directories targeted by this run.",
    )
    parser.add_argument(
        "--resume_ppo_checkpoint",
        type=str,
        default="",
        help=(
            "Load PPO adapter and value-head weights from this directory. This is "
            "weight continuation, not an exact optimizer-state resume."
        ),
    )
    parser.add_argument(
        "--resume_training_state",
        type=str,
        default="",
        help=(
            "Accelerate state directory paired with --resume_ppo_checkpoint. "
            "Restores optimizer, value head, per-rank RNG, KL controller, and "
            "the completed PPO step count."
        ),
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Epoch number used for the first checkpoint saved by this process.",
    )
    parser.add_argument(
        "--input_format",
        choices=["prompt", "prompt_task_only", "numeric_plain"],
        default="prompt_task_only",
        help=(
            "Use prompt_task_only to end immediately after the prediction task; "
            "use prompt for the explicit Prediction/[Result:] cue; use "
            "numeric_plain for the unlabeled numeric-list ablation input."
        ),
    )
    parser.add_argument(
        "--log_with",
        choices=["tensorboard", "none"],
        default="tensorboard",
        help="TRL metric logger.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="",
        help="TensorBoard directory; defaults to results_PPO/<model-name>.",
    )
    parser.add_argument(
        "--rollout_worker_devices",
        type=str,
        default="",
        help=(
            "Comma-separated local CUDA device indices used only for parallel "
            "response generation. PPO optimization remains on local device 0."
        ),
    )
    parser.add_argument(
        "--rollout_worker_timeout",
        type=int,
        default=600,
        help="Seconds to wait for rollout-worker startup or generation.",
    )
    parser.add_argument(
        "--max_ppo_steps_per_epoch",
        type=int,
        default=0,
        help="Limit batches per outer epoch for smoke tests; zero uses all batches.",
    )
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help=(
            "Save a weights-only PPO checkpoint every N batches within each "
            "outer epoch; zero disables intermediate checkpoints."
        ),
    )
    parser.add_argument(
        "--efficiency_record",
        type=str,
        default="",
        help="Optional JSON output for the complete PPO optimization stage.",
    )
    return parser


def _validate_args(parser, args):
    if args.mini_batch_size is None:
        args.mini_batch_size = args.batch_size
    positive_integer_fields = {
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        "mini_batch_size": args.mini_batch_size,
        "ppo_epochs": args.ppo_epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
    }
    for name, value in positive_integer_fields.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    backward_batch_size = (
        args.mini_batch_size * args.gradient_accumulation_steps
    )
    if args.batch_size % backward_batch_size != 0:
        parser.error(
            "--batch_size must be divisible by "
            "--mini_batch_size * --gradient_accumulation_steps"
        )
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning_rate must be a positive finite number")
    if not np.isfinite(args.reward_baseline) or args.reward_baseline < 0:
        parser.error("--reward_baseline must be a non-negative finite number")
    if not np.isfinite(args.penalty_lambda) or args.penalty_lambda < 0:
        parser.error("--penalty_lambda must be a non-negative finite number")
    if not np.isfinite(args.valid_min) or not np.isfinite(args.valid_max):
        parser.error("valid-range bounds must be finite")
    if args.valid_min >= args.valid_max:
        parser.error("--valid_min must be smaller than --valid_max")
    if args.max_decimal_places < 0:
        parser.error("--max_decimal_places must be non-negative")
    if args.expected_train_cities < 0 or args.expected_samples_per_city < 0:
        parser.error("expected dataset sizes must be non-negative")
    if args.start_epoch < 0:
        parser.error("--start_epoch must be non-negative")
    if args.rollout_worker_timeout <= 0:
        parser.error("--rollout_worker_timeout must be positive")
    if args.max_ppo_steps_per_epoch < 0:
        parser.error("--max_ppo_steps_per_epoch must be non-negative")
    if args.save_every_steps < 0:
        parser.error("--save_every_steps must be non-negative")
    if not args.resume_ppo_checkpoint and args.start_epoch != 0:
        parser.error("--start_epoch requires --resume_ppo_checkpoint")
    if args.resume_training_state and not args.resume_ppo_checkpoint:
        parser.error("--resume_training_state requires --resume_ppo_checkpoint")
    if args.quant:
        parser.error("quantized PPO is not implemented or validated in this release")


def main():
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    if args.device == "cpu" and torch.cuda.is_available():
        parser.error(
            "CPU PPO must be selected before Python initializes Accelerate; "
            "launch with CUDA_VISIBLE_DEVICES='' and --device cpu"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but no CUDA device is visible")
    set_seed(args.seed)
    launch_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launch_rank = int(os.environ.get("RANK", "0"))
    if launch_world_size < 1:
        parser.error("WORLD_SIZE must be a positive integer")
    rollout_worker_devices = []
    if args.rollout_worker_devices:
        try:
            rollout_worker_devices = [
                int(value.strip())
                for value in args.rollout_worker_devices.split(",")
                if value.strip()
            ]
        except ValueError:
            parser.error("--rollout_worker_devices must contain CUDA device indices")
        if not rollout_worker_devices or min(rollout_worker_devices) < 1:
            parser.error("rollout workers must use local CUDA devices 1 or higher")
        if len(set(rollout_worker_devices)) != len(rollout_worker_devices):
            parser.error("rollout worker devices must be unique")
        if args.device != "cuda":
            parser.error("parallel rollout requires --device cuda")
        if launch_world_size != 1:
            parser.error("parallel rollout cannot be combined with DDP/Accelerate")
        if args.batch_size < len(rollout_worker_devices) + 1:
            parser.error(
                "batch size must provide at least one query for the updater GPU "
                "and each rollout worker"
            )
        visible_cuda_devices = torch.cuda.device_count()
        if max(rollout_worker_devices) >= visible_cuda_devices:
            parser.error(
                "rollout worker device index exceeds the visible CUDA devices"
            )

    dataset_dir = _resolve_project_path(args.dataset_path).resolve()
    model_dir = _resolve_model_dir(args.model)
    model_label = model_dir.name
    reward_baseline_report_path = None
    if args.reward_baseline_report:
        reward_baseline_report_path = _resolve_project_path(
            args.reward_baseline_report
        ).resolve()
        _validate_reward_baseline_report(
            reward_baseline_report_path, args.reward_baseline
        )
    resume_dir = None
    resume_state_dir = None
    resume_metadata = None
    if args.resume_ppo_checkpoint:
        resume_dir = _resolve_project_path(args.resume_ppo_checkpoint).resolve()
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"resume PPO checkpoint not found: {resume_dir}")
        if args.resume_training_state:
            resume_state_dir, resume_metadata = _load_training_resume_metadata(
                _resolve_project_path(args.resume_training_state),
                resume_dir,
                launch_world_size,
            )
            logger.warning(
                "Resuming PPO weights and training state from %s at epoch %s step %s",
                resume_dir,
                resume_metadata["epoch"],
                resume_metadata["ppo_steps_completed_in_epoch"],
            )
            if resume_metadata["epoch"] != args.start_epoch:
                raise ValueError(
                    "--start_epoch does not match the training state: "
                    f"{args.start_epoch} != {resume_metadata['epoch']}"
                )
            _validate_resume_run_config(resume_dir, args, launch_world_size)
        else:
            logger.warning(
                "Continuing PPO model/value-head weights from %s; optimizer, "
                "KL-controller, and RNG states are reinitialized.",
                resume_dir,
            )

    save_root = (
        _resolve_project_path(args.save_path).resolve()
        if args.save_path
        else (PROJECT_ROOT / "model_adapters" / "PPO" / model_label).resolve()
    )
    checkpoint_dirs = _checkpoint_directories(
        save_root, args.train_epochs, args.start_epoch
    )
    if launch_rank == 0:
        _prepare_checkpoint_directories(
            checkpoint_dirs, overwrite=args.overwrite, resume_dir=resume_dir
        )

    global_batch_size = args.batch_size * launch_world_size
    global_mini_batch_size = args.mini_batch_size * launch_world_size
    if launch_rank == 0:
        print(
            "PPO config: "
            f"outer_epochs={args.train_epochs}, processes={launch_world_size}, "
            f"per_process_batch_size={args.batch_size}, "
            f"global_batch_size={global_batch_size}, "
            f"per_process_mini_batch_size={args.mini_batch_size}, "
            f"global_mini_batch_size={global_mini_batch_size}, "
            f"ppo_epochs={args.ppo_epochs}, learning_rate={args.learning_rate}, "
            f"seed={args.seed}"
        )
        print(
            "Reward config: "
            f"y_bar={args.reward_baseline}, lambda={args.penalty_lambda}, "
            f"valid_range=[{args.valid_min}, {args.valid_max}], "
            f"max_decimal_places={args.max_decimal_places}"
        )
        print(
            f"Input format: {args.input_format}; model={model_dir}; "
            f"dataset={dataset_dir}"
        )
        if rollout_worker_devices:
            print(
                "Parallel rollout: updater=local cuda:0, "
                f"workers={rollout_worker_devices}; PPO global mini-batch remains "
                f"{global_mini_batch_size}."
            )

    train_dataset, city_ids, metrics_path = _load_training_dataset(
        dataset_dir,
        expected_train_cities=args.expected_train_cities,
        expected_samples_per_city=args.expected_samples_per_city,
    )
    if len(train_dataset) < global_batch_size:
        raise ValueError(
            f"training dataset has {len(train_dataset)} rows, fewer than batch size "
            f"{global_batch_size}"
        )
    if len(train_dataset) % global_batch_size:
        logger.warning(
            "TRL drop_last=True will omit %d sample(s) per outer epoch.",
            len(train_dataset) % global_batch_size,
        )
    print(
        f"Train dataset init done: cities={len(city_ids)}, "
        f"samples={len(train_dataset)}."
    )

    format_ppo = {
        "prompt": format_prompt_PPO,
        "prompt_task_only": format_task_only_prompt_PPO,
        "numeric_plain": format_numeric_plain_prompt_PPO,
    }[args.input_format]
    train_dataset = train_dataset.map(
        format_ppo,
        desc="Formatting PPO prompts",
        keep_in_memory=True,
    )
    print("Train dataset format done.")

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        fan_in_fan_out=False,
        target_modules=args.target_modules,
    )

    model_dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        str(model_dir),
        torch_dtype=model_dtype,
        peft_config=peft_config,
        trust_remote_code=True,
        local_files_only=True,
    )
    if resume_dir:
        loaded_adapter_tensors = _load_local_ppo_weights(model, resume_dir)
        print(
            f"Loaded {loaded_adapter_tensors} PPO adapter tensors and value head "
            f"from {resume_dir}"
        )

    if model_label == "qwen-7b_merged":
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            pad_token="<|endoftext|>",
            local_files_only=True,
        )
        tokenizer.eos_token = tokenizer.pad_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer has no EOS token to use for PPO padding")
        tokenizer.pad_token = tokenizer.eos_token

    model.pretrained_model.config.pad_token_id = tokenizer.pad_token_id
    context_window = getattr(
        model.pretrained_model.config, "max_position_embeddings", None
    )
    maximum_total_tokens = args.max_length + args.max_new_tokens
    if context_window is not None and maximum_total_tokens > context_window:
        raise ValueError(
            "PPO token budget exceeds the model context window: "
            f"max_length={args.max_length} + max_new_tokens={args.max_new_tokens} "
            f"> max_position_embeddings={context_window}"
        )
    print(
        "PPO token budget: "
        f"prompt<={args.max_length}, generation<={args.max_new_tokens}, "
        f"total<={maximum_total_tokens}, model_context={context_window or 'unknown'}"
    )

    def tokenize(sample):
        encoded = tokenizer(
            sample["prompt"],
            padding=False,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        sample["input_ids"] = encoded["input_ids"].squeeze(0)
        sample["attention_mask"] = encoded["attention_mask"].squeeze(0)
        return sample

    train_dataset = train_dataset.map(
        tokenize,
        batched=False,
        desc="Tokenizing PPO prompts",
        keep_in_memory=True,
    )
    print("Train dataset tokenized done.")

    log_with = None if args.log_with == "none" else args.log_with
    logging_dir = (
        _resolve_project_path(args.logging_dir).resolve()
        if args.logging_dir
        else (PROJECT_ROOT / "results_PPO" / model_label).resolve()
    )
    ppo_config = PPOConfig(
        seed=args.seed,
        model_name=model_label,
        query_dataset=str(metrics_path),
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        remove_unused_columns=False,
        is_peft_model=True,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        log_with=log_with,
        project_kwargs={"logging_dir": str(logging_dir)},
    )

    def collate_fn(batch):
        return {
            "input_ids": [torch.as_tensor(item["input_ids"]) for item in batch],
            "attention_mask": [
                torch.as_tensor(item["attention_mask"]) for item in batch
            ],
            "ground_truth": [item["ground_truth"] for item in batch],
        }

    ppo_trainer = FiniteSafePPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collate_fn,
    )
    accelerator = ppo_trainer.accelerator
    accelerator.wait_for_everyone()

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": args.max_new_tokens,
    }
    runtime_versions = {
        "python": os.sys.version.split()[0],
        "torch": _package_version("torch"),
        "cuda_runtime": torch.version.cuda,
        "transformers": _package_version("transformers"),
        "trl": _package_version("trl"),
        "peft": _package_version("peft"),
        "datasets": _package_version("datasets"),
        "accelerate": _package_version("accelerate"),
    }
    run_config = {
        "schema_version": 1,
        "model": args.model,
        "resolved_model_dir": str(model_dir),
        "resume_ppo_checkpoint": str(resume_dir) if resume_dir else None,
        "resume_training_state": (
            str(resume_state_dir) if resume_state_dir else None
        ),
        "resume_semantics": (
            "optimizer_value_head_rng_kl_and_step"
            if resume_state_dir
            else "weights_only" if resume_dir else None
        ),
        "dataset_dir": str(dataset_dir),
        "training_city_manifest": str(metrics_path),
        "training_cities": len(city_ids),
        "training_samples": len(train_dataset),
        "outer_train_epochs": args.train_epochs,
        "start_epoch": args.start_epoch,
        "batch_size": global_batch_size,
        "mini_batch_size": global_mini_batch_size,
        "per_process_batch_size": args.batch_size,
        "per_process_mini_batch_size": args.mini_batch_size,
        "distributed_processes": accelerator.num_processes,
        "ppo_epochs_per_batch": args.ppo_epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_every_steps": args.save_every_steps,
        "learning_rate": args.learning_rate,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
        "precision": "float32" if args.device == "cpu" else "bfloat16",
        "max_prompt_tokens": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "max_total_tokens": maximum_total_tokens,
        "model_context_window": context_window,
        "generation": {
            "do_sample": True,
            "top_k": 0.0,
            "top_p": 1.0,
        },
        "numerical_safety": {
            "synchronize_non_finite_checks": True,
            "skip_non_finite_loss_updates": True,
            "skip_non_finite_gradient_updates": True,
        },
        "rollout_execution": {
            "mode": (
                "parallel_generation_single_updater"
                if rollout_worker_devices
                else "local_generation"
            ),
            "updater_device": 0 if rollout_worker_devices else None,
            "worker_devices": rollout_worker_devices,
            "optimizer_semantics": "single_process",
        },
        "reward": {
            "formula": "e_bar - abs(y - y_hat) - lambda * e_bar * I_invalid",
            "reward_symbol": "e_bar",
            "y_bar": args.reward_baseline,
            "baseline_statistic": (
                "post-fusion SFT training MAE"
                if reward_baseline_report_path
                else "fixed user-supplied value"
            ),
            "baseline_report": (
                str(reward_baseline_report_path)
                if reward_baseline_report_path
                else None
            ),
            "lambda": args.penalty_lambda,
            "valid_min": args.valid_min,
            "valid_max": args.valid_max,
            "max_decimal_places": args.max_decimal_places,
        },
        "input_format": args.input_format,
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": args.target_modules,
        },
        "runtime_versions": runtime_versions,
        "checkpoint_directories": [str(path) for path in checkpoint_dirs],
    }

    resume_epoch_iterator = None
    resume_completed_steps = 0
    if resume_state_dir:
        resume_completed_steps = resume_metadata["ppo_steps_completed_in_epoch"]
        if resume_completed_steps >= len(ppo_trainer.dataloader):
            raise ValueError(
                "resume step must be smaller than the number of batches in the "
                f"epoch: {resume_completed_steps} >= {len(ppo_trainer.dataloader)}"
            )
        resume_epoch_iterator = iter(ppo_trainer.dataloader)
        for _ in tqdm(
            range(resume_completed_steps),
            desc="Restoring PPO dataloader position",
            disable=not accelerator.is_main_process,
        ):
            next(resume_epoch_iterator)
        accelerator.load_state(str(resume_state_dir), strict=False)
        ppo_trainer.kl_ctl.value = float(
            resume_metadata["kl_controller"]["value"]
        )
        optimizer_tensors = [
            value
            for optimizer_state in ppo_trainer.optimizer.state.values()
            for value in optimizer_state.values()
            if torch.is_tensor(value)
        ]
        if not _all_ranks_tensors_finite(
            list(ppo_trainer.model_params) + optimizer_tensors
        ):
            raise FloatingPointError("resumed PPO model or optimizer is non-finite")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(
                "Restored PPO training state: "
                f"epoch={resume_metadata['epoch']}, "
                f"completed_steps={resume_completed_steps}, "
                f"kl_coef={ppo_trainer.kl_ctl.value}"
            )

    rollout_pool = None
    if rollout_worker_devices:
        print("Starting parallel rollout workers")
        rollout_pool = ParallelRolloutPool(
            policy_model=ppo_trainer.model,
            model_dir=model_dir,
            resume_dir=resume_dir,
            target_modules=args.target_modules,
            tokenizer_dir=model_dir,
            worker_devices=rollout_worker_devices,
            seed=args.seed,
            timeout=args.rollout_worker_timeout,
        )
        print("Parallel rollout workers ready")

    is_main_process = accelerator.is_main_process
    if is_main_process:
        print("Start training")
    synchronize_for_timing = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
    synchronize_for_timing()
    ppo_training_started = time.perf_counter()
    try:
        for epoch_offset in tqdm(
            range(args.train_epochs),
            desc="epoch",
            disable=not accelerator.is_main_process,
        ):
            epoch = args.start_epoch + epoch_offset
            total_prediction_error = 0.0
            total_reward = 0.0
            valid_predictions = 0
            invalid_outputs = 0
            rounded_outputs = 0
            invalid_reasons = Counter()
            resumed_epoch = epoch_offset == 0 and resume_epoch_iterator is not None
            step_count = resume_completed_steps if resumed_epoch else 0
            loss_skips_at_epoch_start = ppo_trainer.non_finite_loss_skips
            gradient_skips_at_epoch_start = (
                ppo_trainer.non_finite_gradient_skips
            )
            total_update_seconds = 0.0
            generation_start = (
                rollout_pool.total_generation_seconds if rollout_pool else 0.0
            )
            sync_start = rollout_pool.total_sync_seconds if rollout_pool else 0.0
            epoch_batches = (
                resume_epoch_iterator
                if resumed_epoch
                else iter(ppo_trainer.dataloader)
            )
            for batch in tqdm(
                epoch_batches,
                desc=f"PPO epoch{epoch}",
                total=len(ppo_trainer.dataloader),
                initial=step_count,
                disable=not accelerator.is_main_process,
            ):
                if (
                    args.max_ppo_steps_per_epoch
                    and step_count >= args.max_ppo_steps_per_epoch
                ):
                    break
                query_tensors = batch["input_ids"]
                if rollout_pool:
                    generated_tensors = rollout_pool.generate(
                        query_tensors,
                        generation_kwargs,
                        local_generate=lambda local_queries: ppo_trainer.generate(
                            query_tensor=local_queries, **generation_kwargs
                        ),
                    )
                else:
                    generated_tensors = ppo_trainer.generate(
                        query_tensor=query_tensors, **generation_kwargs
                    )
                response_tensors = [
                    generated[query.size(0) :].reshape(-1)
                    for query, generated in zip(query_tensors, generated_tensors)
                ]
                batch["response"] = [
                    tokenizer.decode(response, skip_special_tokens=True)
                    for response in response_tensors
                ]

                rewards = []
                for index, text in enumerate(batch["response"]):
                    parsed = parse_prediction(
                        text,
                        valid_min=args.valid_min,
                        valid_max=args.valid_max,
                        max_decimal_places=args.max_decimal_places,
                    )
                    breakdown = compute_ppo_reward(
                        float(batch["ground_truth"][index]),
                        parsed,
                        y_bar=args.reward_baseline,
                        penalty_lambda=args.penalty_lambda,
                    )
                    if parsed.is_valid:
                        valid_predictions += 1
                        total_prediction_error += breakdown.prediction_error
                    else:
                        invalid_outputs += 1
                        invalid_reasons[parsed.invalid_reason or "unknown"] += 1
                        logger.info(
                            "Invalid output (%s): %r",
                            parsed.invalid_reason,
                            text[:160],
                        )
                    if parsed.was_rounded:
                        rounded_outputs += 1
                    total_reward += breakdown.reward
                    rewards.append(
                        torch.tensor(breakdown.reward, dtype=torch.float32)
                    )

                update_started = time.perf_counter()
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                total_update_seconds += time.perf_counter() - update_started
                if rollout_pool:
                    rollout_pool.publish_policy(ppo_trainer.model)
                if args.log_with != "none":
                    ppo_trainer.log_stats(stats, batch, rewards)
                step_count += 1
                if args.save_every_steps and step_count % args.save_every_steps == 0:
                    intermediate_dir = (
                        checkpoint_dirs[epoch_offset]
                        / f"checkpoint-step-{step_count:06d}"
                    )
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        _save_ppo_checkpoint_locally(
                            ppo_trainer, intermediate_dir
                        )
                        _write_json(
                            intermediate_dir / "ppo_checkpoint_state.json",
                            {
                                "epoch": epoch,
                                "ppo_steps_completed_in_epoch": step_count,
                                "resume_semantics": "weights_only",
                            },
                        )
                        _write_json(
                            intermediate_dir / "ppo_run_config.json", run_config
                        )
                        print(
                            "Saved intermediate PPO checkpoint to "
                            f"{intermediate_dir}"
                        )
                    accelerator.wait_for_everyone()
                    _save_latest_training_state(
                        accelerator,
                        save_root / "training_state_latest",
                        {
                            "schema_version": 1,
                            "adapter_checkpoint": str(intermediate_dir),
                            "epoch": epoch,
                            "ppo_steps_completed_in_epoch": step_count,
                            "kl_controller": {
                                "class": type(ppo_trainer.kl_ctl).__name__,
                                "value": float(ppo_trainer.kl_ctl.value),
                            },
                        },
                    )

            totals = torch.tensor(
                [
                    total_prediction_error,
                    total_reward,
                    valid_predictions,
                    invalid_outputs,
                    rounded_outputs,
                    step_count,
                ],
                dtype=torch.float64,
                device=accelerator.device,
            )
            totals = accelerator.reduce(totals, reduction="sum").cpu().tolist()
            (
                total_prediction_error,
                total_reward,
                valid_predictions,
                invalid_outputs,
                rounded_outputs,
                summed_step_count,
            ) = totals
            valid_predictions = int(valid_predictions)
            invalid_outputs = int(invalid_outputs)
            rounded_outputs = int(rounded_outputs)
            step_count = int(round(summed_step_count / accelerator.num_processes))

            gathered_reason_counters = gather_object([dict(invalid_reasons)])
            invalid_reasons = Counter()
            for reason_counter in gathered_reason_counters:
                invalid_reasons.update(reason_counter)

            response_count = valid_predictions + invalid_outputs
            mean_valid_error = (
                total_prediction_error / valid_predictions
                if valid_predictions
                else None
            )
            mean_reward = total_reward / response_count if response_count else None
            printable_error = (
                f"{mean_valid_error:.6f}"
                if mean_valid_error is not None
                else "n/a"
            )
            save_dir = checkpoint_dirs[epoch_offset]
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print(
                    f"Epoch {epoch} PPO steps={step_count}, "
                    f"valid_predictions={valid_predictions}, "
                    f"invalid_outputs={invalid_outputs}, "
                    f"mean_valid_abs_error={printable_error}"
                )
                save_dir.parent.mkdir(parents=True, exist_ok=True)
                _save_ppo_checkpoint_locally(ppo_trainer, save_dir)
                epoch_summary = {
                    "epoch": epoch,
                    "ppo_steps": step_count,
                    "responses": response_count,
                    "valid_predictions": valid_predictions,
                    "invalid_outputs": invalid_outputs,
                    "invalid_output_rate": (
                        invalid_outputs / response_count if response_count else None
                    ),
                    "invalid_reasons": dict(sorted(invalid_reasons.items())),
                    "rounded_outputs": rounded_outputs,
                    "skipped_non_finite_loss_updates": (
                        ppo_trainer.non_finite_loss_skips
                        - loss_skips_at_epoch_start
                    ),
                    "skipped_non_finite_gradient_updates": (
                        ppo_trainer.non_finite_gradient_skips
                        - gradient_skips_at_epoch_start
                    ),
                    "mean_valid_absolute_error": mean_valid_error,
                    "mean_reward": mean_reward,
                    "timing": {
                        "ppo_update_seconds": total_update_seconds,
                        "parallel_generation_seconds": (
                            rollout_pool.total_generation_seconds - generation_start
                            if rollout_pool
                            else None
                        ),
                        "adapter_sync_seconds": (
                            rollout_pool.total_sync_seconds - sync_start
                            if rollout_pool
                            else None
                        ),
                    },
                }
                _write_json(save_dir / "ppo_run_config.json", run_config)
                _write_json(save_dir / "ppo_epoch_summary.json", epoch_summary)
                print(f"Saved PPO adapter to {save_dir}")
            accelerator.wait_for_everyone()
            _save_latest_training_state(
                accelerator,
                save_root / "training_state_latest",
                {
                    "schema_version": 1,
                    "adapter_checkpoint": str(save_dir),
                    "epoch": epoch,
                    "ppo_steps_completed_in_epoch": step_count,
                    "epoch_complete": True,
                    "kl_controller": {
                        "class": type(ppo_trainer.kl_ctl).__name__,
                        "value": float(ppo_trainer.kl_ctl.value),
                    },
                },
            )
    finally:
        if rollout_pool:
            rollout_pool.close()
        synchronize_for_timing()
        ppo_training_seconds = time.perf_counter() - ppo_training_started
        accelerator.end_training()
    if is_main_process and args.efficiency_record:
        write_measurement(
            args.efficiency_record,
            kind="training",
            method="UHI-E3LLM",
            stage="PPO",
            value=ppo_training_seconds,
            unit="seconds",
            note="two-epoch PPO generation/update loop and checkpoint writes; excludes reward-baseline inference and model loading",
            metadata={
                "epochs": args.train_epochs,
                "global_batch_size": global_batch_size,
                "training_samples": len(train_dataset),
            },
        )


if __name__ == "__main__":
    main()
