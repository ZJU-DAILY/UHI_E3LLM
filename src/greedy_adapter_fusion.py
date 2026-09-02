"""Greedy validation-based fusion of epoch-level LoRA adapters."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import peft
import torch
import transformers
import trl
from datasets import concatenate_datasets, load_from_disk
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from lora_fusion import (
    exponential_mae_weights,
    greedy_select_adapters,
    pairwise_delta_cosine,
)
from metrics import regression_metrics, target_in_period, target_month_from_sample
from output_parsing import parse_prediction
from utils import create_numeric_plain_prompt, create_prompt, sort_dataset_by_year_month


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VALIDATION_TARGETS = [
    (year, month) for year in range(2018, 2021) for month in range(1, 13)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-8b")
    parser.add_argument("--adapter_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--dataset_path", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output_adapter_dir",
        type=Path,
        default=PROJECT_ROOT / "model_adapters" / "merged" / "deepseek-8b",
    )
    parser.add_argument("--merged_model_dir", type=Path, default=None)
    parser.add_argument(
        "--report_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "lora_fusion",
    )
    parser.add_argument("--merged_name", default="manuscript_fusion")
    parser.add_argument(
        "--selection_mode",
        choices=["greedy", "fixed"],
        default="greedy",
        help="Use validation-based greedy selection or force all candidates.",
    )
    parser.add_argument(
        "--fixed_weights",
        nargs="+",
        type=float,
        default=None,
        help="Ordered weights for fixed mode; they must be non-negative and sum to 1.",
    )
    parser.add_argument("--similarity_threshold", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--min_mae_improvement", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_length", type=int, default=1300)
    parser.add_argument("--max_new_tokens", type=int, default=19)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_min", type=float, default=-5.0)
    parser.add_argument("--valid_max", type=float, default=5.0)
    parser.add_argument("--max_decimal_places", type=int, default=3)
    parser.add_argument(
        "--input_format",
        choices=["prompt", "numeric_plain"],
        default="prompt",
        help="Use the same serialization as the SFT adapters being fused.",
    )
    parser.add_argument("--expected_validation_cities", type=int, default=500)
    parser.add_argument("--expected_samples_per_city", type=int, default=36)
    parser.add_argument(
        "--validation_city_limit",
        type=int,
        default=0,
        help="Use only the first N training cities for a smoke test; 0 uses all.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.adapter_paths:
        raise ValueError("At least one candidate adapter is required.")
    if not -1.0 <= args.similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [-1, 1].")
    if args.beta < 0:
        raise ValueError("beta must be non-negative.")
    if args.min_mae_improvement < 0:
        raise ValueError("min_mae_improvement must be non-negative.")
    for name in ("batch_size", "max_length", "max_new_tokens"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.num_workers < 0 or args.validation_city_limit < 0:
        raise ValueError("Worker and city-limit values must be non-negative.")
    if args.valid_min >= args.valid_max:
        raise ValueError("valid_min must be smaller than valid_max.")
    if args.expected_validation_cities <= 0:
        raise ValueError("expected_validation_cities must be positive.")
    if args.expected_samples_per_city != len(EXPECTED_VALIDATION_TARGETS):
        raise ValueError(
            "expected_samples_per_city must be 36 for the fixed 2018-2020 "
            "validation period."
        )
    if args.selection_mode == "fixed":
        if args.fixed_weights is None:
            raise ValueError("fixed mode requires --fixed_weights.")
        if len(args.fixed_weights) != len(args.adapter_paths):
            raise ValueError(
                "fixed_weights must contain exactly one weight per adapter."
            )
        if any(not math.isfinite(value) or value < 0 for value in args.fixed_weights):
            raise ValueError("fixed_weights must be finite and non-negative.")
        if not math.isclose(sum(args.fixed_weights), 1.0, abs_tol=1e-8):
            raise ValueError("fixed_weights must sum to 1.")
    elif args.fixed_weights is not None:
        raise ValueError("--fixed_weights is only valid in fixed mode.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_path(model: str) -> Path:
    supplied = Path(model).expanduser()
    if supplied.is_absolute():
        return supplied
    return PROJECT_ROOT / "model" / supplied


def adapter_name(path: Path, index: int) -> str:
    match = re.fullmatch(r"checkpoint-epoch(\d+)", path.name)
    if match:
        return f"epoch{int(match.group(1)) + 1}"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", path.name).strip("_")
    return cleaned or f"adapter{index}"


def resolve_adapters(paths: list[Path]) -> dict[str, Path]:
    adapters: dict[str, Path] = {}
    for index, supplied_path in enumerate(paths):
        path = supplied_path.expanduser().resolve()
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing adapter_config.json: {path}")
        name = adapter_name(path, index)
        if name in adapters:
            raise ValueError(f"Duplicate adapter name after normalization: {name}")
        adapters[name] = path
    return adapters


def load_validation_dataset(args: argparse.Namespace):
    metrics_dir = args.dataset_path / "city_metrics"
    train_metrics = pd.read_csv(metrics_dir / "city_metrics_train.csv")
    test_metrics = pd.read_csv(metrics_dir / "city_metrics_test.csv")
    city_ids = train_metrics["id"].astype(int).tolist()
    if len(city_ids) != len(set(city_ids)):
        raise ValueError("Training city IDs are not unique.")
    overlap = set(city_ids).intersection(test_metrics["id"].astype(int))
    if overlap:
        raise ValueError(f"Training and test city lists overlap: {sorted(overlap)}")
    if args.validation_city_limit > 0:
        city_ids = city_ids[: args.validation_city_limit]
    elif len(city_ids) != args.expected_validation_cities:
        raise ValueError(
            f"Expected {args.expected_validation_cities} validation cities, "
            f"found {len(city_ids)}"
        )

    city_datasets = []
    for city_id in city_ids:
        dataset = load_from_disk(
            str(args.dataset_path / "valid" / f"UHI_city_{city_id}")
        )
        period_indices = [
            index
            for index in range(len(dataset))
            if target_in_period(dataset[index])
        ]
        dataset = dataset.select(period_indices)
        dataset = sort_dataset_by_year_month(dataset)
        if len(dataset) != args.expected_samples_per_city:
            raise ValueError(
                f"City {city_id}: expected {args.expected_samples_per_city} "
                f"validation samples, found {len(dataset)}"
            )
        targets = [target_month_from_sample(row) for row in dataset]
        if targets != EXPECTED_VALIDATION_TARGETS:
            raise ValueError(
                f"City {city_id}: validation targets must cover every month "
                "from 2018-01 through 2020-12 exactly once."
            )
        city_datasets.append(dataset)
    if not city_datasets:
        raise ValueError("The validation city list is empty.")

    combined = concatenate_datasets(city_datasets)
    expected_samples = len(city_ids) * args.expected_samples_per_city
    if len(combined) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} validation samples, found {len(combined)}"
        )
    print(
        f"Validation data: {len(city_ids)} training cities, "
        f"{len(combined)} targets from 2018-01 through 2020-12",
        flush=True,
    )
    return combined, city_ids


def batch_targets(info_batch) -> torch.Tensor:
    values = info_batch["y"]
    while isinstance(values, (list, tuple)) and len(values) == 1:
        values = values[0]
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().reshape(-1).float()
    return torch.as_tensor(values, dtype=torch.float32).reshape(-1)


class ValidationEvaluator:
    def __init__(
        self,
        args: argparse.Namespace,
        model,
        tokenizer,
        validation_dataset,
    ) -> None:
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.validation_dataset = validation_dataset
        self.prompt_builder = (
            create_numeric_plain_prompt
            if args.input_format == "numeric_plain"
            else create_prompt
        )

    def evaluate(self, adapter: str) -> dict[str, object]:
        self.model.set_adapter(adapter)
        self.model.eval()
        dataloader = DataLoader(
            self.validation_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.num_workers,
        )
        y_true: list[float] = []
        y_pred: list[float] = []
        invalid = 0
        print(f"Evaluating adapter {adapter}", flush=True)
        with torch.inference_mode():
            for info_batch in tqdm(
                dataloader,
                desc=f"Validation {adapter}",
                unit="batch",
                dynamic_ncols=True,
            ):
                targets = batch_targets(info_batch)
                prompts = [
                    self.prompt_builder(info_batch, index)
                    for index in range(len(targets))
                ]
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.args.max_length,
                ).to(self.args.device)
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=self.args.max_new_tokens,
                    do_sample=False,
                )
                generated = outputs[:, inputs.input_ids.size(1) :]
                for target, token_ids in zip(targets.tolist(), generated):
                    text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                    parsed = parse_prediction(
                        text,
                        valid_min=self.args.valid_min,
                        valid_max=self.args.valid_max,
                        max_decimal_places=self.args.max_decimal_places,
                    )
                    if not parsed.is_valid or parsed.parsed_value is None:
                        invalid += 1
                        continue
                    y_true.append(float(target))
                    y_pred.append(float(parsed.parsed_value))

        if y_true:
            metrics = regression_metrics(y_true, y_pred)
        else:
            metrics = {"mae": math.inf, "mse": math.inf, "rmse": math.inf}
        expected = len(self.validation_dataset)
        result: dict[str, object] = {
            **metrics,
            "valid_predictions": len(y_true),
            "invalid_predictions": invalid,
            "expected_predictions": expected,
            "selection_mae": metrics["mae"] if invalid == 0 else math.inf,
        }
        print(
            f"Validation result {adapter}: MAE={metrics['mae']:.6f}, "
            f"invalid={invalid}/{expected}",
            flush=True,
        )
        return result


def load_model_and_adapters(
    args: argparse.Namespace, adapters: dict[str, Path]
):
    model_path = resolve_model_path(args.model)
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing base model config: {model_path}")
    print(f"Loading base model: {model_path}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(args.device)
    first_name = next(iter(adapters))
    model = PeftModel.from_pretrained(
        base_model,
        str(adapters[first_name]),
        adapter_name=first_name,
        is_trainable=False,
    )
    for name, path in list(adapters.items())[1:]:
        model.load_adapter(str(path), adapter_name=name, is_trainable=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        padding_side="left",
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, model_path


def add_rank_concatenation_fusion(
    model,
    adapter_names: list[str],
    weights: list[float],
    merged_name: str,
) -> None:
    """Represent the exact weighted delta sum by concatenating LoRA ranks."""

    if not adapter_names or len(adapter_names) != len(weights):
        raise ValueError("Adapter names and weights must be non-empty and aligned.")
    if merged_name in model.peft_config:
        model.delete_adapter(merged_name)
    model.add_weighted_adapter(
        adapter_names,
        weights,
        merged_name,
        combination_type="cat",
    )

    # PEFT initializes a new adapter using the base-layer dtype. Reconstruct the
    # concatenated factors at the source-adapter precision so an fp32 LoRA is not
    # silently rounded when the frozen base model uses bf16.
    with torch.no_grad():
        for module in model.modules():
            lora_a = getattr(module, "lora_A", None)
            lora_b = getattr(module, "lora_B", None)
            scaling = getattr(module, "scaling", None)
            if (
                lora_a is None
                or lora_b is None
                or scaling is None
                or merged_name not in lora_a
            ):
                continue
            available = [
                (name, weight)
                for name, weight in zip(adapter_names, weights)
                if name in lora_a
            ]
            if not available:
                continue
            source_dtype = lora_a[available[0][0]].weight.dtype
            for name, _weight in available[1:]:
                source_dtype = torch.promote_types(
                    source_dtype, lora_a[name].weight.dtype
                )
            target_a = lora_a[merged_name].to(dtype=source_dtype)
            target_b = lora_b[merged_name].to(dtype=source_dtype)
            target_scale = float(scaling[merged_name])
            concatenated_a = torch.cat(
                [
                    lora_a[name].weight.detach().to(dtype=source_dtype)
                    * float(weight)
                    * float(scaling[name])
                    / target_scale
                    for name, weight in available
                ],
                dim=0,
            )
            concatenated_b = torch.cat(
                [
                    lora_b[name].weight.detach().to(dtype=source_dtype)
                    for name, _weight in available
                ],
                dim=1,
            )
            target_a.weight.copy_(concatenated_a)
            target_b.weight.copy_(concatenated_b)
    model.set_adapter(merged_name)


def remove_adapter(model, adapter_name: str, fallback: str) -> None:
    model.set_adapter(fallback)
    if adapter_name in model.peft_config:
        model.delete_adapter(adapter_name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Pass --overwrite to replace it."
            )
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def require_available_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Pass --overwrite to replace it."
        )


def save_named_adapter(model, adapter_name: str, output_dir: Path) -> None:
    staging = Path(tempfile.mkdtemp(prefix="adapter-fusion-", dir=output_dir.parent))
    try:
        model.save_pretrained(
            str(staging),
            selected_adapters=[adapter_name],
            safe_serialization=True,
        )
        nested = staging / adapter_name
        source = nested if (nested / "adapter_config.json").is_file() else staging
        shutil.copytree(source, output_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    adapters = resolve_adapters(args.adapter_paths)
    if args.merged_name in adapters:
        raise ValueError(
            f"Merged adapter name collides with candidate: {args.merged_name}"
        )

    require_available_output(args.report_dir, args.overwrite)
    require_available_output(args.output_adapter_dir, args.overwrite)
    if args.merged_model_dir:
        require_available_output(args.merged_model_dir, args.overwrite)
    prepare_output(args.report_dir, args.overwrite)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    print("Candidate adapters:", flush=True)
    for name, path in adapters.items():
        print(f"  {name}: {path}", flush=True)

    similarities, similarity_rows = pairwise_delta_cosine(adapters)
    for row in similarity_rows:
        row["threshold"] = args.similarity_threshold
        row["eligible"] = row["cosine_similarity"] >= args.similarity_threshold
    pd.DataFrame(
        similarity_rows,
        columns=[
            "adapter_a",
            "adapter_b",
            "common_modules",
            "cosine_similarity",
            "threshold",
            "eligible",
        ],
    ).to_csv(
        args.report_dir / "pairwise_lora_similarity.csv", index=False
    )

    validation_dataset = None
    validation_city_ids: list[int] = []
    if args.selection_mode == "greedy":
        validation_dataset, validation_city_ids = load_validation_dataset(args)
    model, tokenizer, model_path = load_model_and_adapters(args, adapters)

    evaluation_rows: list[dict[str, object]] = []
    individual_results: dict[str, dict[str, object]] = {}
    selection_steps: list[dict[str, object]] = []
    selected_validation_result: dict[str, object] | None = None
    selected_validation_mae: float | None = None

    if args.selection_mode == "greedy":
        assert validation_dataset is not None
        evaluator = ValidationEvaluator(args, model, tokenizer, validation_dataset)
        for name in adapters:
            result = evaluator.evaluate(name)
            individual_results[name] = result
            evaluation_rows.append(
                {
                    "stage": "individual",
                    "adapter": name,
                    "adapters": name,
                    "weights": "1",
                    **result,
                }
            )
        individual_mae = {
            name: float(result["selection_mae"])
            for name, result in individual_results.items()
        }

        trial_index = 0
        fusion_results: dict[tuple[str, ...], dict[str, object]] = {}

        def evaluate_fusion(names: list[str], weights: list[float]) -> float:
            nonlocal trial_index
            trial_index += 1
            trial_name = f"greedy_trial_{trial_index}"
            add_rank_concatenation_fusion(
                model,
                names,
                weights,
                trial_name,
            )
            try:
                result = evaluator.evaluate(trial_name)
                evaluation_rows.append(
                    {
                        "stage": "greedy_trial",
                        "adapter": trial_name,
                        "adapters": ",".join(names),
                        "weights": ",".join(
                            f"{value:.12g}" for value in weights
                        ),
                        **result,
                    }
                )
                fusion_results[tuple(names)] = result
                return float(result["selection_mae"])
            finally:
                remove_adapter(model, trial_name, names[0])

        selection = greedy_select_adapters(
            individual_mae=individual_mae,
            similarities=similarities,
            similarity_threshold=args.similarity_threshold,
            beta=args.beta,
            evaluate_fusion=evaluate_fusion,
            min_mae_improvement=args.min_mae_improvement,
        )
        selected_names = selection.selected
        final_weights = exponential_mae_weights(
            [individual_mae[name] for name in selected_names], args.beta
        )
        selection_steps = selection.steps
        selected_validation_mae = selection.validation_mae
        if len(selected_names) == 1:
            selected_validation_result = individual_results[selected_names[0]]
        else:
            selected_validation_result = fusion_results[tuple(selected_names)]
    else:
        selected_names = list(adapters)
        final_weights = [float(value) for value in args.fixed_weights or []]
        minimum_similarity = min(similarities.values(), default=1.0)
        selection_steps = [
            {
                "step": 0,
                "candidate": ",".join(selected_names),
                "selected_before": "",
                "trial_adapters": ",".join(selected_names),
                "weights": ",".join(
                    f"{value:.12g}" for value in final_weights
                ),
                "minimum_cosine_similarity": minimum_similarity,
                "trial_mae": math.nan,
                "mae_before": math.nan,
                "accepted": True,
                "reason": "fixed_fusion_no_validation_selection",
            }
        ]

    add_rank_concatenation_fusion(
        model,
        selected_names,
        final_weights,
        args.merged_name,
    )

    evaluation_columns = [
        "stage",
        "adapter",
        "adapters",
        "weights",
        "mae",
        "mse",
        "rmse",
        "valid_predictions",
        "invalid_predictions",
        "expected_predictions",
        "selection_mae",
    ]
    pd.DataFrame(evaluation_rows, columns=evaluation_columns).to_csv(
        args.report_dir / "validation_evaluations.csv", index=False
    )
    pd.DataFrame(selection_steps).to_csv(
        args.report_dir / "greedy_selection_steps.csv", index=False
    )
    report = {
        "method": (
            "greedy_validation_mae_with_cosine_screening_and_rank_concatenation"
            if args.selection_mode == "greedy"
            else "fixed_rank_concatenation_adapter_fusion"
        ),
        "selection_mode": args.selection_mode,
        "delta_fusion": (
            "single_adapter_passthrough"
            if len(selected_names) == 1
            else "exact_weighted_sum_via_lora_rank_concatenation"
        ),
        "rank_concatenation": {
            "resulting_rank": "sum of the selected adapter ranks",
            "delta_equivalence": "Delta_W_merged = sum_i w_i * Delta_W_i",
            "implementation": "peft.LoraModel.add_weighted_adapter",
        },
        "weight_formula": (
            "w_i = softmax(-beta * validation_MAE_i)"
            if args.selection_mode == "greedy"
            else "user_supplied_fixed_weights"
        ),
        "software_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
        },
        "base_model": str(model_path),
        "dataset_path": str(args.dataset_path.resolve()),
        "validation_role": (
            "training-city validation, targets from 2018-01 through 2020-12"
            if args.selection_mode == "greedy"
            else None
        ),
        "expected_full_validation_cities": args.expected_validation_cities,
        "validation_city_limit": args.validation_city_limit,
        "validation_cities": len(validation_city_ids),
        "validation_samples": (
            len(validation_dataset) if validation_dataset is not None else 0
        ),
        "test_cities_used_for_selection": False,
        "evaluation_config": (
            {
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "max_new_tokens": args.max_new_tokens,
                "decoding": "greedy",
                "valid_range": [args.valid_min, args.valid_max],
                "max_decimal_places": args.max_decimal_places,
                "seed": args.seed,
                "device": args.device,
                "input_format": args.input_format,
            }
            if args.selection_mode == "greedy"
            else None
        ),
        "similarity_threshold": args.similarity_threshold,
        "similarity_screening_applied": args.selection_mode == "greedy",
        "beta": args.beta,
        "min_mae_improvement": args.min_mae_improvement,
        "pairwise_similarities": similarity_rows,
        "candidates": {
            name: {
                "path": str(path),
                **individual_results.get(name, {}),
            }
            for name, path in adapters.items()
        },
        "selected_adapters": selected_names,
        "selected_weights": final_weights,
        "selected_validation_mae": selected_validation_mae,
        "selected_validation_metrics": selected_validation_result,
        "greedy_steps": (
            selection_steps if args.selection_mode == "greedy" else []
        ),
        "fixed_fusion": (
            {
                "validation_selection_skipped": True,
                "cosine_screening_applied": False,
                "weights": final_weights,
            }
            if args.selection_mode == "fixed"
            else None
        ),
        "output_adapter_dir": str(args.output_adapter_dir.resolve()),
        "merged_model_dir": str(args.merged_model_dir.resolve())
        if args.merged_model_dir
        else None,
    }
    report_path = args.report_dir / "fusion_report.json"
    report_path.write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    prepare_output(args.output_adapter_dir, args.overwrite)
    save_named_adapter(model, args.merged_name, args.output_adapter_dir)
    shutil.copy2(report_path, args.output_adapter_dir / "fusion_report.json")
    print(f"Merged adapter saved to {args.output_adapter_dir}", flush=True)

    if args.merged_model_dir:
        prepare_output(args.merged_model_dir, args.overwrite)
        model.set_adapter(args.merged_name)
        merged_model = model.merge_and_unload(
            safe_merge=True,
            adapter_names=[args.merged_name],
        )
        merged_model.save_pretrained(
            str(args.merged_model_dir), safe_serialization=True, max_shard_size="5GB"
        )
        tokenizer.save_pretrained(str(args.merged_model_dir))
        shutil.copy2(report_path, args.merged_model_dir / "fusion_report.json")
        print(f"Merged full model saved to {args.merged_model_dir}", flush=True)

    print(
        "Selected adapters: "
        + ", ".join(
            f"{name} (weight={weight:.8f})"
            for name, weight in zip(selected_names, final_weights)
        ),
        flush=True,
    )
    if selected_validation_mae is None:
        print("Validation-based selection skipped (fixed mode).", flush=True)
    else:
        print(
            f"Selected validation MAE: {selected_validation_mae:.8f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
