import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets, load_from_disk
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from metrics import previous_month, target_in_period, target_in_train_period
from output_parsing import parse_prediction
from efficiency import benchmark_model_call, write_measurement
from utils import (
    create_numeric_plain_prompt,
    create_prompt,
    create_result_only_prompt,
    create_task_only_prompt,
    sort_dataset_by_year_month,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def load_local_lora_adapter(model, adapter_dir):
    adapter_config = LoraConfig.from_pretrained(adapter_dir)
    model = get_peft_model(model, adapter_config)
    saved_adapter = load_safetensors(
        os.path.join(adapter_dir, "adapter_model.safetensors"), device="cpu"
    )
    adapter_parameters = {}
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        checkpoint_name = name.replace(".default.", ".")
        adapter_parameters[checkpoint_name] = parameter
    missing = sorted(set(adapter_parameters) - set(saved_adapter))
    unexpected = sorted(set(saved_adapter) - set(adapter_parameters))
    if missing or unexpected:
        raise ValueError(
            "LoRA adapter keys do not match the inference model: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    with torch.no_grad():
        for name, parameter in adapter_parameters.items():
            value = saved_adapter[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"LoRA adapter shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    print(f"Loaded {len(saved_adapter)} local LoRA adapter tensors")
    return model


def benchmark_inference(args, model, tokenizer, test_datasets, prompt_builder):
    prompts = []
    target_count = args.efficiency_batch_size
    for dataset in test_datasets.values():
        loader = DataLoader(dataset, batch_size=min(target_count, len(dataset)))
        for info_batch in loader:
            y = info_batch["y"][0]
            batch_size = y.size(0) if isinstance(y, torch.Tensor) else len(y)
            prompts.extend(prompt_builder(info_batch, index) for index in range(batch_size))
            if len(prompts) >= target_count:
                break
        if len(prompts) >= target_count:
            break
    if len(prompts) < target_count:
        raise RuntimeError(
            f"Efficiency benchmark requested {target_count} samples, but only "
            f"{len(prompts)} were available"
        )
    prompts = prompts[:target_count]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    ).to(args.device)
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
    latency = benchmark_model_call(
        lambda: model.generate(**generation_kwargs),
        sample_count=target_count,
        warmup_calls=args.efficiency_warmup_calls,
        measured_calls=args.efficiency_measured_calls,
    )
    write_measurement(
        args.efficiency_record,
        kind="inference",
        method="UHI-E3LLM",
        stage="inference",
        value=latency,
        unit="ms_per_sample",
        note="model execution only; excludes loading, prompt construction, tokenization, parsing, and file I/O",
        metadata={
            "batch_size": target_count,
            "warmup_calls": args.efficiency_warmup_calls,
            "measured_calls": args.efficiency_measured_calls,
            "timing_protocol": "independent_repetitions_mean",
            "max_new_tokens": args.max_new_tokens,
        },
    )


def inference(args, model, tokenizer, test_datasets):
    model.eval()
    prompt_builder = {
        "prompt": create_prompt,
        "prompt_result_only": create_result_only_prompt,
        "prompt_task_only": create_task_only_prompt,
        "numeric_plain": create_numeric_plain_prompt,
    }[args.input_format]
    if args.efficiency_record:
        benchmark_inference(args, model, tokenizer, test_datasets, prompt_builder)
    os.makedirs(args.save_path, exist_ok=True)
    model_save_dir = os.path.join(args.save_path, args.model)
    adapter_dir = (
        model_save_dir
        if args.adapter_path == ""
        else os.path.join(model_save_dir, args.adapter_path)
    )
    city_results_dir = os.path.join(adapter_dir, "city_results")
    os.makedirs(city_results_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    def process(dataloader, city_idx):
        squared_error_sum = 0.0
        absolute_error_sum = 0.0

        def batch_inference(args, model, info_batch):

            y = info_batch["y"][0]
            if isinstance(y, torch.Tensor):
                batch_size = y.size(0)
            else:
                batch_size = len(y)
            prompts = []

            for i in range(batch_size):
                prompt = prompt_builder(info_batch, i)
                prompts.append(prompt)

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(args.device)

            gen_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "pad_token_id": tokenizer.eos_token_id,
                "max_new_tokens": args.max_new_tokens,
                # Override the base model's generation_config, which may enable
                # sampling by default even when the CLI requests greedy decoding.
                "do_sample": args.temperature > 0,
            }
            if args.temperature > 0:
                gen_kwargs["temperature"] = args.temperature

            # Multi-run per sample: run N times with different seeds, average predictions
            n_runs = args.num_runs
            all_parsed = []  # [run][sample_idx]
            for r in range(n_runs):
                if n_runs > 1 and args.temperature > 0:
                    torch.manual_seed(args.base_seed + r)
                outputs = model.generate(**gen_kwargs)
                generated_tokens = outputs[:, inputs.input_ids.size(1) :]
                run_parsed = []
                for seq in generated_tokens:
                    text = tokenizer.decode(seq, skip_special_tokens=True)
                    parsed = parse_prediction(
                        text,
                        valid_min=args.valid_min,
                        valid_max=args.valid_max,
                        max_decimal_places=args.max_decimal_places,
                    )
                    if not parsed.is_valid:
                        print(
                            f"  [WARN] Invalid output ({parsed.invalid_reason}): "
                            f"'{text[:120]}'"
                        )
                    run_parsed.append(parsed)
                all_parsed.append(run_parsed)

            results = []
            validity = []
            for sample_index in range(len(all_parsed[0])):
                parsed_runs = [all_parsed[r][sample_index] for r in range(n_runs)]
                valid_values = [
                    parsed.parsed_value
                    for parsed in parsed_runs
                    if parsed.is_valid and parsed.parsed_value is not None
                ]
                results.append(
                    float(np.mean(valid_values)) if valid_values else float("nan")
                )
                validity.append(
                    {
                        "valid_runs": len(valid_values),
                        "total_runs": n_runs,
                        "is_valid": len(valid_values) == n_runs,
                        "generated_is_numeric": all(
                            parsed.is_numeric for parsed in parsed_runs
                        ),
                        "generated_in_valid_range": all(
                            parsed.in_valid_range for parsed in parsed_runs
                        ),
                        "generated_decimal_places": ";".join(
                            "" if parsed.decimal_places is None else str(parsed.decimal_places)
                            for parsed in parsed_runs
                        ),
                        "generated_was_rounded": any(
                            parsed.was_rounded for parsed in parsed_runs
                        ),
                        "invalid_reasons": ";".join(
                            sorted(
                                {
                                    parsed.invalid_reason
                                    for parsed in parsed_runs
                                    if parsed.invalid_reason
                                }
                            )
                        ),
                        "raw_outputs": " || ".join(
                            parsed.raw_text for parsed in parsed_runs
                        ),
                    }
                )

            true = list()
            result = list()
            squared_errors = list()
            records = []

            if results is not None:
                for i in range(len(results)):
                    output = results[i]
                    target_year = int(info_batch["year_info"][0][i].item())
                    target_month = int(info_batch["month_info"][0][i].item())
                    history_end_year, history_end_month = previous_month(
                        target_year, target_month
                    )
                    if np.isnan(output):
                        if args.invalid_policy == "error":
                            raise RuntimeError(
                                f"City {city_idx} target {target_year}-{target_month:02d} "
                                f"has no valid generated prediction: "
                                f"{validity[i]['invalid_reasons']}"
                            )
                        records.append(
                            {
                                "history_end_year": history_end_year,
                                "history_end_month": history_end_month,
                                "history_end_year_month": f"{history_end_year}-{history_end_month:02d}",
                                "target_year": target_year,
                                "target_month": target_month,
                                "target_year_month": f"{target_year}-{target_month:02d}",
                                "forecast_horizon_months": 1,
                                "true_uhi": y[i].item(),
                                "predicted_uhi": np.nan,
                                "absolute_error": np.nan,
                                "squared_error": np.nan,
                                **validity[i],
                            }
                        )
                        continue
                    error = output - y[i].item()
                    true.append(y[i].item())
                    result.append(output)
                    squared_errors.append(error**2)
                    records.append(
                        {
                            "history_end_year": history_end_year,
                            "history_end_month": history_end_month,
                            "history_end_year_month": f"{history_end_year}-{history_end_month:02d}",
                            "target_year": target_year,
                            "target_month": target_month,
                            "target_year_month": f"{target_year}-{target_month:02d}",
                            "forecast_horizon_months": 1,
                            "true_uhi": y[i].item(),
                            "predicted_uhi": output,
                            "absolute_error": abs(error),
                            "squared_error": error**2,
                            **validity[i],
                        }
                    )

            return squared_errors, true, result, records

        city_records = []
        total_samples = 0
        for index, info in enumerate(dataloader):
            l, true, output, records = batch_inference(args, model, info)
            squared_error_sum += sum(l)
            for i in range(len(output)):
                absolute_error_sum += np.abs(true[i] - output[i])
            total_samples += len(true)
            if records:
                city_records.extend(records)

        if total_samples > 0:
            mse_loss = squared_error_sum / total_samples
            mae_loss = absolute_error_sum / total_samples
            rmse_loss = math.sqrt(mse_loss)
        else:
            mse_loss = float("nan")
            mae_loss = float("nan")
            rmse_loss = float("nan")
        if city_records:
            city_df = pd.DataFrame(city_records)
            city_df.insert(0, "city_idx", city_idx)
            city_df.sort_values(by=["target_year", "target_month"], inplace=True)
            city_df.reset_index(drop=True, inplace=True)
            city_df.to_csv(
                os.path.join(city_results_dir, f"city_{city_idx}_predictions.csv"),
                index=False,
            )
        print(
            f"City {city_idx} inference finished, MAE: {mae_loss}, "
            f"MSE Loss: {mse_loss}, RMSE Loss: {rmse_loss}"
        )
        return city_idx, mae_loss, mse_loss, rmse_loss, len(city_records)

    data = []

    for city_idx, test_dataset in test_datasets.items():
        print(f"Inference city {city_idx}")
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, drop_last=False
        )
        city_idx, mae_loss, mse_loss, rmse_loss, sample_count = process(
            test_loader, city_idx
        )
        data.append([city_idx, sample_count, mae_loss, mse_loss, rmse_loss])

    df = pd.DataFrame(
        data, columns=["city_idx", "samples", "mae", "mse", "rmse"]
    )
    if not args.city_limit:
        expected_cities = args.expected_cities or len(test_datasets)
        expected_samples = expected_cities * args.expected_samples_per_city
        wrong_city_counts = df.loc[
            df["samples"] != args.expected_samples_per_city,
            ["city_idx", "samples"],
        ]
        if (
            len(df) != expected_cities
            or int(df["samples"].sum()) != expected_samples
            or not wrong_city_counts.empty
        ):
            raise RuntimeError(
                f"Expected {expected_cities} {args.mode} cities and "
                f"{args.expected_samples_per_city} targets per city "
                f"({expected_samples} total); observed {len(df)} cities and "
                f"{int(df['samples'].sum())} samples. Wrong per-city counts: "
                f"{wrong_city_counts.to_dict(orient='records')}"
            )
    df.to_csv(
        os.path.join(
            adapter_dir, f"{args.adapter_path}_{args.checkpoint}_inference.csv"
        ),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_length", type=int, default=1300)
    parser.add_argument("--max_new_tokens", type=int, default=19)
    parser.add_argument("--model", type=str, default="deepseek-8b")
    parser.add_argument("--adapter_path", type=str, default="SFT")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="",
        help="Direct PEFT adapter directory; overrides adapter_path/model/checkpoint lookup.",
    )
    parser.add_argument("--checkpoint", type=str, default=".")
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "outputs", "inference"),
    )
    parser.add_argument(
        "--mode",
        choices=["test", "test_full", "validation", "train"],
        default="test",
        help=(
            "test: 100 held-out cities, 2018-2020 targets from data/valid. "
            "test_full: the same 100 held-out cities, 2004-2020 targets from "
            "data/train and data/valid. "
            "validation: 500 training cities, 2018-2020 targets from data/valid. "
            "train: 500 training cities, 2004-2017 targets from data/train "
            "(used to derive the PPO reward baseline e_bar)."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data"),
    )
    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of inference runs for averaging"
    )
    parser.add_argument(
        "--base_seed", type=int, default=42, help="Base seed for multi-run"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0=greedy, >0 enables do_sample)",
    )
    parser.add_argument(
        "--input_format",
        choices=["prompt", "prompt_result_only", "prompt_task_only", "numeric_plain"],
        default="prompt",
        help=(
            "Use prompt_result_only to end the structured prompt with only "
            "'[Result:]'; use prompt_task_only to end immediately after the task; "
            "use numeric_plain for the unlabeled numeric-list input."
        ),
    )
    parser.add_argument(
        "--city_limit",
        type=int,
        default=0,
        help="Limit the number of test/train cities for quick inference benchmarking. 0 keeps all cities.",
    )
    parser.add_argument(
        "--city_ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Evaluate an explicit subset of eligible cities while keeping the selected "
            "mode's temporal split. Used by the case-study workflow; omitted for "
            "complete manuscript tables."
        ),
    )
    parser.add_argument(
        "--expected_cities",
        type=int,
        default=0,
        help="Require this many evaluated cities; zero uses the selected manifest size.",
    )
    parser.add_argument(
        "--expected_samples_per_city",
        type=int,
        default=36,
        help=(
            "Required number of targets for every evaluated city: "
            "36 for test/validation (2018-2020), 168 for train (2004-2017)."
        ),
    )
    parser.add_argument(
        "--torch_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--device_map_auto",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shard model weights across all visible GPUs for memory-constrained inference.",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=int,
        default=0,
        help="Optional per-GPU model-memory limit in GiB when --device_map_auto is enabled.",
    )
    parser.add_argument("--valid_min", type=float, default=-5.0)
    parser.add_argument("--valid_max", type=float, default=5.0)
    parser.add_argument("--max_decimal_places", type=int, default=3)
    parser.add_argument(
        "--invalid_policy",
        choices=["error", "record"],
        default="error",
        help="Fail on invalid output or retain an explicit invalid record.",
    )
    parser.add_argument("--efficiency_record", type=str, default="")
    parser.add_argument("--efficiency_batch_size", type=int, default=36)
    parser.add_argument("--efficiency_warmup_calls", type=int, default=5)
    parser.add_argument("--efficiency_measured_calls", type=int, default=5)

    args = parser.parse_args()
    if args.expected_cities < 0:
        parser.error("--expected_cities must be non-negative")
    if args.expected_samples_per_city <= 0:
        parser.error("--expected_samples_per_city must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.max_length <= 0:
        parser.error("--max_length must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be positive")
    if args.num_runs <= 0:
        parser.error("--num_runs must be positive")
    if args.efficiency_batch_size <= 0:
        parser.error("--efficiency_batch_size must be positive")
    if args.efficiency_warmup_calls < 0 or args.efficiency_measured_calls <= 0:
        parser.error("efficiency warm-up/measured calls are invalid")
    print(f"Input format: {args.input_format}")

    if args.num_runs > 1 and args.temperature == 0:
        print(
            "Warning: num_runs > 1 but temperature=0. Greedy decoding is deterministic. Use --temperature > 0 for meaningful multi-run averaging."
        )
        args.num_runs = 1

    torch.manual_seed(args.base_seed)
    np.random.seed(args.base_seed)
    random.seed(args.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.base_seed)

    model_dir = os.path.join(PROJECT_ROOT, "model", args.model)
    model_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]
    model_kwargs = {
        "torch_dtype": model_dtype,
        "trust_remote_code": True,
    }
    if args.device_map_auto:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            parser.error("--device_map_auto requires at least two visible CUDA devices")
        model_kwargs["device_map"] = "balanced"
        if args.max_memory_per_gpu > 0:
            model_kwargs["max_memory"] = {
                index: f"{args.max_memory_per_gpu}GiB"
                for index in range(torch.cuda.device_count())
            }
    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    if not args.device_map_auto:
        model = model.to(args.device)
    if args.adapter_path != "":
        print(f"Adapter: model_adapters/{args.adapter_path}")
        print(f"Checkpoint: {args.model}/{args.checkpoint}")
        resolved_adapter_dir = args.adapter_dir or os.path.join(
            PROJECT_ROOT,
            "model_adapters",
            args.adapter_path,
            args.model,
            args.checkpoint,
        )
        model = load_local_lora_adapter(model, resolved_adapter_dir)

    if args.model == "qwen-7b" or args.model == "qwen-7b_merged":
        print("Qwen model")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            pad_token="<|endoftext|>",
            padding_side="left",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.mode in {"test", "test_full"}:
        pd_test = pd.read_csv(f"{args.dataset_path}/city_metrics/city_metrics_test.csv")
    else:
        pd_test = pd.read_csv(
            f"{args.dataset_path}/city_metrics/city_metrics_train.csv"
        )
    manifest_city_ids = pd_test["id"].astype(int).tolist()
    if args.city_ids is None:
        city_id_list = manifest_city_ids
    else:
        all_city_path = f"{args.dataset_path}/city_metrics/city_metrics_all.csv"
        eligible_city_ids = set(pd.read_csv(all_city_path)["id"].astype(int))
        unknown = sorted(set(args.city_ids).difference(eligible_city_ids))
        if unknown:
            raise ValueError(
                f"Requested city IDs are not in the eligible-city manifest: {unknown}"
            )
        if len(set(args.city_ids)) != len(args.city_ids):
            raise ValueError("--city_ids must not contain duplicates")
        city_id_list = args.city_ids
        outside_role = sorted(set(city_id_list).difference(manifest_city_ids))
        if outside_role:
            print(
                f"Explicit case-study cities outside the default {args.mode} role "
                f"manifest: {outside_role}. Temporal filtering still follows mode={args.mode}."
            )
    if args.expected_cities and len(city_id_list) != args.expected_cities:
        raise RuntimeError(
            f"Expected {args.expected_cities} cities in the {args.mode} manifest; "
            f"found {len(city_id_list)}"
        )
    if args.city_limit and args.city_limit > 0:
        city_id_list = city_id_list[: args.city_limit]
        print(f"City limit enabled: running first {len(city_id_list)} cities.")
    if args.mode == "train":
        split_dir = "train"
        period_filter = target_in_train_period
        period_text = "2004-01 through 2017-12"
    elif args.mode == "test_full":
        split_dir = None
        period_filter = lambda sample: target_in_period(
            sample, start=(2004, 1), end=(2020, 12)
        )
        period_text = "2004-01 through 2020-12"
    else:
        split_dir = "valid"
        period_filter = target_in_period
        period_text = "2018-01 through 2020-12"
    test_datasets = {}

    for index, i in enumerate(city_id_list):
        if args.mode == "test_full":
            test_dataset = concatenate_datasets(
                [
                    load_from_disk(f"{args.dataset_path}/train/UHI_city_{i}"),
                    load_from_disk(f"{args.dataset_path}/valid/UHI_city_{i}"),
                ]
            )
        else:
            test_dataset = load_from_disk(
                f"{args.dataset_path}/{split_dir}/UHI_city_{i}"
            )
        # Rebuild temporal filtering from the current target-month fields.
        test_dataset = test_dataset.filter(
            period_filter, load_from_cache_file=False
        )
        test_dataset = sort_dataset_by_year_month(test_dataset)
        if len(test_dataset) != args.expected_samples_per_city:
            raise RuntimeError(
                f"City {i} must provide {args.expected_samples_per_city} targets "
                f"from {period_text}; found {len(test_dataset)}"
            )
        test_datasets[i] = test_dataset

    print(f"{args.mode.capitalize()} dataset init done.")

    inference(args=args, model=model, tokenizer=tokenizer, test_datasets=test_datasets)
