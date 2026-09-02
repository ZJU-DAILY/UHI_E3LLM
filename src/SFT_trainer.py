import argparse
import inspect
import os
import re
import shutil
import time
from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets, disable_caching, load_from_disk
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer
from efficiency import synchronize_cuda, write_measurement
from utils import format_numeric_plain_prompt, format_prompt

os.environ["WANDB_DISABLED"] = "true"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
EPOCH_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-epoch(\d+)$")


class EpochCheckpointCallback(TrainerCallback):
    """Rename Trainer step checkpoints to stable, zero-based epoch names."""

    def on_save(self, args, state, control, **kwargs):
        if state.epoch is None:
            return control

        epoch_index = max(int(round(state.epoch)) - 1, 0)
        step_checkpoint = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        epoch_checkpoint = os.path.join(
            args.output_dir, f"checkpoint-epoch{epoch_index}"
        )
        if not os.path.isdir(step_checkpoint):
            return control

        if os.path.lexists(epoch_checkpoint):
            if os.path.isdir(epoch_checkpoint) and not os.path.islink(
                epoch_checkpoint
            ):
                shutil.rmtree(epoch_checkpoint)
            else:
                os.remove(epoch_checkpoint)
        os.replace(step_checkpoint, epoch_checkpoint)
        print(f"Checkpoint saved to {epoch_checkpoint}")
        return control


def _resolve_resume_checkpoint(value, output_dir):
    if not value:
        return None

    if value == "latest":
        checkpoints = []
        if os.path.isdir(output_dir):
            for name in os.listdir(output_dir):
                match = EPOCH_CHECKPOINT_PATTERN.fullmatch(name)
                if match:
                    checkpoints.append((int(match.group(1)), name))
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint-epochN directory found under {output_dir}"
            )
        return os.path.join(output_dir, max(checkpoints)[1])

    expanded = os.path.abspath(os.path.expanduser(value))
    candidates = [expanded, os.path.join(output_dir, value)]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Resume checkpoint does not exist: {value}")


def _sft_trainer_tokenizer_kwargs(tokenizer):
    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def _is_batched_sft_examples(examples):
    year_info = examples.get("year_info") if isinstance(examples, Mapping) else None
    if not isinstance(year_info, (list, tuple)) or not year_info:
        return False
    return len(year_info) != 1 or isinstance(
        year_info[0], (list, tuple, np.ndarray, torch.Tensor)
    )


def _row_from_batched_sft_examples(examples, index):
    return {key: value[index] for key, value in examples.items()}


def _make_sft_formatting_func(format_one):
    def format_examples(examples):
        if _is_batched_sft_examples(examples):
            return [
                format_one(_row_from_batched_sft_examples(examples, index))
                for index in range(len(examples["year_info"]))
            ]
        return format_one(examples)

    return format_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--max_length", type=int, default=1300)
    parser.add_argument("--model", type=str, default="deepseek-8b")
    parser.add_argument("--target_modules", nargs="*", default=["q_proj", "v_proj"])
    parser.add_argument(
        "--quant", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path, checkpoint-epochN name, or 'latest'.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data"),
    )
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_cities", type=int, default=0)
    parser.add_argument("--expected_samples_per_city", type=int, default=0)
    parser.add_argument(
        "--data_seed",
        type=int,
        default=None,
        help="Optional data-loader seed; defaults to the Trainer seed behavior.",
    )
    parser.add_argument(
        "--force_retokenize",
        action="store_true",
        help="Ignore cached tokenized datasets and rebuild them from the active prompt.",
    )
    parser.add_argument(
        "--ddp_find_unused_parameters", action="store_true", default=False
    )
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument(
        "--efficiency_record",
        type=str,
        default="",
        help="Optional JSON output for the measured SFT optimization stage.",
    )
    parser.add_argument(
        "--input_format",
        choices=["prompt", "numeric_plain"],
        default="prompt",
        help="Use numeric_plain for the unlabeled numeric-list ablation input.",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    if args.force_retokenize:
        disable_caching()
        print("Dataset cache reuse disabled; prompts will be tokenized again.")
    format_one = (
        format_numeric_plain_prompt
        if args.input_format == "numeric_plain"
        else format_prompt
    )
    formatting_func = _make_sft_formatting_func(format_one)
    print(f"Input format: {args.input_format}")

    df_train = pd.read_csv(f"{args.dataset_path}/city_metrics/city_metrics_train.csv")
    city_id_list = df_train["id"].tolist()
    print(f"Dataset size: {len(city_id_list)}")

    if not city_id_list:
        raise ValueError("The training city list is empty.")
    if len(city_id_list) != len(set(city_id_list)):
        raise ValueError("The training city list contains duplicate IDs.")
    if args.expected_cities > 0 and len(city_id_list) != args.expected_cities:
        raise ValueError(
            f"Expected {args.expected_cities} training cities, got {len(city_id_list)}."
        )

    city_datasets = []
    for city_id in tqdm(
        city_id_list, desc="Loading city datasets", total=len(city_id_list)
    ):
        city_dataset = load_from_disk(
            f"{args.dataset_path}/train/UHI_city_{city_id}"
        )
        if (
            args.expected_samples_per_city > 0
            and len(city_dataset) != args.expected_samples_per_city
        ):
            raise ValueError(
                f"City {city_id} has {len(city_dataset)} training samples; "
                f"expected {args.expected_samples_per_city}."
            )
        city_datasets.append(city_dataset)
    train_dataset = concatenate_datasets(city_datasets)
    print(f"Train dataset init done: {len(train_dataset)} samples.")

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        fan_in_fan_out=False,
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )

    if args.quant:
        print("Quant model")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(PROJECT_ROOT, "model", args.model),
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype="bfloat16",
        )
    else:
        print("No quant")
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(PROJECT_ROOT, "model", args.model),
            trust_remote_code=True,
            torch_dtype="bfloat16",
        )

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(PROJECT_ROOT, "model", args.model), trust_remote_code=True
    )

    if args.gradient_checkpointing:
        print("Gradient checkpointing enabled.")
        model.gradient_checkpointing_enable()

    model.tokenizer = tokenizer

    training_args = SFTConfig(
        num_train_epochs=args.train_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_length,
        # keep in sync with inference/merge scripts that expect model_adapters/*
        output_dir=f"model_adapters/SFT/{args.model}"
        if not args.output_dir
        else args.output_dir,
        bf16=True,
        bf16_full_eval=True,
        save_strategy="epoch",
        gradient_accumulation_steps=args.gradient_accumulation,
        dataloader_num_workers=args.num_workers,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        report_to="none",
        seed=args.seed,
        data_seed=args.data_seed,
    )

    trainer = SFTTrainer(
        model=model,
        **_sft_trainer_tokenizer_kwargs(tokenizer),
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        callbacks=[EpochCheckpointCallback()],
    )

    resume_checkpoint = _resolve_resume_checkpoint(
        args.resume_from_checkpoint, training_args.output_dir
    )
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    synchronize_cuda()
    training_started = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()
    synchronize_cuda()
    training_seconds = time.perf_counter() - training_started
    if args.efficiency_record:
        write_measurement(
            args.efficiency_record,
            kind="training",
            method="UHI-E3LLM",
            stage="SFT",
            value=training_seconds,
            unit="seconds",
            note="optimization, in-loop validation, and checkpoint writes; excludes data/model loading",
            metadata={
                "epochs": args.train_epochs,
                "batch_size": args.batch_size,
                "training_samples": len(train_dataset),
            },
        )
    print(f"Model adapter saved to {training_args.output_dir}")
