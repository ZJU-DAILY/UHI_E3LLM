"""LoRA distillation with loss applied only to the result and explanation."""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from statistics import mean

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from efficiency import synchronize_cuda, write_measurement


RESULT_VALUE_RE = re.compile(
    r"\[Result:\]\s*Predicted UHI intensity for the next month:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
)
NUMERIC_LIST_RESULT_RE = re.compile(
    r"\]\s*\n([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\n\s*\n"
    r"(?=Geographical Context:)"
)
EXPLANATION_MARKER = "Geographical Context:"


def _percentile(values, quantile):
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return ordered[index]


def find_supervised_spans(text):
    result_match = RESULT_VALUE_RE.search(text)
    if result_match is None:
        result_match = NUMERIC_LIST_RESULT_RE.search(text)
    if result_match is None:
        raise ValueError("Missing structured or numeric-list result.")
    explanation_start = text.find(EXPLANATION_MARKER, result_match.end())
    if explanation_start < 0:
        raise ValueError(f"Missing explanation marker {EXPLANATION_MARKER!r}.")
    return result_match.span(1), explanation_start


def selected_explanation_files(data_path, selection_manifest, expected_samples):
    data_path = Path(data_path)
    manifest_path = None
    if selection_manifest is not None:
        manifest_path = Path(selection_manifest)
    else:
        default_manifest = data_path.parent / "selection_manifest.json"
        if default_manifest.is_file():
            manifest_path = default_manifest

    if manifest_path is None:
        direct_files = list(data_path.glob("sample_*.txt"))
        indexed = {}
        for path in direct_files:
            match = re.fullmatch(r"sample_(\d+)\.txt", path.name)
            if match is None:
                continue
            index = int(match.group(1))
            if index in indexed:
                raise ValueError(f"Duplicate flat explanation index: {index}")
            indexed[index] = path
        expected_indices = set(range(1, expected_samples + 1))
        if set(indexed) != expected_indices:
            missing = sorted(expected_indices.difference(indexed))
            unexpected = sorted(set(indexed).difference(expected_indices))
            raise ValueError(
                "Flat explanation samples must be consecutively numbered from 1 "
                f"through {expected_samples}: missing={missing[:5]}, "
                f"unexpected={unexpected[:5]}."
            )
        nested_files = {
            path.resolve()
            for path in data_path.rglob("sample_*.txt")
            if path.parent != data_path
        }
        if nested_files:
            raise ValueError(
                "Flat explanation mode cannot be mixed with nested sample files."
            )
        return [indexed[index] for index in range(1, expected_samples + 1)]

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Explanation selection manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_split") != "train":
        raise ValueError(
            f"Explanation samples must come from the training split: {manifest_path}"
        )
    records = manifest.get("samples")
    if not isinstance(records, list) or len(records) != expected_samples:
        observed = len(records) if isinstance(records, list) else "invalid"
        raise ValueError(
            f"Expected {expected_samples} selected explanation records, got {observed}."
        )
    ranks = [record.get("selection_rank") for record in records]
    if sorted(ranks) != list(range(1, expected_samples + 1)):
        raise ValueError("Explanation selection ranks must be unique and contiguous.")

    selected = []
    for record in sorted(records, key=lambda item: item["selection_rank"]):
        relative = Path(record.get("content_file", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe explanation content path in manifest: {relative}")
        if not relative.parts or relative.parts[0] != "content":
            raise ValueError(
                f"Manifest content paths must start with 'content/': {relative}"
            )
        selected.append(data_path.joinpath(*relative.parts[1:]))
    if len(set(selected)) != len(selected):
        raise ValueError("Explanation selection manifest contains duplicate content files.")

    expected_set = {path.resolve() for path in selected}
    actual_set = {path.resolve() for path in data_path.rglob("sample_*.txt")}
    if actual_set != expected_set:
        missing = sorted(str(path) for path in expected_set - actual_set)
        extra = sorted(str(path) for path in actual_set - expected_set)
        raise ValueError(
            "Explanation files do not match selection_manifest.json: "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    return selected


class ExplanationDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length,
        result_loss_weight,
        explanation_loss_weight,
        expected_samples,
        selection_manifest=None,
        allow_truncation=False,
        supervision_scope="weighted_response",
        append_eos_token=False,
    ):
        if supervision_scope not in {"weighted_response", "full_text"}:
            raise ValueError(f"Unsupported supervision scope: {supervision_scope}")
        self.supervision_scope = supervision_scope
        self.append_eos_token = append_eos_token
        self.samples = []
        self.file_paths = []
        self.stats = {
            "tokens": [],
            "result_tokens": [],
            "explanation_tokens": [],
            "supervised_tokens": [],
            "truncated": 0,
        }
        self.file_paths = selected_explanation_files(
            data_path,
            selection_manifest,
            expected_samples,
        )

        for file_path in self.file_paths:
            text = file_path.read_text(encoding="utf-8").strip()
            result_span, explanation_start = find_supervised_spans(text)
            encoded, labels, weights, truncated, span_counts = self._encode(
                tokenizer,
                text,
                max_length,
                result_span,
                explanation_start,
                result_loss_weight,
                explanation_loss_weight,
                supervision_scope,
                append_eos_token,
            )
            self.stats["truncated"] += int(truncated)
            self.stats["tokens"].append(int(encoded["attention_mask"].sum().item()))
            self.stats["result_tokens"].append(span_counts["result"])
            self.stats["explanation_tokens"].append(span_counts["explanation"])
            self.stats["supervised_tokens"].append(span_counts["supervised"])
            self.samples.append(
                {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": labels.squeeze(0),
                    "loss_weights": weights.squeeze(0),
                }
            )

        self._print_stats(max_length, result_loss_weight, explanation_loss_weight)
        if self.stats["truncated"] and not allow_truncation:
            raise ValueError(
                f"{self.stats['truncated']} explanation samples exceed max_length="
                f"{max_length}; increase DISTILL_MAX_LENGTH."
            )

    @staticmethod
    def _encode(
        tokenizer,
        text,
        max_length,
        result_span,
        explanation_start,
        result_loss_weight,
        explanation_loss_weight,
        supervision_scope,
        append_eos_token=False,
    ):
        encoded = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").squeeze(0)
        labels = encoded["input_ids"].clone()
        weights = torch.zeros_like(labels, dtype=torch.float32)
        attention_mask = encoded["attention_mask"].squeeze(0)
        if append_eos_token:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no EOS token to append.")
            eos_index = int(attention_mask.sum().item())
            if eos_index >= max_length:
                raise ValueError(
                    "Cannot append EOS because the tokenized sample fills max_length."
                )
            encoded["input_ids"][0, eos_index] = tokenizer.eos_token_id
            encoded["attention_mask"][0, eos_index] = 1
            attention_mask[eos_index] = 1
            offsets[eos_index] = torch.tensor(
                [len(text), len(text)], dtype=offsets.dtype
            )
            labels[0, eos_index] = tokenizer.eos_token_id
        result_tokens = 0
        explanation_tokens = 0

        for token_index, (start, end) in enumerate(offsets.tolist()):
            is_padding = attention_mask[token_index].item() == 0
            has_no_span = start == 0 and end == 0
            in_result = end > result_span[0] and start < result_span[1]
            in_explanation = end > explanation_start
            result_tokens += int(not is_padding and not has_no_span and in_result)
            explanation_tokens += int(
                not is_padding and not has_no_span and in_explanation
            )
            if is_padding or has_no_span:
                labels[0, token_index] = -100
            elif supervision_scope == "full_text":
                weights[0, token_index] = 1.0
            elif in_result:
                weights[0, token_index] = result_loss_weight
            elif in_explanation:
                weights[0, token_index] = explanation_loss_weight
            else:
                labels[0, token_index] = -100

        non_padding_offsets = offsets[attention_mask.bool()]
        last_character = (
            int(non_padding_offsets[-1][1].item()) if len(non_padding_offsets) else 0
        )
        truncated = last_character < len(text)
        if not torch.any(weights > 0):
            raise ValueError("No supervised tokens remain after tokenization.")
        supervised_tokens = int(((labels != -100) & (weights > 0)).sum().item())
        return encoded, labels, weights, truncated, {
            "result": result_tokens,
            "explanation": explanation_tokens,
            "supervised": supervised_tokens,
        }

    def _print_stats(self, max_length, result_weight, explanation_weight):
        tokens = self.stats["tokens"]
        print(f"Supervision scope: {self.supervision_scope}")
        print(f"Append supervised EOS token: {self.append_eos_token}")
        print(f"Loaded {len(self.samples)} explanation-distillation samples")
        print(
            f"Loss weights: result={result_weight}, explanation={explanation_weight}"
        )
        print(
            f"Tokens at max_length={max_length}: min={min(tokens)}, "
            f"p50={_percentile(tokens, 0.50)}, p90={_percentile(tokens, 0.90)}, "
            f"max={max(tokens)}, mean={mean(tokens):.1f}, "
            f"truncated={self.stats['truncated']}"
        )
        print(
            "Supervised tokens: "
            f"result_mean={mean(self.stats['result_tokens']):.1f}, "
            f"explanation_mean={mean(self.stats['explanation_tokens']):.1f}, "
            f"total_mean={mean(self.stats['supervised_tokens']):.1f}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def weighted_causal_lm_loss(logits, labels, loss_weights):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    shifted_weights = loss_weights[..., 1:].contiguous()
    valid = shifted_labels != -100
    token_losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shifted_labels)
    denominator = (shifted_weights * valid).sum().clamp_min(1e-12)
    return (token_losses * shifted_weights * valid).sum() / denominator


def _torch_dtype(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--input_format",
        choices=("prompt", "numeric_plain"),
        default="prompt",
        help="Student input serialization recorded with the distilled adapter.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--max_length", type=int, default=6000)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Optional smoke-test limit; 0 trains every sample.",
    )
    parser.add_argument(
        "--save_at_steps",
        type=int,
        nargs="*",
        default=(),
        help="Save adapter-only checkpoints after these completed optimizer steps.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--result_loss_weight", type=float, default=1.0)
    parser.add_argument("--explanation_loss_weight", type=float, default=0.001)
    parser.add_argument(
        "--supervision_scope",
        choices=("weighted_response", "full_text"),
        default="weighted_response",
        help=(
            "weighted_response supervises only the numerical result and explanation; "
            "full_text supervises every non-padding token in each released sample."
        ),
    )
    parser.add_argument(
        "--append_eos_token",
        action="store_true",
        help="Append and supervise one EOS token after each complete sample.",
    )
    parser.add_argument("--expected_samples", type=int, default=900)
    parser.add_argument(
        "--selection_manifest",
        default=None,
        help="Defaults to selection_manifest.json next to the content directory.",
    )
    parser.add_argument("--allow_truncation", action="store_true")
    parser.add_argument("--efficiency_record", type=str, default="")
    parser.add_argument("--efficiency_member_epoch", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    for name in (
        "batch_size",
        "max_length",
        "num_epochs",
        "log_interval",
        "expected_samples",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive.")
    if args.max_steps < 0:
        raise ValueError("--max_steps must be non-negative.")
    if any(step <= 0 for step in args.save_at_steps):
        raise ValueError("--save_at_steps values must be positive.")
    if len(set(args.save_at_steps)) != len(args.save_at_steps):
        raise ValueError("--save_at_steps values must be unique.")
    if args.result_loss_weight <= 0 or args.explanation_loss_weight <= 0:
        raise ValueError("Both loss weights must be positive.")
    if args.efficiency_member_epoch is not None and args.efficiency_member_epoch < 0:
        raise ValueError("--efficiency_member_epoch must be non-negative.")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Explanation adapter output is not empty: {output_dir}. "
            "Choose a new EXPLANATION_ADAPTER name."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = ExplanationDataset(
        args.data_path,
        tokenizer,
        args.max_length,
        args.result_loss_weight,
        args.explanation_loss_weight,
        args.expected_samples,
        args.selection_manifest,
        args.allow_truncation,
        args.supervision_scope,
        args.append_eos_token,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=_torch_dtype(args.torch_dtype),
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config).to(args.device)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    global_step = 0
    training_seconds = 0.0
    save_at_steps = set(args.save_at_steps)
    stop_training = False
    for epoch in range(args.num_epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        for batch in data_loader:
            input_ids = batch["input_ids"].to(args.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)
            loss_weights = batch["loss_weights"].to(args.device, non_blocking=True)
            if args.efficiency_record:
                synchronize_cuda()
                step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = weighted_causal_lm_loss(outputs.logits, labels, loss_weights)
            loss.backward()
            optimizer.step()
            if args.efficiency_record:
                synchronize_cuda()
                training_seconds += time.perf_counter() - step_started
            if global_step % args.log_interval == 0:
                print(f"Step {global_step} | Loss: {loss.item():.6f}")
            global_step += 1
            if global_step in save_at_steps:
                checkpoint_dir = output_dir / f"checkpoint-step{global_step:06d}"
                if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                    raise FileExistsError(
                        f"Intermediate checkpoint is not empty: {checkpoint_dir}"
                    )
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(
                    f"Saved intermediate explanation adapter at step {global_step} "
                    f"to {checkpoint_dir}"
                )
            if args.max_steps and global_step >= args.max_steps:
                stop_training = True
                break
        if stop_training:
            print(f"Stopped after max_steps={args.max_steps} (smoke-test limit).")
            break

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    run_config = vars(args).copy()
    run_config["model_path"] = str(Path(args.model_path).resolve())
    run_config["data_path"] = str(Path(args.data_path).resolve())
    run_config["output_dir"] = str(output_dir.resolve())
    run_config["training_samples"] = len(dataset)
    run_config["global_steps"] = global_step
    if args.efficiency_record:
        run_config["training_seconds"] = training_seconds
    (output_dir / "distillation_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.efficiency_record:
        write_measurement(
            args.efficiency_record,
            kind="training",
            method="UHI-E3LLM",
            stage=(
                "Distillation"
                if args.efficiency_member_epoch is None
                else f"Distillation_epoch{args.efficiency_member_epoch}"
            ),
            value=training_seconds,
            unit="seconds",
            note="student forward/backward/optimizer execution only; excludes teacher generation, model/data loading, tokenization, and checkpoint writes",
            metadata={
                "policy_epoch": args.efficiency_member_epoch,
                "epochs": args.num_epochs,
                "global_steps": global_step,
                "training_samples": len(dataset),
                "batch_size": args.batch_size,
                "model_path": run_config["model_path"],
                "adapter_dir": run_config["output_dir"],
                "timing_protocol": "summed_training_steps",
            },
        )
    print(f"Saved explanation adapter to {output_dir}")


if __name__ == "__main__":
    main()
