import argparse
import os
from typing import List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def _sanitize_adapter_name(name: str) -> str:
    cleaned = name.replace(".", "_")
    if cleaned != name:
        print(f"Adapter name '{name}' contains '.', sanitized to '{cleaned}'")
    return cleaned


def _find_adapters(adapter_root: str) -> Tuple[List[str], List[str]]:
    adapter_names: List[str] = []
    adapter_paths: List[str] = []

    root_config = os.path.join(adapter_root, "adapter_config.json")
    if os.path.exists(root_config):
        adapter_names.append(_sanitize_adapter_name(os.path.basename(os.path.normpath(adapter_root))))
        adapter_paths.append(adapter_root)

    for entry in sorted(os.listdir(adapter_root)):
        candidate = os.path.join(adapter_root, entry)
        if not os.path.isdir(candidate):
            continue
        if os.path.exists(os.path.join(candidate, "adapter_config.json")):
            adapter_names.append(_sanitize_adapter_name(entry))
            adapter_paths.append(candidate)

    if not adapter_paths:
        raise ValueError(f"No adapters with adapter_config.json found under {adapter_root}")

    return adapter_names, adapter_paths

## Default to equal weights if none provided
def _resolve_weights(cli_weights: List[float], count: int) -> List[float]:
    if cli_weights:
        if len(cli_weights) != count:
            raise ValueError(f"Expected {count} weights, got {len(cli_weights)}")
        return cli_weights
    return [1.0] * count


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple PEFT LoRA checkpoints into a single weighted adapter."
    )
    parser.add_argument("--model", type=str, default="deepseek-8b")
    parser.add_argument("--adapter_root", type=str, default="model_adapters/SFT")
    parser.add_argument("--output_dir", type=str, default="model_adapters/merged")
    parser.add_argument("--merged_name", type=str, default="merge")
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--combination_type", type=str, default="dare_ties", choices=["linear", "dare_ties"])
    parser.add_argument("--density", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    adapter_names, adapter_paths = _find_adapters(f'{args.adapter_root}/{args.model}')
    weights = _resolve_weights(args.weights, len(adapter_names))

    print(f"Loading base model from model/{args.model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        f'model/{args.model}',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)

    print(f"Loading base adapter: {adapter_names[0]} ({adapter_paths[0]})")
    model = PeftModel.from_pretrained(base_model, adapter_paths[0], adapter_name=adapter_names[0])

    for path, name in zip(adapter_paths[1:], adapter_names[1:]):
        print(f"Loading additional adapter: {name} ({path})")
        model.load_adapter(path, adapter_name=name)

    print(f"Combining {len(adapter_names)} adapters with weights {weights}")
    model.add_weighted_adapter(
        adapter_names,
        weights,
        args.merged_name,
        combination_type=args.combination_type,
        density=args.density,
    )
    model.set_adapter(args.merged_name)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.model)
    print(f"Saving merged adapter to {output_path}")
    model.save_pretrained(output_path, selected_adapters=[args.merged_name])
    print("Merge complete.")


if __name__ == "__main__":
    main()
