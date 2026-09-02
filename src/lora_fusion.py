"""LoRA update-matrix similarity and greedy adapter-selection utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from safetensors.torch import load_file


@dataclass(frozen=True)
class AdapterDeltaState:
    modules: dict[str, tuple[torch.Tensor, torch.Tensor]]
    scale: float


@dataclass(frozen=True)
class GreedySelection:
    selected: list[str]
    weights: list[float]
    validation_mae: float
    steps: list[dict[str, object]]


def _adapter_weight_path(adapter_path: Path) -> Path:
    safetensors_path = adapter_path / "adapter_model.safetensors"
    if safetensors_path.is_file():
        return safetensors_path
    raise FileNotFoundError(
        f"Missing adapter_model.safetensors under {adapter_path}"
    )


def load_adapter_delta_state(adapter_path: Path) -> AdapterDeltaState:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("use_dora", False):
        raise ValueError("DoRA adapters are not supported by delta-matrix fusion.")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError(
            "Per-module rank/alpha patterns are not supported by this fusion path."
        )

    rank = float(config["r"])
    alpha = float(config["lora_alpha"])
    scale = alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)
    weights = load_file(str(_adapter_weight_path(adapter_path)), device="cpu")

    modules: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    suffix = ".lora_A.weight"
    for key in sorted(weights):
        if not key.endswith(suffix):
            continue
        module_name = key[: -len(suffix)]
        b_key = f"{module_name}.lora_B.weight"
        if b_key not in weights:
            raise ValueError(f"Missing matching LoRA B matrix for {key}")
        modules[module_name] = (weights[key].float(), weights[b_key].float())
    if not modules:
        raise ValueError(f"No standard LoRA A/B matrices found in {adapter_path}")
    return AdapterDeltaState(modules=modules, scale=scale)


def delta_cosine_similarity(
    state_a: AdapterDeltaState, state_b: AdapterDeltaState
) -> tuple[float, int]:
    modules_a = set(state_a.modules)
    modules_b = set(state_b.modules)
    if modules_a != modules_b:
        missing_a = sorted(modules_b - modules_a)
        missing_b = sorted(modules_a - modules_b)
        raise ValueError(
            "Adapter target modules differ; "
            f"missing from first={missing_a}, missing from second={missing_b}"
        )

    inner = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for module_name in sorted(modules_a):
        a1, b1 = state_a.modules[module_name]
        a2, b2 = state_b.modules[module_name]
        if b1.shape[0] != b2.shape[0] or a1.shape[1] != a2.shape[1]:
            raise ValueError(f"LoRA module shape mismatch for {module_name}")

        inner += (
            float(torch.trace((b1.T @ b2) @ (a2 @ a1.T)))
            * state_a.scale
            * state_b.scale
        )
        norm_a += (
            float(torch.trace((b1.T @ b1) @ (a1 @ a1.T)))
            * state_a.scale**2
        )
        norm_b += (
            float(torch.trace((b2.T @ b2) @ (a2 @ a2.T)))
            * state_b.scale**2
        )
    if norm_a <= 0 or norm_b <= 0:
        raise ValueError("Cannot compute cosine similarity for a zero LoRA update.")
    similarity = inner / math.sqrt(norm_a * norm_b)
    return max(-1.0, min(1.0, similarity)), len(modules_a)


def pairwise_delta_cosine(
    adapters: Mapping[str, Path],
) -> tuple[dict[tuple[str, str], float], list[dict[str, object]]]:
    states = {
        name: load_adapter_delta_state(path) for name, path in adapters.items()
    }
    similarities: dict[tuple[str, str], float] = {}
    rows: list[dict[str, object]] = []
    for name_a, name_b in combinations(adapters, 2):
        similarity, module_count = delta_cosine_similarity(
            states[name_a], states[name_b]
        )
        key = tuple(sorted((name_a, name_b)))
        similarities[key] = similarity
        rows.append(
            {
                "adapter_a": name_a,
                "adapter_b": name_b,
                "common_modules": module_count,
                "cosine_similarity": similarity,
            }
        )
    return similarities, rows


def exponential_mae_weights(maes: Sequence[float], beta: float) -> list[float]:
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if not maes or any(not math.isfinite(value) for value in maes):
        raise ValueError(f"All MAE values must be finite: {maes}")
    minimum = min(maes)
    unnormalized = [math.exp(-beta * (value - minimum)) for value in maes]
    total = sum(unnormalized)
    return [value / total for value in unnormalized]


def _similarity(
    similarities: Mapping[tuple[str, str], float], name_a: str, name_b: str
) -> float:
    if name_a == name_b:
        return 1.0
    return similarities[tuple(sorted((name_a, name_b)))]


def greedy_select_adapters(
    individual_mae: Mapping[str, float],
    similarities: Mapping[tuple[str, str], float],
    similarity_threshold: float,
    beta: float,
    evaluate_fusion: Callable[[list[str], list[float]], float],
    min_mae_improvement: float = 0.0,
) -> GreedySelection:
    if not individual_mae:
        raise ValueError("No candidate adapters were provided.")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [-1, 1]")
    if min_mae_improvement < 0:
        raise ValueError("min_mae_improvement must be non-negative")

    ordered = sorted(individual_mae, key=lambda name: (individual_mae[name], name))
    best = ordered[0]
    if not math.isfinite(individual_mae[best]):
        raise ValueError("No candidate has a finite validation MAE.")

    selected = [best]
    current_mae = float(individual_mae[best])
    steps: list[dict[str, object]] = [
        {
            "step": 0,
            "candidate": best,
            "selected_before": "",
            "trial_adapters": best,
            "weights": "1",
            "minimum_cosine_similarity": 1.0,
            "trial_mae": current_mae,
            "mae_before": current_mae,
            "accepted": True,
            "reason": "best_individual_validation_mae",
        }
    ]

    for step_index, candidate in enumerate(ordered[1:], start=1):
        selected_before = list(selected)
        if not math.isfinite(individual_mae[candidate]):
            steps.append(
                {
                    "step": step_index,
                    "candidate": candidate,
                    "selected_before": ",".join(selected_before),
                    "trial_adapters": ",".join(selected + [candidate]),
                    "weights": "",
                    "minimum_cosine_similarity": math.nan,
                    "trial_mae": math.nan,
                    "mae_before": current_mae,
                    "accepted": False,
                    "reason": "non_finite_individual_validation_mae",
                }
            )
            continue
        candidate_similarities = [
            _similarity(similarities, candidate, retained) for retained in selected
        ]
        minimum_similarity = min(candidate_similarities)
        trial_adapters = selected + [candidate]
        trial_weights = exponential_mae_weights(
            [individual_mae[name] for name in trial_adapters], beta
        )

        if minimum_similarity < similarity_threshold:
            steps.append(
                {
                    "step": step_index,
                    "candidate": candidate,
                    "selected_before": ",".join(selected_before),
                    "trial_adapters": ",".join(trial_adapters),
                    "weights": ",".join(f"{value:.12g}" for value in trial_weights),
                    "minimum_cosine_similarity": minimum_similarity,
                    "trial_mae": math.nan,
                    "mae_before": current_mae,
                    "accepted": False,
                    "reason": "cosine_similarity_below_threshold",
                }
            )
            continue

        trial_mae = float(evaluate_fusion(trial_adapters, trial_weights))
        accepted = (
            math.isfinite(trial_mae)
            and trial_mae < current_mae - min_mae_improvement
        )
        steps.append(
            {
                "step": step_index,
                "candidate": candidate,
                "selected_before": ",".join(selected_before),
                "trial_adapters": ",".join(trial_adapters),
                "weights": ",".join(f"{value:.12g}" for value in trial_weights),
                "minimum_cosine_similarity": minimum_similarity,
                "trial_mae": trial_mae,
                "mae_before": current_mae,
                "accepted": accepted,
                "reason": "validation_mae_improved"
                if accepted
                else "validation_mae_not_improved",
            }
        )
        if accepted:
            selected = trial_adapters
            current_mae = trial_mae

    final_weights = exponential_mae_weights(
        [individual_mae[name] for name in selected], beta
    )
    return GreedySelection(
        selected=selected,
        weights=final_weights,
        validation_mae=current_mae,
        steps=steps,
    )
