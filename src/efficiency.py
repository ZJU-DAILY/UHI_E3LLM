"""Shared timing helpers for reproducible efficiency measurements."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def detected_hardware(device: str | None = None) -> str:
    requested = (device or "").lower()
    if torch.cuda.is_available() and requested != "cpu":
        index = torch.cuda.current_device()
        return f"{torch.cuda.get_device_name(index)} (single GPU)"

    try:
        completed = subprocess.run(
            ["lscpu", "-J"],
            check=True,
            capture_output=True,
            text=True,
        )
        fields = {
            row["field"].rstrip(":"): row["data"]
            for row in json.loads(completed.stdout)["lscpu"]
        }
        model = fields.get("Model name")
        if model:
            return model
    except (FileNotFoundError, KeyError, json.JSONDecodeError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def write_measurement(
    path: str | os.PathLike[str],
    *,
    kind: str,
    method: str,
    stage: str,
    value: float,
    unit: str,
    note: str,
    hardware: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"efficiency value must be finite and positive; got {value}")
    if kind not in {"training", "inference"}:
        raise ValueError(f"unsupported efficiency kind: {kind}")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "kind": kind,
        "method": method,
        "stage": stage,
        "value": value,
        "unit": unit,
        "hardware": hardware or detected_hardware(),
        "note": note,
        "metadata": metadata or {},
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Efficiency measurement written to {output}")


def timed_call(function: Callable[[], Any]) -> tuple[Any, float]:
    synchronize_cuda()
    started = time.perf_counter()
    result = function()
    synchronize_cuda()
    return result, time.perf_counter() - started


def benchmark_model_call(
    function: Callable[[], Any],
    *,
    sample_count: int,
    warmup_calls: int = 5,
    measured_calls: int = 5,
) -> float:
    """Return the mean of independently timed model calls in ms per sample."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if warmup_calls < 0 or measured_calls <= 0:
        raise ValueError("warmup_calls must be non-negative and measured_calls positive")

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        with torch.inference_mode():
            for _ in range(warmup_calls):
                function()
            synchronize_cuda()
            elapsed_calls = []
            for _ in range(measured_calls):
                synchronize_cuda()
                started = time.perf_counter()
                function()
                synchronize_cuda()
                elapsed_calls.append(time.perf_counter() - started)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    return (sum(elapsed_calls) / measured_calls) * 1000.0 / sample_count
