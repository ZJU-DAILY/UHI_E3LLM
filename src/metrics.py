"""Shared regression metrics and target-period helpers."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def scalar(value) -> int | float:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return value


def next_month(year: int, month: int) -> tuple[int, int]:
    if month < 1 or month > 12:
        raise ValueError(f"invalid month: {month}")
    if month == 12:
        return year + 1, 1
    return year, month + 1


def previous_month(year: int, month: int) -> tuple[int, int]:
    if month < 1 or month > 12:
        raise ValueError(f"invalid month: {month}")
    if month == 1:
        return year - 1, 12
    return year, month - 1


def target_month_from_sample(sample) -> tuple[int, int]:
    return (
        int(scalar(sample["year_info"])),
        int(scalar(sample["month_info"])),
    )


def target_in_period(
    sample,
    start: tuple[int, int] = (2018, 1),
    end: tuple[int, int] = (2020, 12),
) -> bool:
    target = target_month_from_sample(sample)
    return start <= target <= end


def target_in_train_period(sample) -> bool:
    return target_in_period(sample, start=(2004, 1), end=(2017, 12))


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and have equal shape")
    error = pred - true
    mse = float(np.mean(np.square(error)))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": math.sqrt(mse),
    }
