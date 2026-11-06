from __future__ import annotations

from dataclasses import dataclass

from output_parsing import ParsedOutput


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    prediction_error: float
    invalid_penalty: float
    is_invalid: bool


def compute_ppo_reward(
    ground_truth: float,
    parsed: ParsedOutput,
    *,
    y_bar: float,
    penalty_lambda: float,
) -> RewardBreakdown:

    if y_bar < 0:
        raise ValueError("y_bar must be non-negative")
    if penalty_lambda < 0:
        raise ValueError("penalty_lambda must be non-negative")

    if not parsed.is_valid:
        prediction_error = 0.0
    else:
        assert parsed.parsed_value is not None
        prediction_error = abs(float(ground_truth) - parsed.parsed_value)
    is_invalid = not parsed.is_valid
    invalid_penalty = penalty_lambda * y_bar if is_invalid else 0.0
    reward = y_bar - prediction_error - invalid_penalty
    return RewardBreakdown(
        reward=reward,
        prediction_error=prediction_error,
        invalid_penalty=invalid_penalty,
        is_invalid=is_invalid,
    )
