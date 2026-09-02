"""Shared numerical-output parsing for PPO training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re


NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)")


@dataclass(frozen=True)
class ParsedOutput:
    raw_text: str
    matched_text: str | None
    parsed_value: float | None
    is_numeric: bool
    numbers_found: int
    decimal_places: int | None
    was_rounded: bool
    in_valid_range: bool
    is_valid: bool
    invalid_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _decimal_places(token: str) -> int:
    unsigned = token.lstrip("+-")
    if "." not in unsigned:
        return 0
    return len(unsigned.rsplit(".", 1)[1])


def parse_prediction(
    text: str,
    *,
    valid_min: float = -5.0,
    valid_max: float = 5.0,
    max_decimal_places: int = 3,
) -> ParsedOutput:
    """Parse one generated UHI value using the manuscript validity rules.

    The prediction is the first numeric token of the generated sequence; any
    trailing text the model continues to write is ignored. A response is
    valid when that value is finite and lies in the inclusive physical range.
    Values exceeding the reporting precision are rounded before evaluation
    and marked through ``was_rounded``. ``numbers_found`` records how many
    numeric tokens the response contained so verbose generations stay
    auditable without being discarded.
    """

    if valid_min >= valid_max:
        raise ValueError("valid_min must be smaller than valid_max")
    if max_decimal_places < 0:
        raise ValueError("max_decimal_places must be non-negative")

    raw_text = "" if text is None else str(text)
    matches = NUMBER_PATTERN.findall(raw_text)
    if not matches:
        return ParsedOutput(
            raw_text=raw_text,
            matched_text=None,
            parsed_value=None,
            is_numeric=False,
            numbers_found=0,
            decimal_places=None,
            was_rounded=False,
            in_valid_range=False,
            is_valid=False,
            invalid_reason="missing_numeric_value",
        )

    token = matches[0]
    try:
        value = float(token)
    except ValueError:
        return ParsedOutput(
            raw_text=raw_text,
            matched_text=token,
            parsed_value=None,
            is_numeric=False,
            numbers_found=len(matches),
            decimal_places=None,
            was_rounded=False,
            in_valid_range=False,
            is_valid=False,
            invalid_reason="malformed_numeric_value",
        )

    places = _decimal_places(token)
    finite = math.isfinite(value)
    in_range = finite and valid_min <= value <= valid_max
    was_rounded = finite and places > max_decimal_places
    evaluated_value = round(value, max_decimal_places) if was_rounded else value
    if not finite:
        reason = "non_finite_value"
    elif not in_range:
        reason = "out_of_range"
    else:
        reason = None

    return ParsedOutput(
        raw_text=raw_text,
        matched_text=token,
        parsed_value=evaluated_value,
        is_numeric=True,
        numbers_found=len(matches),
        decimal_places=places,
        was_rounded=was_rounded,
        in_valid_range=in_range,
        is_valid=reason is None,
        invalid_reason=reason,
    )
