import datetime
import numbers

import numpy as np
import torch
from dateutil.relativedelta import relativedelta

PROMPT = {
    "Instruction_": "Introduction\n\nYou are an advanced Urban Heat Island (UHI) model designed to forecast "
    "the future UHI intensity for the next month ({0}) based on historical UHI data ({1}) and other "
    "UHI-related factors, including structural factors (rarely vary) and liquidity factors (vary monthly).",
    "Historical": "Historical UHI Intensity\n\nCaptures the last 12 months of urban–rural temperature "
    "contrast, reflecting seasonality and trend.",
    "UHI_data": "[UHI Data:] (Historical urban heat island intensity, monthly, °C)\n{} (12 data points)",
    "Structural": "Structural Factors\n\nDescribe the urban morphology and land-use pattern that set the "
    "long-term background for UHI.",
    "Area": "[Area:] {:.2f} km²",
    "Built_height": "[Built Height:] {:.2f} m",
    "Built_surface": "[Built Surface Area:] {:.0f} m²/km²",
    "Compactness": "[City Compactness:] {:.3f}",
    "Urbanization_level": "[Urbanization Level:] {:.2f}",
    "Location": "[Location:] ({0:.3f}, {1:.3f})",
    "Urban_land_use": "[Urban Land Use:] Open Spaces: {0:.2f}%, "
    "Water Surfaces: {1:.2f}%, "
    "Road Surfaces: {2:.2f}%, "
    "Residential: {3:.2f}%, "
    "Non-Residential: {4:.2f}%",
    "Pop": "[Population Density:] {:.0f} people/km²",
    "Liquidity": "Liquidity Factors\n\nReflect short-term factors of UHI, consisting of average observations "
    "of the last 12 months (12 data points).",
    "Cloud": "[Average Cloud Cover:] {}",
    "Solar": "[Solar Radiation:] {} J m⁻² d⁻¹",
    "Precipitation": "[Precipitation Flux:] {} mm/day",
    "Snow": "[Snow Thickness:] {} m",
    "Evi": "[Enhanced Vegetation Index (EVI):] {}",
    "Ndvi": "[Normalized Difference Vegetation Index (NDVI):] {}",
    "PM25": "[PM2.5 Concentration:] {} µg/m³",
    "Temperature": "[Average Temperature:] {} K",
    "Vapour": "[Average Vapour Pressure:] {} hPa",
    "Wind": "[Average Wind Speed:] {} m/s",
    "Task": "Task\n\nBased on the provided historical UHI data and the accompanying structural and "
    "liquidity factors, predict the UHI intensity for the next month.",
    "Result": "\n\nPrediction\n\n[Result:] Predicted UHI intensity for the next month:{}",
}


_STRUCTURED_FIELD_COUNTS = {
    "year_info": 1,
    "month_info": 1,
    "x": 12,
    "area_info": 1,
    "built_height_info": 1,
    "built_surface_info": 1,
    "urbanization_level_info": 1,
    "compactness_info": 1,
    "urban_land_use_info": 5,
    "location_info": 2,
    "pop_info": 1,
    "cloud_info": 12,
    "solar_info": 12,
    "precipitation_info": 12,
    "snow_info": 12,
    "evi_info": 12,
    "ndvi_info": 12,
    "pm25_info": 12,
    "temperature_info": 12,
    "vapour_info": 12,
    "wind_info": 12,
}


def _prompt_time_context(current_year: int, current_month: int):
    """Return the target month and its immediately preceding 12-month range.

    ``year_info/month_info`` identifies the prediction target. Historical
    inputs cover the 12 consecutive months immediately before that target.
    """
    target = datetime.datetime(current_year, current_month, 1)
    history_end = target - relativedelta(months=1)
    history_start = history_end - relativedelta(months=11)
    return (
        target.strftime("%Y-%m"),
        f"{history_start:%Y-%m} to {history_end:%Y-%m}",
    )


def _numeric_to_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.flatten()[0].item()
    if isinstance(value, numbers.Number):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _flatten_numeric(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_flatten_numeric(item))
        return values
    return [_numeric_to_scalar(value)]


def _select_numeric_field(info, field_name, index):
    value = info[field_name]
    if index is None:
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        return value

    value = value[0]
    if isinstance(value, (list, tuple)):
        return [_numeric_to_scalar(item[index]) for item in value]
    return _numeric_to_scalar(value[index])


def _structured_values(info, index=None):
    values = {}
    for field_name, expected_count in _STRUCTURED_FIELD_COUNTS.items():
        field_values = _flatten_numeric(
            _select_numeric_field(info, field_name, index)
        )
        if len(field_values) != expected_count:
            raise ValueError(
                f"structured prompt expected {expected_count} values for "
                f"{field_name}, got {len(field_values)}"
            )
        values[field_name] = [float(value) for value in field_values]
    return values


def _numeric_list_prompt(info, index=None):
    """Serialize the same displayed numbers as the publication prompt.

    This is the controlled input for the w/o UHI Knowledge Enhancer ablation:
    labels, prose, sections, and units are removed, while feature order,
    publication-facing unit conversion, and decimal precision remain fixed.
    """

    values = _structured_values(info, index=index)
    pieces = [
        str(int(round(values["year_info"][0]))),
        str(int(round(values["month_info"][0]))),
        *(f"{value:.3f}" for value in values["x"]),
        f"{values['area_info'][0]:.2f}",
        f"{values['built_height_info'][0]:.2f}",
        f"{values['built_surface_info'][0] * 100.0:.0f}",
        f"{values['urbanization_level_info'][0]:.2f}",
        *(f"{value:.3f}" for value in values["location_info"]),
        f"{values['compactness_info'][0] / 1000.0:.3f}",
        f"{values['pop_info'][0] * 100.0:.0f}",
        *(f"{value:.2f}" for value in values["urban_land_use_info"]),
        *(f"{value:.2f}" for value in values["cloud_info"]),
        *(f"{value:.0f}" for value in values["solar_info"]),
        *(f"{value:.2f}" for value in values["precipitation_info"]),
        *(f"{value:.2f}" for value in values["snow_info"]),
        *(f"{value:.2f}" for value in values["evi_info"]),
        *(f"{value:.2f}" for value in values["ndvi_info"]),
        *(f"{value:.2f}" for value in values["pm25_info"]),
        *(f"{value:.2f}" for value in values["temperature_info"]),
        *(f"{value:.2f}" for value in values["vapour_info"]),
        *(f"{value:.2f}" for value in values["wind_info"]),
    ]
    if len(pieces) != 147:
        raise ValueError(f"numeric list expected 147 values, got {len(pieces)}")
    return "[" + ", ".join(pieces) + "]\n"


def create_numeric_plain_prompt(info, index: int = 0) -> str:
    """Create the unlabeled numeric-list input used by the prompt ablation."""
    return _numeric_list_prompt(info, index=index)


def format_numeric_plain_prompt(info):
    target = _target_value(info)
    return _numeric_list_prompt(info) + f"{target:.3f}\n"


def format_numeric_plain_prompt_PPO(info):
    target = _target_value(info)
    return {
        "prompt": _numeric_list_prompt(info),
        "completion": f"{target:.3f}\n",
        "ground_truth": info["y"][0],
    }


def _format_series(values, decimal_places=2):
    return ", ".join(f"{value:.{decimal_places}f}" for value in values)


def _structured_prompt(info, index=None, target=None):
    values = _structured_values(info, index=index)
    current_time, time_range = _prompt_time_context(
        int(round(values["year_info"][0])),
        int(round(values["month_info"][0])),
    )

    # The processed fields use 0.01-km² spatial cells and a compactness scale
    # factor of 1,000. Convert them to the publication-facing units here.
    built_surface_per_km2 = values["built_surface_info"][0] * 100.0
    population_per_km2 = values["pop_info"][0] * 100.0
    compactness = values["compactness_info"][0] / 1000.0

    historical_block = "\n".join(
        [
            PROMPT["Historical"],
            PROMPT["UHI_data"].format(_format_series(values["x"], 3)),
        ]
    )
    structural_block = "\n".join(
        [
            PROMPT["Structural"],
            PROMPT["Area"].format(values["area_info"][0]),
            PROMPT["Built_height"].format(values["built_height_info"][0]),
            PROMPT["Built_surface"].format(built_surface_per_km2),
            PROMPT["Urbanization_level"].format(
                values["urbanization_level_info"][0]
            ),
            PROMPT["Location"].format(*values["location_info"]),
            PROMPT["Compactness"].format(compactness),
            PROMPT["Pop"].format(population_per_km2),
            PROMPT["Urban_land_use"].format(*values["urban_land_use_info"]),
        ]
    )
    liquidity_block = "\n".join(
        [
            PROMPT["Liquidity"],
            PROMPT["Cloud"].format(_format_series(values["cloud_info"])),
            # Solar values remain in joules; no J-to-kJ scaling is applied.
            PROMPT["Solar"].format(_format_series(values["solar_info"], 0)),
            PROMPT["Precipitation"].format(
                _format_series(values["precipitation_info"])
            ),
            PROMPT["Snow"].format(_format_series(values["snow_info"])),
            PROMPT["Evi"].format(_format_series(values["evi_info"])),
            PROMPT["Ndvi"].format(_format_series(values["ndvi_info"])),
            PROMPT["PM25"].format(_format_series(values["pm25_info"])),
            PROMPT["Temperature"].format(
                _format_series(values["temperature_info"])
            ),
            PROMPT["Vapour"].format(_format_series(values["vapour_info"])),
            PROMPT["Wind"].format(_format_series(values["wind_info"])),
        ]
    )
    prompt = "\n\n".join(
        [
            PROMPT["Instruction_"].format(current_time, time_range),
            historical_block,
            structural_block,
            liquidity_block,
            PROMPT["Task"],
        ]
    )
    rendered_target = "" if target is None else f" {float(target):.3f} °C"
    prompt += PROMPT["Result"].format(rendered_target)
    if target is not None:
        prompt += "\n"
    return prompt


def _target_value(info, index=None):
    target_values = _flatten_numeric(_select_numeric_field(info, "y", index))
    if len(target_values) != 1:
        raise ValueError(
            f"structured prompt expected 1 value for y, got {len(target_values)}"
        )
    return float(target_values[0])


def create_prompt(info, index: int = 0) -> str:
    """Create a structured inference prompt from batched tensor-style fields."""
    return _structured_prompt(info, index=index)


def create_result_only_prompt(info, index: int = 0) -> str:
    """Create the structured prompt with only ``[Result:]`` as its output cue."""
    prompt = _structured_prompt(info, index=index)
    standard_suffix = PROMPT["Result"].format("")
    if not prompt.endswith(standard_suffix):
        raise ValueError("Structured inference prompt has an unexpected result suffix.")
    return prompt[: -len(standard_suffix)] + "\n\nPrediction\n\n[Result:]"


def create_task_only_prompt(info, index: int = 0) -> str:
    """Create the structured prompt ending immediately after the task text."""
    prompt = _structured_prompt(info, index=index)
    standard_suffix = PROMPT["Result"].format("")
    if not prompt.endswith(standard_suffix):
        raise ValueError("Structured inference prompt has an unexpected result suffix.")
    return prompt[: -len(standard_suffix)]


def format_prompt(info):
    """Create a structured supervised-training example with its target."""
    return _structured_prompt(info, target=_target_value(info))


def format_prompt_Lime(info):
    """Create the same target-free structured prompt for LIME inference."""
    return _structured_prompt(info)


def format_prompt_PPO(info):
    """Split the common structured representation into PPO prompt/completion."""
    target = _target_value(info)
    return {
        "prompt": _structured_prompt(info),
        "completion": f" {target:.3f} °C\n",
        "ground_truth": info["y"][0],
    }


def format_task_only_prompt_PPO(info):
    """Create a PPO example whose prompt ends immediately after the task."""
    target = _target_value(info)
    prompt = _structured_prompt(info)
    standard_suffix = PROMPT["Result"].format("")
    if not prompt.endswith(standard_suffix):
        raise ValueError("Structured PPO prompt has an unexpected result suffix.")
    return {
        "prompt": prompt[: -len(standard_suffix)],
        "completion": f" {target:.3f} °C\n",
        "ground_truth": info["y"][0],
    }


def _value_to_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.flatten()[0].item()
    if isinstance(value, (list, tuple)):
        return _value_to_scalar(value[0])
    if isinstance(value, numbers.Number):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def sort_dataset_by_year_month(dataset):
    """Return a dataset view sorted by (year_info, month_info)."""
    indices = sorted(
        range(len(dataset)),
        key=lambda idx: (
            _value_to_scalar(dataset[idx]["year_info"]),
            _value_to_scalar(dataset[idx]["month_info"]),
        ),
    )
    return dataset.select(indices)


class EarlyStopping(object):
    def __init__(self):
        self.patience = 5
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            return True


__all__ = [
    "PROMPT",
    "create_prompt",
    "create_task_only_prompt",
    "create_result_only_prompt",
    "create_numeric_plain_prompt",
    "format_numeric_plain_prompt",
    "format_numeric_plain_prompt_PPO",
    "format_prompt",
    "format_prompt_Lime",
    "format_prompt_PPO",
    "EarlyStopping",
]
