import datetime
from dateutil.relativedelta import relativedelta
import torch
import numpy as np
import numbers

PROMPT = {
    'Instruction_': 'Instruction: You are an advanced Urban Heat Island (UHI) model designed to forecast the future '
                    'UHI intensity ({0}) for the next month based on historical UHI data ({1}) and other UHI-related '
                    'factors, including structure factors (vary annually) and liquidity factors (vary monthly).\n',
    'UHI_data': 'UHI_data: {} (°C)\n',
    'Area': 'Area: {:.2f} km²\n',
    'Built_height': 'Built Height: {:.2f} m\n',
    'Built_surface': 'Built Surface Area: {:.2f} m²\n',
    'Compactness': 'City Compactness: {:.2f}\n',
    'Urbanization_level': 'Urbanization Level: {:.2f}\n',
    'Location': 'Location: ({0:.3f}, {1:.3f})\n',
    'Urban_land_use': 'Urban Land Use: Open Spaces: {0:.2f}%, '
                      'Water Surfaces: {1:.2f}%, '
                      'Road Surfaces: {2:.2f}%, '
                      'Residential: {3:.2f}%, '
                      'Non-Residential: {4:.2f}% ',
    'Cloud': 'Average Cloud Cover: {} m/s\n',
    'Evi': 'Enhanced Vegetation Index (EVI): {}\n',
    'Ndvi': 'Normalized Difference Vegetation Index (NDVI): {}\n',
    'PM25': 'PM2.5 Concentration: {} µg/m³\n',
    'Pop': 'Population Density: {:.2f} people/0.01km²\n',
    'Precipitation': 'Precipitation Flux: {} mm/day\n',
    'Snow': 'Average Snow Thickness: {} m\n',
    'Solar': 'Solar Radiation: {} J m-2d-1\n',
    'Temperature': 'Average Temperature: {} K\n',
    'Vapour': 'Average Vapour Pressure: {} hPa\n',
    'Wind': 'Average Wind Speed: {} m/s\n',
    'Task': 'Task: Based on the provided historical UHI data to predict the UHI intensity for the next month.\n',
    'Result': 'Result: Predicted UHI intensity for the next month: {}\n',
}


def create_prompt(info, index: int = 0) -> str:
    """Create the UHI forecast prompt using batched tensor-style fields.

    Matches the inference flow: no ground-truth value appended.
    """
    current_year = info['year_info'][0][index].item()
    current_month = info['month_info'][0][index].item()
    current_time = f"{current_year}-{current_month:02d}"

    current_date = datetime.datetime(current_year, current_month, 1)
    end_date = current_date - relativedelta(months=1)
    start_date = end_date - relativedelta(months=11)

    start_time = start_date.strftime("%Y-%m")
    end_time = end_date.strftime("%Y-%m")
    time_range = f"{start_time} to {end_time}"

    prompt = PROMPT['Instruction_'].format(current_time, time_range)

    prompt += PROMPT['UHI_data'].format(
        ",".join([f"{val:.3f}" for val in [tensor[index].item() for tensor in info['x'][0]]])
    )

    prompt += "Structure Factors (1 data point, annual): \n"
    prompt += PROMPT['Area'].format(info['area_info'][0][index].item())
    prompt += PROMPT['Built_height'].format(info['built_height_info'][0][index].item())
    prompt += PROMPT['Built_surface'].format(info['built_surface_info'][0][index].item())
    prompt += PROMPT['Urbanization_level'].format(info['urbanization_level_info'][0][index].item())
    prompt += PROMPT['Compactness'].format(info['compactness_info'][0][index].item())
    prompt += PROMPT['Urban_land_use'].format(*[tensor[index].item() for tensor in info['urban_land_use_info'][0]])
    prompt += PROMPT['Location'].format(*[tensor[index].item() for tensor in info['location_info'][0]])
    prompt += PROMPT['Pop'].format(info['pop_info'][0][index].item())

    prompt += "Liquidity Factors (12 data points, monthly): \n"
    cloud_str = ",".join([f"{val:.2f}" for val in [tensor[index].item() for tensor in info['cloud_info'][0]]])
    prompt += PROMPT['Cloud'].format(cloud_str)

    solar_str = ",".join([f"{val:.2f}" for val in [tensor[index].item() for tensor in info['solar_info'][0]]])
    prompt += PROMPT['Solar'].format(solar_str)

    precip_str = ",".join([f"{val:.2f}" for val in [tensor[index].item() for tensor in info['precipitation_info'][0]]])
    prompt += PROMPT['Precipitation'].format(precip_str)

    snow_str = ",".join([f"{val:.2f}" for val in [tensor[index].item() for tensor in info['snow_info'][0]]])
    prompt += PROMPT['Snow'].format(snow_str)

    evi_str = ",".join([f"{val:.2f}" for val in [tensor[index].item() for tensor in info['evi_info'][0]]])
    prompt += PROMPT['Evi'].format(evi_str)

    ndvi_str = ",".join([f"{val:.2f}" for val in [tensor[index].item() for tensor in info['ndvi_info'][0]]])
    prompt += PROMPT['Ndvi'].format(ndvi_str)

    pm25_str = ",".join([f"{val:.2f}" for val in [tensor[index].item() for tensor in info['pm25_info'][0]]])
    prompt += PROMPT['PM25'].format(pm25_str)

    temp_str = ",".join([f"{val:.2f}" for val in [tensor[index].item() for tensor in info['temperature_info'][0]]])
    prompt += PROMPT['Temperature'].format(temp_str)

    vapour_str = ",".join([f"{val:.2f}" for val in [tensor[index].item() for tensor in info['vapour_info'][0]]])
    prompt += PROMPT['Vapour'].format(vapour_str)

    wind_str = ",".join([f"{val:.2f}" for val in [tensor[index].item() for tensor in info['wind_info'][0]]])
    prompt += PROMPT['Wind'].format(wind_str)

    prompt += PROMPT['Task']
    return prompt

## with ground-truth value appended
def format_prompt(info):
    current_year = info['year_info'][0]
    current_month = info['month_info'][0]

    current_time = f"{current_year}-{current_month:02d}"

    current_date = datetime.datetime(current_year, current_month, 1)
    end_date = current_date - relativedelta(months=1)
    start_date = end_date - relativedelta(months=11)

    start_time = start_date.strftime("%Y-%m")
    end_time = end_date.strftime("%Y-%m")
    time_range = f"{start_time} to {end_time}"

    instruction = f'Instruction_'

    prompt = PROMPT[instruction].format(current_time, time_range)

    prompt += PROMPT['UHI_data'].format(",".join([f"{val:.3f}" for val in info['x'][0]]))

    prompt += "Structure Factors (1 data point, annual): \n"
    prompt += PROMPT['Area'].format(info['area_info'][0])
    prompt += PROMPT['Built_height'].format(info['built_height_info'][0])
    prompt += PROMPT['Built_surface'].format(info['built_surface_info'][0])
    prompt += PROMPT['Urbanization_level'].format(info['urbanization_level_info'][0])
    prompt += PROMPT['Compactness'].format(info['compactness_info'][0])
    prompt += PROMPT['Urban_land_use'].format(*info['urban_land_use_info'][0])
    prompt += PROMPT['Location'].format(*info['location_info'][0])
    prompt += PROMPT['Pop'].format(info['pop_info'][0])

    prompt += "Liquidity Factors (12 data points, monthly): \n"
    cloud_str = ",".join([f"{val:.2f}" for val in info['cloud_info'][0]])
    prompt += PROMPT['Cloud'].format(cloud_str)

    solar_str = ",".join([f"{val:.2f}" for val in info['solar_info'][0]])
    prompt += PROMPT['Solar'].format(solar_str)

    precip_str = ",".join([f"{val:.2f}" for val in info['precipitation_info'][0]])
    prompt += PROMPT['Precipitation'].format(precip_str)

    snow_str = ",".join([f"{val:.2f}" for val in info['snow_info'][0]])
    prompt += PROMPT['Snow'].format(snow_str)

    evi_str = ",".join([f"{val:.2f}" for val in info['evi_info'][0]])
    prompt += PROMPT['Evi'].format(evi_str)

    ndvi_str = ",".join([f"{val:.2f}" for val in info['ndvi_info'][0]])
    prompt += PROMPT['Ndvi'].format(ndvi_str)

    pm25_str = ",".join([f"{val:.2f}" for val in info['pm25_info'][0]])
    prompt += PROMPT['PM25'].format(pm25_str)

    temp_str = ",".join([f"{val:.2f}" for val in info['temperature_info'][0]])
    prompt += PROMPT['Temperature'].format(temp_str)

    vapour_str = ",".join([f"{val:.2f}" for val in info['vapour_info'][0]])
    prompt += PROMPT['Vapour'].format(vapour_str)

    wind_str = ",".join([f"{val:.2f}" for val in info['wind_info'][0]])
    prompt += PROMPT['Wind'].format(wind_str)

    prompt += PROMPT['Task']
    prompt += PROMPT['Result'].format(f"{info['y'][0]:.3f}")

    return prompt

def format_prompt_Lime(info):
    current_year = int(info['year_info'])
    current_month = int(info['month_info'])
    
    current_time = f"{current_year}-{current_month:02d}"

    current_date = datetime.datetime(current_year, current_month, 1)
    end_date = current_date - relativedelta(months=1)
    start_date = end_date - relativedelta(months=11)

    start_time = start_date.strftime("%Y-%m")
    end_time = end_date.strftime("%Y-%m")
    time_range = f"{start_time} to {end_time}"

    prompt = PROMPT['Instruction_'].format(current_time, time_range)

    prompt += PROMPT['UHI_data'].format(",".join([f"{val:.3f}" for val in info['x']]))

    prompt += "Structure Factors (1 data point, annual): \n"
    prompt += PROMPT['Area'].format(info['area_info'])
    prompt += PROMPT['Built_height'].format(info['built_height_info'])
    prompt += PROMPT['Built_surface'].format(info['built_surface_info'])
    prompt += PROMPT['Urbanization_level'].format(info['urbanization_level_info'])
    prompt += PROMPT['Compactness'].format(info['compactness_info'])
    prompt += PROMPT['Urban_land_use'].format(*info['urban_land_use_info'])
    prompt += PROMPT['Location'].format(*info['location_info'])
    prompt += PROMPT['Pop'].format(info['pop_info'])

    prompt += "Liquidity Factors (12 data points, monthly): \n"
    cloud_str = ",".join([f"{val:.2f}" for val in info['cloud_info']])
    prompt += PROMPT['Cloud'].format(cloud_str)
    
    solar_str = ",".join([f"{val:.2f}" for val in info['solar_info']])
    prompt += PROMPT['Solar'].format(solar_str)
    
    precip_str = ",".join([f"{val:.2f}" for val in info['precipitation_info']])
    prompt += PROMPT['Precipitation'].format(precip_str)
    
    snow_str = ",".join([f"{val:.2f}" for val in info['snow_info']])
    prompt += PROMPT['Snow'].format(snow_str)
        
    evi_str = ",".join([f"{val:.2f}" for val in info['evi_info']])
    prompt += PROMPT['Evi'].format(evi_str)
    
    ndvi_str = ",".join([f"{val:.2f}" for val in info['ndvi_info']])
    prompt += PROMPT['Ndvi'].format(ndvi_str)
    
    pm25_str = ",".join([f"{val:.2f}" for val in info['pm25_info']])
    prompt += PROMPT['PM25'].format(pm25_str)

    temp_str = ",".join([f"{val:.2f}" for val in info['temperature_info']])
    prompt += PROMPT['Temperature'].format(temp_str)
    
    vapour_str = ",".join([f"{val:.2f}" for val in info['vapour_info']])
    prompt += PROMPT['Vapour'].format(vapour_str)

    wind_str = ",".join([f"{val:.2f}" for val in info['wind_info']])
    prompt += PROMPT['Wind'].format(wind_str)

    prompt += PROMPT['Task']

    return prompt

def format_prompt_PPO(info):
    current_year = info['year_info'][0]
    current_month = info['month_info'][0]
    current_time = f"{current_year}-{current_month:02d}"

    current_date = datetime.datetime(current_year, current_month, 1)
    end_date = current_date - relativedelta(months=1)
    start_date = end_date - relativedelta(months=11)

    start_time = start_date.strftime("%Y-%m")
    end_time = end_date.strftime("%Y-%m")
    time_range = f"{start_time} to {end_time}"

    prompt = PROMPT['Instruction_'].format(current_time, time_range)

    prompt += PROMPT['UHI_data'].format(",".join([f"{val:.3f}" for val in info['x'][0]]))

    prompt += "Structure Factors (1 data point, annual): \n"
    prompt += PROMPT['Area'].format(info['area_info'][0])
    prompt += PROMPT['Built_height'].format(info['built_height_info'][0])
    prompt += PROMPT['Built_surface'].format(info['built_surface_info'][0])
    prompt += PROMPT['Urbanization_level'].format(info['urbanization_level_info'][0])
    prompt += PROMPT['Compactness'].format(info['compactness_info'][0])
    prompt += PROMPT['Urban_land_use'].format(*info['urban_land_use_info'][0])
    prompt += PROMPT['Location'].format(*info['location_info'][0])
    prompt += PROMPT['Pop'].format(info['pop_info'][0])

    prompt += "Liquidity Factors (12 data points, monthly): \n"
    cloud_str = ",".join([f"{val:.2f}" for val in info['cloud_info'][0]])
    prompt += PROMPT['Cloud'].format(cloud_str)
    
    solar_str = ",".join([f"{val:.2f}" for val in info['solar_info'][0]])
    prompt += PROMPT['Solar'].format(solar_str)
    
    precip_str = ",".join([f"{val:.2f}" for val in info['precipitation_info'][0]])
    prompt += PROMPT['Precipitation'].format(precip_str)
    
    snow_str = ",".join([f"{val:.2f}" for val in info['snow_info'][0]])
    prompt += PROMPT['Snow'].format(snow_str)
        
    evi_str = ",".join([f"{val:.2f}" for val in info['evi_info'][0]])
    prompt += PROMPT['Evi'].format(evi_str)
    
    ndvi_str = ",".join([f"{val:.2f}" for val in info['ndvi_info'][0]])
    prompt += PROMPT['Ndvi'].format(ndvi_str)
    
    pm25_str = ",".join([f"{val:.2f}" for val in info['pm25_info'][0]])
    prompt += PROMPT['PM25'].format(pm25_str)

    temp_str = ",".join([f"{val:.2f}" for val in info['temperature_info'][0]])
    prompt += PROMPT['Temperature'].format(temp_str)
    
    vapour_str = ",".join([f"{val:.2f}" for val in info['vapour_info'][0]])
    prompt += PROMPT['Vapour'].format(vapour_str)

    wind_str = ",".join([f"{val:.2f}" for val in info['wind_info'][0]])
    prompt += PROMPT['Wind'].format(wind_str)

    prompt += PROMPT['Task']

    return {"prompt": prompt, "completion": PROMPT['Result'].format(f"{info['y'][0]:.3f}"), "ground_truth": info['y'][0]}

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
            _value_to_scalar(dataset[idx]['year_info']),
            _value_to_scalar(dataset[idx]['month_info']),
        ),
    )
    return dataset.select(indices)


class EarlyStopping(object):
    def __init__(self):
        self.patience = 5
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            return True


__all__ = ['PROMPT', 'create_prompt', 'EarlyStopping']
