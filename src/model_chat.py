import argparse
import random

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_length', type=int, default=2000)
    parser.add_argument('--model', type=str, default='deepseek-8b')
    parser.add_argument('--adapter_path', type=str, default='SFT')
    parser.add_argument('--checkpoint', type=str, default='.')

    args = parser.parse_args()
        
    
    input = \
'''
Instruction: You are an advanced Urban Heat Island (UHI) model designed to forecast the future UHI intensity (2017-12) for the next month based on historical UHI data (2016-12 to 2017-11) and other UHI-related factors, including structure factors (vary annually) and liquidity factors (vary monthly).
UHI_data: 0.381,0.365,0.489,0.496,0.502,0.664,0.334,0.763,0.783,0.834,0.564,0.416 (°C)
Structure Factors (1 data point, annual): 
Area: 751.39 km²
Built Height: 6.35 m
Built Surface Area: 1492.72 m²
Urbanization Level: 26.61
City Compactness: 141.07
Urban Land Use: Open Spaces: 27.73%, Water Surfaces: 0.74%, Road Surfaces: 2.63%, Residential: 26.64%, Non-Residential: 1.15% Location: (120.626, 27.784)
Population Density: 65.94 people/0.01km²
Liquidity Factors (12 data points, monthly): 
Average Cloud Cover: 0.48,0.59,0.53,0.73,0.64,0.76,0.94,0.66,0.59,0.74,0.49,0.75 m/s
Solar Radiation: 9537774.00,9068701.00,13440205.00,12153092.00,17798866.00,18765418.00,10813583.00,22706950.00,20769466.00,16667150.00,14009356.00,7712372.00 J m-2d-1
Precipitation Flux: 1.33,1.99,0.83,4.68,4.27,2.39,13.80,5.57,2.77,3.24,3.53,4.53 mm/day
Average Snow Thickness: 0.00,0.00,0.01,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00 m
Enhanced Vegetation Index (EVI): 0.18,0.18,0.17,0.17,0.19,0.24,0.24,0.26,0.27,0.28,0.22,0.17
Normalized Difference Vegetation Index (NDVI): 0.34,0.34,0.32,0.32,0.34,0.39,0.42,0.43,0.45,0.44,0.40,0.35
PM2.5 Concentration: 39.81,31.74,35.82,29.20,44.57,33.97,22.95,21.43,21.65,21.79,24.11,28.55 µg/m³
Average Temperature: 284.83,283.17,282.30,284.89,290.41,294.93,296.70,301.42,301.51,299.29,294.05,288.43 K
Average Vapour Pressure: 11.15,9.93,8.56,10.99,15.92,21.37,26.50,31.88,32.97,27.38,18.92,14.27 hPa
Average Wind Speed: 2.28,2.19,2.23,1.90,1.70,1.70,1.75,2.44,2.08,2.22,3.08,2.41 m/s
Task: Based on the provided historical UHI data to predict the UHI intensity for the next month.
'''

    print(input)
    model = AutoModelForCausalLM.from_pretrained(f'model/{args.model}', dtype=torch.float16).to(args.device)
    model = PeftModel.from_pretrained(model, f'model_adapters/{args.adapter_path}/{args.model}/{args.checkpoint}').to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(f'model/{args.model}', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(
                input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length
            ).to(args.device)
    
    input_ids = inputs["input_ids"]
    
    output = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=args.max_length,
                # max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
    
    result = output[0, input_ids.shape[1]:]

    text = tokenizer.decode(result, skip_special_tokens=True)
    print(text)