# 新增依赖
import argparse
import datetime
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta
from datasets import load_from_disk
from lime import lime_tabular
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import PROMPT, sort_dataset_by_year_month, format_prompt_Lime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _get_feature_names():
    names = []

    names += ['year_info', 'month_info']
    struct_features = [
        'longitude', 'latitude',
        'area', 'pop', 'built_surface',
        'built_height',
        'urbanization_level', 'compactness',
        'open_space', 'water', 'road',
        'residential', 'non_residential',
    ]
    names += struct_features
    liquid_features = [
        'wind', 'temperature', 'cloud',
        'precipitation', 'snow', 'vapour',
        'solar', 'ndvi', 'evi', 'pm25',
    ]
    for feat in liquid_features:
        names += [f"{feat}_{i+1}" for i in range(12)]
        
    names += [f"past_uhi_{i+1}" for i in range(12)]
    return names
    
def get_tabular_data(dataset_loader):
    tabular_data = []
    result = []
    for info in dataset_loader:

        x = [tensor.item () for tensor in info['x'][0]]
        
        y = info['y'][0].item()

        year = info['year_info'][0].item()
        month = info['month_info'][0].item()
        
        structure_list = [
            'location_info', 'area_info', 'pop_info', 'built_surface_info', 'built_height_info',
            'urbanization_level_info', 'compactness_info', 'urban_land_use_info',
        ]
        
        liquid_list = [
            'wind_info', 'temperature_info', 'cloud_info', 'precipitation_info',
            'snow_info', 'vapour_info', 'solar_info', 'ndvi_info', 'evi_info', 'pm25_info',
            
        ]
        
        struct = []
        liquid = []
        
        for key in structure_list:
            struct.extend([tensor.item () for tensor in info[key][0]])
            
        for key in liquid_list:
            liquid.extend([tensor.item () for tensor in info[key][0]])

        tabular_sample = np.concatenate([
            np.array([year, month], dtype=np.float32),
            np.array(struct, dtype=np.float32),
            np.array(liquid, dtype=np.float32),
            x
        ])
        tabular_data.append(tabular_sample)
        result.append(y)

    return np.array(tabular_data), np.array(result)

class TabularPredictor:
    def __init__(self, args, text_model, tokenizer):
        self.args = args
        self.text_model = text_model
        self.tokenizer = tokenizer
        
    def _tabular_to_prompt(self, instance, y=None):
        raw_data = instance
        info = {}
        pointer = 0

        info = {
            'year_info': raw_data[pointer],
            'month_info': raw_data[pointer+1]
        }
        pointer+=2

        struct = {
            'location_info': raw_data[pointer:pointer+2],
            'area_info': raw_data[pointer+2],
            'pop_info': raw_data[pointer+3],
            'built_surface_info': raw_data[pointer+4],
            'built_height_info': raw_data[pointer+5],
            'urbanization_level_info': raw_data[pointer+6],
            'compactness_info': raw_data[pointer+7],
            'urban_land_use_info': raw_data[pointer+8:pointer+13],
        }
        pointer +=13
        info.update(struct)
        
        liquid = {
            'wind_info': raw_data[pointer:pointer+12],
            'temperature_info': raw_data[pointer+12:pointer+24],
            'cloud_info': raw_data[pointer+24:pointer+36],
            'precipitation_info': raw_data[pointer+36:pointer+48],
            'snow_info': raw_data[pointer+48:pointer+60],
            'vapour_info': raw_data[pointer+60:pointer+72],
            'solar_info': raw_data[pointer+72:pointer+84],
            'ndvi_info': raw_data[pointer+84:pointer+96],
            'evi_info': raw_data[pointer+96:pointer+108],
            'pm25_info': raw_data[pointer+108:pointer+120]
        }
        pointer += 120
        info.update(liquid)
        uhi = {'x': raw_data[pointer:pointer+12]}
        pointer += 12
        info.update(uhi)

        prompt = format_prompt_Lime(info)

        return prompt

    def predict(self, instances):
        predictions = []
        
        def batch_inference(args, model, prompts):
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length
            ).to(args.device)

            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=19,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )

            generated_tokens = outputs[:, inputs.input_ids.shape[1]:]

            for seq in generated_tokens:
                text = self.tokenizer.decode(seq, skip_special_tokens=True)
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
                predictions.append(float(numbers[0]))

            return predictions

        for i in range(0, len(instances), args.batch_size):
            batch = instances[i:min(i+args.batch_size, len(instances))]
            prompts = list()
            for instance in batch:
                prompts.append(self._tabular_to_prompt(instance))
            
            predictions = batch_inference(args, model, prompts)

        return np.array(predictions)

def explain_tabular(args, model, tokenizer, city_idx):

    dataset = load_from_disk(f"{args.dataset_path}/valid/UHI_city_{city_idx}")
    print(f"City {city_idx} dataset size: {len(dataset)}")
    dataset = sort_dataset_by_year_month(dataset)

    dataset_loader = DataLoader(dataset, batch_size=1, drop_last=True)

    X, y = get_tabular_data(dataset_loader)
    feature_names = _get_feature_names()

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X,
        feature_names=feature_names,
        categorical_features=[0, 1],
        mode="regression",
        verbose=True,
        discretize_continuous=False
    )

    predictor = TabularPredictor(args, model, tokenizer)
    
    print("Predictor done.")

    image_dir = os.path.join(args.save_path, f"City{city_idx}")
    os.makedirs(image_dir, exist_ok=True)

    for sample_idx in range(len(dataset)):
        print(f"Processing sample {sample_idx}...")
        exp = explainer.explain_instance(
            X[sample_idx], 
            predictor.predict,
            num_features=147,
            num_samples=args.num_samples
        )

        fig = exp.as_pyplot_figure()
        plt.title(f"Feature Importance for Sample {sample_idx}")
        plt.savefig(f"{image_dir}/tabular_explanation_{sample_idx}.png")
        plt.close()

        with open(f"{image_dir}/explanation_{sample_idx}.txt", "w") as f:
            f.write("Top Features:\n")
            for feat, weight in exp.as_list():
                f.write(f"{feat}: {weight:.4f}\n")

        instance = X[sample_idx]
        prompt_text = predictor._tabular_to_prompt(instance)
        prompt_text += PROMPT['Result'].format(f"{y[sample_idx]:.3f}")
        with open(f"{image_dir}/prompt_{sample_idx}.txt", "w") as f:
            f.write(prompt_text)
        
    return exp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=1300)
    parser.add_argument('--model', type=str, default='deepseek-8b_merged')
    parser.add_argument('--adapter_path', type=str, default="PPO")
    parser.add_argument('--checkpoint', type=str, default='epoch0')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=os.path.join(PROJECT_ROOT, "Dataset"),
    )
    parser.add_argument('--city_ids', type=int, nargs='+')
    parser.add_argument(
        '--save_path',
        type=str,
        default=os.path.join(PROJECT_ROOT, "Lime_explanation"),
    )

    args = parser.parse_args()
    model_dir = os.path.join(PROJECT_ROOT, "model", args.model)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
    ).to(args.device)
    if args.adapter_path != "":
        adapter_dir = os.path.join(
            PROJECT_ROOT,
            "model_adapters",
            args.adapter_path,
            args.model,
            args.checkpoint,
        )
        model = PeftModel.from_pretrained(model, adapter_dir).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if args.city_ids is None:
        city_metrics_path = os.path.join(
            args.dataset_path,
            "city_metrics",
            "city_metrics_test.csv",
        )
        city_id_list = pd.read_csv(city_metrics_path)['id'].tolist()
    else:
        city_id_list = args.city_ids

    for city_idx in city_id_list:
        explain_tabular(args, model, tokenizer, city_idx)
