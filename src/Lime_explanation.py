import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lime import lime_tabular
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from Inference import load_local_lora_adapter
from metrics import target_in_period
from output_parsing import parse_prediction
from sequence_dataset import load_city_dataset
from utils import sort_dataset_by_year_month, format_prompt_Lime

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
    def __init__(self, args, text_models, tokenizer):
        if len(text_models) != 2:
            raise ValueError("LIME requires both distilled ensemble members")
        self.args = args
        self.text_models = text_models
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

    @torch.inference_mode()
    def predict(self, instances):
        predictions = []

        for i in range(0, len(instances), self.args.batch_size):
            batch = instances[i:i + self.args.batch_size]
            prompts = [self._tabular_to_prompt(instance) for instance in batch]
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.max_length
            ).to(self.args.device)

            member_predictions = []
            for member_index, model in enumerate(self.text_models):
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=self.args.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                )
                generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
                values = []
                for seq in generated_tokens:
                    text = self.tokenizer.decode(seq, skip_special_tokens=True)
                    parsed = parse_prediction(
                        text,
                        valid_min=self.args.valid_min,
                        valid_max=self.args.valid_max,
                        max_decimal_places=self.args.max_decimal_places,
                    )
                    if not parsed.is_valid:
                        raise ValueError(
                            f"LIME ensemble member {member_index + 1} produced "
                            f"an invalid output ({parsed.invalid_reason}): '{text[:120]}'"
                        )
                    values.append(parsed.parsed_value)
                member_predictions.append(values)
            predictions.extend(np.mean(member_predictions, axis=0).tolist())

        return np.array(predictions)

def explain_tabular(args, models, tokenizer, city_idx):

    dataset = load_city_dataset(args.dataset_path, city_idx, "valid")
    dataset = dataset.filter(target_in_period, load_from_cache_file=False)
    dataset = sort_dataset_by_year_month(dataset)
    if len(dataset) != 36:
        raise RuntimeError(
            f"City {city_idx} must have 36 LIME targets from 2018-01 through "
            f"2020-12; found {len(dataset)}"
        )
    print(f"City {city_idx} evaluation dataset size: {len(dataset)}")

    dataset_loader = DataLoader(dataset, batch_size=1, drop_last=True)

    X, y = get_tabular_data(dataset_loader)
    feature_names = _get_feature_names()

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X,
        feature_names=feature_names,
        categorical_features=[0, 1],
        mode="regression",
        verbose=True,
        discretize_continuous=False,
        random_state=args.seed,
    )

    predictor = TabularPredictor(args, models, tokenizer)

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
        prompt_text += f"{y[sample_idx]:.3f}\n"
        with open(f"{image_dir}/prompt_{sample_idx}.txt", "w") as f:
            f.write(prompt_text)

    return exp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=1300)
    parser.add_argument('--max_new_tokens', type=int, default=19)
    parser.add_argument(
        '--models',
        type=str,
        nargs=2,
        default=[
            'deepseek-8b_merged__Global500_PPO_epoch0_merged',
            'deepseek-8b_merged__Global500_PPO_epoch1_merged',
        ],
        help="Materialized base models for the two distilled policies.",
    )
    parser.add_argument(
        '--adapter_dirs',
        type=str,
        nargs=2,
        default=[
            os.path.join(
                PROJECT_ROOT, 'model_adapters', 'explanation_distilled',
                'deepseek-8b_merged__Global500', f'epoch{epoch}',
            )
            for epoch in (0, 1)
        ],
        help="Explanation adapters corresponding to --models, in the same order.",
    )
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260720)
    parser.add_argument('--valid_min', type=float, default=-5.0)
    parser.add_argument('--valid_max', type=float, default=5.0)
    parser.add_argument('--max_decimal_places', type=int, default=3)
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=os.path.join(PROJECT_ROOT, "data"),
    )
    parser.add_argument('--city_ids', type=int, nargs='+')
    parser.add_argument(
        '--save_path',
        type=str,
        default=os.path.join(PROJECT_ROOT, "outputs", "lime_explanation"),
    )

    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.max_length <= 0 or args.max_new_tokens <= 0:
        parser.error("generation lengths must be positive")
    if args.valid_min >= args.valid_max or args.max_decimal_places < 0:
        parser.error("numerical output constraints are invalid")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    models = []
    for model_name, adapter_dir in zip(args.models, args.adapter_dirs):
        model_dir = os.path.join(PROJECT_ROOT, "model", model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(args.device)
        model = load_local_lora_adapter(model, adapter_dir)
        model.eval()
        models.append(model)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(PROJECT_ROOT, "model", args.models[0]),
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

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
        explain_tabular(args, models, tokenizer, city_idx)
