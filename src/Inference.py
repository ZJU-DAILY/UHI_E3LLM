import argparse
import math
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets, load_from_disk
from peft import PeftModel
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import create_prompt, sort_dataset_by_year_month

def inference(args, model, tokenizer, test_datasets):
    model.eval()
    # true_uhi = []
    # predict_uhi = []
    os.makedirs(args.save_path, exist_ok=True)
    model_save_dir = os.path.join(args.save_path, args.model)
    adapter_dir = model_save_dir if args.adapter_path == "" else os.path.join(model_save_dir, args.adapter_path)
    city_results_dir = os.path.join(adapter_dir, "city_results")
    os.makedirs(city_results_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    def process(dataloader, city_idx):
        criterion = nn.MSELoss()
        loss = 0
        mae_loss = 0

        def batch_inference(args, model, info_batch):

            y = info_batch['y'][0]
            if isinstance(y, torch.Tensor):
                batch_size = y.size(0)
            else:
                batch_size = len(y)
            prompts = []

            for i in range(batch_size):
                prompt = create_prompt(info_batch, i)
                # print(prompt)
                prompts.append(prompt)
                

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length
            ).to(args.device)

            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                max_new_tokens=19,
            )

            generated_tokens = outputs[:, inputs.input_ids.size(1):]

            results = []
            for seq in generated_tokens:
                text = tokenizer.decode(seq, skip_special_tokens=True)
                # print(text)
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
                if len(numbers) > 0:
                    results.append(float(numbers[0]))
                else:
                    print("Empty result:")
                    print(text)
                    results.append(0)
        
            true = list()
            result = list()
            loss = list()
            records = []
            
            if results is not None:
                for i in range(len(results)):
                    output = results[i]
                    output_tensor = torch.tensor(output, dtype=torch.float64).unsqueeze(0)
                    l = criterion(y[i], output_tensor)
                    if (output > 4):
                        print(f"Error: {output}")
                        true.append(y[i].item())
                        result.append(output)
                        loss.append(0)
                    else:
                        true.append(y[i].item())
                        result.append(output)
                        loss.append(l.item())
                    current_year = int(info_batch['year_info'][0][i].item())
                    current_month = int(info_batch['month_info'][0][i].item())
                    abs_error = abs(y[i].item() - output)
                    records.append({
                        'year': current_year,
                        'month': current_month,
                        'year_month': f"{current_year}-{current_month:02d}",
                        'true_uhi': y[i].item(),
                        'predicted_uhi': output,
                        'mae': abs_error
                    })
            
            return loss, true, result, records
            
        city_records = []
        total_samples = 0
        for index, info in enumerate(dataloader):
            l, true, output, records = batch_inference(args, model, info)
            loss += sum(l)
            for i in range(len(output)):
                mae = np.abs(true[i] - output[i])
                mae_loss += mae
            total_samples += len(true)
            if records:
                city_records.extend(records)

        if total_samples > 0:
            loss /= total_samples
            mae_loss /= total_samples
            rmse_loss = math.sqrt(loss)
        else:
            rmse_loss = 0
        if city_records:
            city_df = pd.DataFrame(city_records)
            city_df.insert(0, 'city_idx', city_idx)
            city_df.sort_values(by=['year', 'month'], inplace=True)
            city_df.reset_index(drop=True, inplace=True)
            city_df.to_csv(os.path.join(city_results_dir, f"city_{city_idx}_predictions.csv"), index=False)
        print(f"City {city_idx} test Finish, MAE Loss: {mae_loss}, MSE Loss: {loss}, RMSE Loss: {rmse_loss}")    
        return city_idx, mae_loss, loss, rmse_loss

    
    data = []

    for city_idx, test_dataset in test_datasets.items():
        print(f"Inference city {city_idx}")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False)
        # print(len(test_loader))
        city_idx, mae_loss, mse_loss, rmse_loss = process(test_loader, city_idx)
        data.append([city_idx, mae_loss, mse_loss, rmse_loss])

    df = pd.DataFrame(
        data,
        columns=['city_idx', 'mae', 'mse', 'rmse']
    )
    df.to_csv(os.path.join(adapter_dir, f"{args.adapter_path}_{args.checkpoint}_inference.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--max_length', type=int, default=1300)
    parser.add_argument('--model', type=str, default='deepseek-8b')
    parser.add_argument('--adapter_path', type=str, default='SFT')
    parser.add_argument('--checkpoint', type=str, default='.')
    parser.add_argument('--save_path', type=str, default='./inference_results')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--gradient_checkpointing', type=bool, default=False)
    parser.add_argument('--dataset_path', type=str, default="/home/zzh/fr/nature/Final_dataset_processed")

    args = parser.parse_args()
    

    model = AutoModelForCausalLM.from_pretrained(f'model/{args.model}', dtype=torch.float16, trust_remote_code=True).to(args.device)
    if args.adapter_path != "":
        print(f"Adapter: model_adapters/{args.adapter_path}")
        print(f"Checkpoint: {args.model}/{args.checkpoint}")
        model = PeftModel.from_pretrained(model, f'model_adapters/{args.adapter_path}/{args.model}/{args.checkpoint}')
    
    if args.model == 'qwen-7b' or args.model == 'qwen-7b_merged':
        print("Qwen model")
        tokenizer = AutoTokenizer.from_pretrained(f'model/{args.model}', trust_remote_code=True, pad_token='<|endoftext|>', padding_side='left')
    else:
        tokenizer = AutoTokenizer.from_pretrained(f'model/{args.model}', padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

    if args.gradient_checkpointing == True:
        model.gradient_checkpointing_enable()
    
    if args.mode == 'test':
        pd_test = pd.read_csv(f'{args.dataset_path}/city_metrics/city_metrics_test.csv')
    else:
        pd_test = pd.read_csv(f'{args.dataset_path}/city_metrics/city_metrics_train.csv')
    city_id_list = pd_test['id'].tolist()
    test_datasets = {}
    
    for index, i in enumerate(city_id_list):
        if args.mode == 'test':
            train_dataset = load_from_disk(f"{args.dataset_path}/train/UHI_city_{i}")
            valid_dataset = load_from_disk(f"{args.dataset_path}/valid/UHI_city_{i}")
            test_dataset = concatenate_datasets([train_dataset, valid_dataset])
            test_dataset = sort_dataset_by_year_month(test_dataset)
        else:
            test_dataset = load_from_disk(f"{args.dataset_path}/valid/UHI_city_{i}")
        test_datasets[i] = test_dataset

    print("Test dataset init done.")

    inference(args=args, model=model, tokenizer=tokenizer, test_datasets=test_datasets)
