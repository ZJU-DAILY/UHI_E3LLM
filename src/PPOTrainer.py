import argparse
import datetime
import logging
import random
import re
import pandas as pd
import os

import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from datasets import concatenate_datasets, load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from tqdm import tqdm

from utils import PROMPT, format_prompt_PPO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# logger.info("Test logger")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_length', type=int, default=1300)
    parser.add_argument('--model', type=str, default='deepseek-8b_merged')
    parser.add_argument('--target_modules', nargs='*', default=["q_proj", "v_proj"])
    parser.add_argument('--quant', type=bool, default=False)
    parser.add_argument('--gradient_checkpointing', type=bool, default=False)
    parser.add_argument('--dataset_path', type=str, default="Dataset")
    args = parser.parse_args()

    pd_train = pd.read_csv(f'{args.dataset_path}/city_metrics/city_metrics_train.csv')
    city_id_list = pd_train['id'].tolist()
    
    for index, i in enumerate(city_id_list):
        if index == 0:
            train_dataset = load_from_disk(f"{args.dataset_path}/train/UHI_city_{i}")
        else:
            city_data = load_from_disk(f"{args.dataset_path}/train/UHI_city_{i}")
            train_dataset = concatenate_datasets([train_dataset, city_data])

    print(train_dataset)
    print("Train dataset init done.")
    train_dataset = train_dataset.map(format_prompt_PPO)
    print("Train dataset format done.")
    

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        fan_in_fan_out=True,
        target_modules = args.target_modules,
    )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    model_dir = os.path.join(project_root, "model", args.model)

    if args.quant == True:
        print("Quant model")
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype="float16",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_storage="bfloat16",
        # )
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_dir,
        dtype="bfloat16",
        peft_config=peft_config,
        trust_remote_code=True,
        local_files_only=True,
    )

    if args.model == 'qwen-7b_merged':
        print("Qwen model")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            pad_token='<|endoftext|>',
            local_files_only=True,
        )
        tokenizer.eos_token = tokenizer.pad_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
    def tokenize(sample):
        encoded = tokenizer(
            sample["prompt"],
            padding=True,        
            truncation=True,             
            max_length=args.max_length,  
            return_tensors="pt",
        )
        sample["input_ids"] = encoded["input_ids"].squeeze()  
        sample["attention_mask"] = encoded["attention_mask"].squeeze()
        return sample
    train_dataset = train_dataset.map(tokenize, batched=False)
    print("Train dataset tokenized done.")

    # trl 0.11.2
    ppo_config = PPOConfig(
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        batch_size=args.batch_size,
        remove_unused_columns=False,
        is_peft_model=True,
        learning_rate=3e-6,
        gradient_checkpointing = args.gradient_checkpointing,
        log_with="tensorboard",
        project_kwargs={'logging_dir': f"./results_PPO/{args.model}"},
    )
    def collate_fn(batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
        # print(batch)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ground_truth": [item["ground_truth"] for item in batch],
        }

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collate_fn,
    )
    
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 20
    }

    print("Start training")
    for epoch in tqdm(range(args.train_epochs), "epoch: "):
        total_loss = 0
        index = 0
        for batch in tqdm(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]

            generated_tensors = ppo_trainer.generate(
                query_tensor=query_tensors,
                **generation_kwargs
            )
            response_tensors = [r[q.size(0):].squeeze() for q, r in zip(query_tensors, generated_tensors)]
            batch["response"] = [tokenizer.decode(r) for r in response_tensors]

            rewards = []
            for i, text in enumerate(batch["response"]):
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)

                if len(numbers) > 0:
                    result = float(numbers[0])
                    if result > 10:
                        logger.info("Empty result:")
                        logger.info(text)
                        result = batch['ground_truth'][i] + 3 
                else:
                    logger.info("Empty result:")
                    # logger.info(text)
                    result = batch['ground_truth'][i] + 3
                
                loss = abs(batch['ground_truth'][i] - result)
                total_loss += loss

                rewards.append(torch.tensor(0.15 - loss))

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            index += 1

        ppo_trainer.save_pretrained(f"./model_adapters/PPO/{args.model}/epoch{epoch}")

    
    




