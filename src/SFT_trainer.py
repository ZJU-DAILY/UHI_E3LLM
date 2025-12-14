import argparse
import datetime
import os
import random

import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from datasets import concatenate_datasets, load_from_disk
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import pandas as pd
from tqdm import tqdm

from utils import PROMPT, format_prompt

os.environ["WANDB_DISABLED"] = "true"

# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)
# torch.cuda.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_length', type=int, default=1300)
    parser.add_argument('--model', type=str, default='deepseek-8b')
    parser.add_argument('--target_modules', nargs='*', default=["q_proj", "v_proj"])
    parser.add_argument('--quant', type=bool, default=False)
    parser.add_argument('--adapter_path', type=str, default="")
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--gradient_checkpointing', type=bool, default=True)
    parser.add_argument('--resume_from_checkpoint', type=bool, default=False)
    parser.add_argument('--dataset_path', type=str, default="./Dataset")
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--ddp_find_unused_parameters', action='store_true', default=False)
    args = parser.parse_args()

    df_train = pd.read_csv(f'{args.dataset_path}/city_metrics/city_metrics_train.csv')
    city_id_list = df_train['id'].tolist()
    print(f"Dataset size: {len(city_id_list)}")

    progress_total = min(len(city_id_list), 51)
    for i, city_id in enumerate(tqdm(city_id_list, desc="Loading city datasets", total=progress_total)):
        if i == 0:
            train_dataset = load_from_disk(f"{args.dataset_path}/train/UHI_city_{city_id}")

        else:
            city_data = load_from_disk(f"{args.dataset_path}/train/UHI_city_{city_id}")
            train_dataset = concatenate_datasets([train_dataset, city_data])

    print("Train dataset init done.")

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        fan_in_fan_out=True,
        target_modules = args.target_modules
    )

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

    if args.quant == True:
        print("Quant model")
        model = AutoModelForCausalLM.from_pretrained(
            f'../nature/model/{args.model}',
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype="bfloat16",
        )

    else:
        print("No quant")
        model = AutoModelForCausalLM.from_pretrained(
            f'../nature/model/{args.model}',
            trust_remote_code=True,
            torch_dtype="bfloat16"
        )

        print(model)

    # tokenizer = AutoTokenizer.from_pretrained(f'model/{args.model}', trust_remote_code=True, pad_token='<|endoftext|>')
    tokenizer = AutoTokenizer.from_pretrained(f'../nature/model/{args.model}', trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # print(tokenizer.pad_token)

    # model.config.pad_token_id = tokenizer.pad_token_id


    if args.gradient_checkpointing == True:
        print("Gradient checkpointing enabled.")
        model.gradient_checkpointing_enable()

    if args.adapter_path != "":
        print(f"Adapter path: {args.adapter_path}")
        print(f"Checkpoint: {args.checkpoint}")
        model = PeftModel.from_pretrained(model, f'logs/{args.adapter_path}/{args.checkpoint}', is_trainable=True)

    
    model.tokenizer = tokenizer

    training_args = SFTConfig(
        num_train_epochs = args.train_epochs,
        per_device_train_batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        max_seq_length = args.max_length,
        # keep in sync with inference/merge scripts that expect model_adapters/*
        output_dir = f'model_adapters/SFT/{args.model}',
        bf16 = True,
        bf16_full_eval = True,
        save_steps = 500,
        gradient_accumulation_steps = args.gradient_accumulation,
        dataloader_num_workers = args.num_workers,
        ddp_find_unused_parameters = args.ddp_find_unused_parameters,
        report_to = "none"
    )

    if args.adapter_path == "":
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config,
            formatting_func=format_prompt,
        )

    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config,
            formatting_func=format_prompt,
        )

    # trainer.accelerator.print(f"{trainer.model}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # trainer.train()
    trainer.save_model()
    print(f"Model adapter saved to {training_args.output_dir}")
    
    
