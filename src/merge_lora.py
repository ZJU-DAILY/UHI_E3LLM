import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import PROMPT
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='deepseek-8b')
    parser.add_argument('--adapter_path', type=str, default='SFT')
    parser.add_argument('--checkpoint', type=str, default=".")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(f'model/{args.model}')
    model = AutoModelForCausalLM.from_pretrained(
        f'model/{args.model}',
        trust_remote_code=True,
        torch_dtype="bfloat16",
    )
    
    print(f"Adapter path: model_adapters/{args.adapter_path}")
    print(f"Checkpoint: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, f'model_adapters/{args.adapter_path}/{args.model}/{args.checkpoint}', is_trainable=True)

    print("Applying LoRA")
    model = model.merge_and_unload()
 
    output_path = f'model/{args.model}_merged'
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}.")
