import argparse
import os

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='deepseek-8b')
    parser.add_argument('--adapter_path', type=str, default='SFT')
    parser.add_argument('--checkpoint', type=str, default=".")
    parser.add_argument('--adapter_dir', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="")
    args = parser.parse_args()

    model_dir = os.path.join(project_root, 'model', args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype="bfloat16",
    )

    print(f"Adapter path: model_adapters/{args.adapter_path}")
    print(f"Checkpoint: {args.checkpoint}")
    adapter_dir = args.adapter_dir or os.path.join(
        project_root, 'model_adapters', args.adapter_path, args.model, args.checkpoint
    )
    model = PeftModel.from_pretrained(model, adapter_dir)

    print("Applying LoRA")
    model = model.merge_and_unload(safe_merge=True)

    output_path = args.output_dir or os.path.join(project_root, 'model', f'{args.model}_merged')
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}.")
