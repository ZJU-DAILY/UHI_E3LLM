import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
import argparse
from torch import nn

class PretrainDataset(Dataset):
    def __init__(self, dir_path, tokenizer, max_length=1300):
        self.tokenizer = tokenizer
        self.max_length = max_length

        def get_folders(path):
            entries = os.listdir(path) 
            folders = [
                os.path.join(path, entry) for entry in entries
                if os.path.isdir(os.path.join(path, entry)) 
            ]
            return folders
        
        folders = get_folders(dir_path)
        print(folders)

        self.file_paths = sorted([
            os.path.join(folder, f) 
            for folder in folders
            for f in os.listdir(folder) 
            if f.endswith('.txt')
        ])
        if not self.file_paths:
            raise ValueError(f"No txt files found in {dir_path}")

        self.samples = []
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()  
                
            # print(len(self.tokenizer.tokenize(text)))

            encoded = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            self.samples.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            })

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return {
            'input_ids': self.samples[idx]['input_ids'],
            'labels': self.samples[idx]['input_ids'].clone(),
            'attention_mask': self.samples[idx]['attention_mask']
        }

class PretrainModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_path = f"model/{args.model}"
        self.args = args
        self.device = torch.device(args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        save_path = f"{args.save_dir}/{args.model}"
        
        os.makedirs(save_path, exist_ok=True)
        self.model_path = save_path

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        )

        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.llm = get_peft_model(self.llm, self.lora_config)
        self.llm.to(self.device)

    def forward(self, input_ids, labels, attention_mask):
        return self.llm(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )
        
    def save_model(self):
        self.llm.half()
        self.llm.save_pretrained(self.model_path)

def pretrain(args, model, dataset):
    device = torch.device(args.device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['attention_mask'].to(device)

            outputs = model(input_ids=inputs, labels=labels, attention_mask=masks)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if global_step % args.log_interval == 0:
                print(f"Step {global_step} | Loss: {loss.item():.4f}")
                
            global_step += 1

    print("Saving final model...")
    model.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_path', type=str, default='explanation_samples/content')
    parser.add_argument('--model', type=str, default='deepseek-8b_merged_PPO')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_length', type=int, default=1500)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='model_adapters/explanation')
    parser.add_argument('--log_interval', type=int, default=100)

    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    model = PretrainModel(args)
    
    dataset = PretrainDataset(
        dir_path=args.data_path,
        tokenizer=model.tokenizer,
        max_length=args.max_length
    )

    pretrain(args, model, dataset)