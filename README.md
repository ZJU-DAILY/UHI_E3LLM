# UHI_E3LLM
UHI_E3LLM is an end-to-end LLM-based model for forecasting monthly Urban Heat Island (UHI) intensity and producing factor-level explanations. The codebase includes supervised fine-tuning (SFT), PPO-based alignment, LoRA merge utilities, and explanation generation.

## Project structure

```text
UHI_E3LLM/
├─ Dataset/                             # Training and validation datasets
├─ model/                               # Base or merged models downloaded from Hugging Face
├─ model_adapters/                      # LoRA adapters, such as SFT, PPO, and merged adapters
├─ scripts/
│  └─ preprocess_new.ipynb              # Preprocessing notebook
├─ src/
│  ├─ SFT_trainer.py                    # Supervised fine-tuning with LoRA
│  ├─ Inference.py                      # Batched inference and metric export
│  ├─ PPOTrainer.py                     # PPO alignment loop
│  ├─ merge_lora.py                     # Merge a single LoRA adapter into the base model
│  ├─ merge_multiple_adapters.py        # Weighted merge of multiple LoRA adapters
│  ├─ Lime_explanation.py               # LIME-based feature attribution
│  ├─ generate_explanation_response.py  # Generate explanation samples
│  ├─ pretrain_sample.py                # Lightweight pretraining on explanation data
│  ├─ model_chat.py                     # Simple chat and text generation
│  ├─ sample.py                         # Collect explanation data through the DeepSeek API
│  └─ util.py                           # Helper functions
├─ README.md
└─ requirements.txt
```

## Environment setup
1) Python 3.9 recommended.
2) Install dependencies (choose the right torch build for your CUDA/Metal stack):
```bash
pip install -r requirements.txt
```

### Download base models (Hugging Face)
- Install the CLI: `pip install "huggingface_hub[cli]"`.
- Authenticate if needed: `huggingface-cli login`.
- Download a model and place it under `model/<model-name>` so scripts can find it, for example:
```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --local-dir model/deepseek-8b \
  --local-dir-use-symlinks False
```

Use a folder name that matches the `--model` flag you pass to training/inference scripts (e.g., `model/deepseek-8b`).

## Dataset
Data files are vailable on Zenodo at: https://doi.org/10.5281/zenodo.17846841

## Usage
### Supervised fine-tuning (LoRA)
```bash
python src/SFT_trainer.py \
  --device cuda \
  --model deepseek-8b \
```
Outputs are written to `model_adpters/SFT/<model>`.

### Inference and metrics
```bash
python src/Inference.py \
  --device cuda \
  --model deepseek-8b \
  --adapter_path SFT \
  --checkpoint . \
  --save_path inference_results
```
Produces per-city CSVs and an aggregate metrics CSV.

### PPO alignment
```bash
python src/PPOTrainer.py \
  --device cuda \
  --model deepseek-8b_merged \
  --batch_size 1 \
  --train_epochs 2
```
Results are stored under `results_PPO/`.

### Merge LoRA back to base
```bash
python src/merge_lora.py \
  --model deepseek-8b \
  --adapter_path SFT \
  --checkpoint . \
```
Creates a merged model in `model/<model>_merged`.

### Merge multiple adapters (weighted)
```bash
python src/merge_multiple_adapters.py \
  --model deepseek-8b \
  --adapter_root model_adapters/SFT \
  --merged_name ensemble \
  --output_dir model_adapters/merged \
  --weights 0.5 0.3 0.2
```
Weights are optional; if omitted, adapters under `model_adapters/SFT/<model>` are merged with equal weights into `model_adapters/merged/<model>`.

### LIME attributions
```bash
python src/Lime_explanation.py 
  --model deepseek-8b_merged
  --adapter_path PPO
  --checkpoint epoch0 
  --num_samples 1000
```
Saves per-sample LIME plots under `Lime_explanation/City<id>`.

## API key handling
`src/sample.py` expects a DeepSeek API key. Set it via environment variable before running:
```bash
export DEEPSEEK_API_KEY="your-key"
```
The script will read `DEEPSEEK_API_KEY` and keep the base URL `https://api.deepseek.com`.

## Notes

- Enable `--gradient_checkpointing` flags to reduce memory usage when supported.
- Quantization paths exist but are off by default; adjust `--quant` and `BitsAndBytesConfig` if needed.
- When producing PPO training, switch trl package to 0.11.2 manually:
```bash
    pip uninstall trl
    pip install trl==0.11.2
```

