# UHI-E3LLM

UHI-E3LLM is an end-to-end large language model (LLM)-based framework for
forecasting monthly Urban Heat Island (UHI) intensity and producing
factor-level explanations. The codebase includes data preprocessing,
supervised fine-tuning (SFT), PPO-based alignment, LoRA adapter merging,
inference and metric export, and LIME-based explanation generation.

> [!IMPORTANT]
> **For results that are consistent with the manuscript and for the best
> predictive performance, we strongly recommend
> [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B).**
> The smaller
> [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
> may be used on resource-limited machines to check installation and the
> inference workflow, but its outputs are not expected to reproduce the
> manuscript results or match the performance of the 8B model.

## Project structure

```text
UHI_E3LLM/
├── Dataset/                             # Training, validation, and test datasets
├── model/                               # Base or merged models downloaded from Hugging Face
├── model_adapters/                      # LoRA adapters, such as SFT and PPO adapters
├── scripts/
│   └── Preprocess_new.ipynb             # Data-preprocessing notebook
├── src/
│   ├── SFT_trainer.py                   # Supervised fine-tuning with LoRA
│   ├── Inference.py                     # Batched inference and metric export
│   ├── PPOTrainer.py                    # PPO alignment loop
│   ├── merge_lora.py                    # Merge one LoRA adapter into a base model
│   ├── merge_multiple_adapters.py       # Weighted merge of multiple LoRA adapters
│   ├── Lime_explanation.py              # LIME-based feature attribution
│   ├── pretrain_sample.py               # Lightweight pretraining on explanation data
│   ├── model_chat.py                    # Single-example text generation
│   ├── sample.py                        # Collect explanation data through the DeepSeek API
│   └── utils.py                         # Helper functions
├── README.md
└── requirements.txt
```

## Environment

The experiments were conducted using the environment listed below.

| Component | Experimental environment |
|---|---|
| Operating system | Ubuntu 20.04.6 LTS, Linux kernel 5.15.0 |
| CPU | Intel Xeon Platinum 8368 @ 2.40 GHz |
| System RAM | 1.0 TiB |
| GPU | NVIDIA A100-SXM4-80GB |
| NVIDIA driver | 575.57.08 |
| CUDA version | CUDA 11.8 |
| Python | 3.9 |

## Recommended environment for the demo

A representative normal desktop configuration for the resource-limited demo
is:

- 64-bit Linux, or Windows 11 with WSL2;
- an 8-core or better desktop CPU;
- 32 GB system RAM;
- an NVIDIA GeForce RTX 5060 Ti with 16 GB VRAM;
- at least 50 GB of free SSD space.

This configuration is intended for running the
DeepSeek-R1-Distill-Qwen-1.5B smoke-test workflow. It can be used to verify
installation, model loading, inference, and output generation.
For results consistent with the manuscript, we strongly recommend using
DeepSeek-R1-Distill-Llama-8B.

## Installation guide

### 1. Obtain the source code

```bash
git clone https://github.com/ZJU-DAILY/UHI_E3LLM.git
cd UHI_E3LLM
```

### 2. Create an isolated Python environment

Using Conda:

```bash
conda create -n uhi-e3llm python=3.9
conda activate uhi-e3llm
```

Alternatively, using `venv`:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install a PyTorch build that matches the local CUDA runtime by following the
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/),
then install the remaining dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For PPO training, the current code requires a different TRL version:

```bash
python -m pip uninstall -y trl
python -m pip install trl==0.11.2
```

Use a separate environment for PPO if both the default and PPO-specific TRL
versions are required.

### 4. Download a base model

Install the Hugging Face command-line client:

```bash
python -m pip install "huggingface_hub[cli]"
huggingface-cli login
```

#### Recommended model: DeepSeek-R1-Distill-Llama-8B

Use this model for manuscript-consistent evaluation and the best supported
results:

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --local-dir model/deepseek-8b \
  --local-dir-use-symlinks False
```

#### Resource-limited alternative: DeepSeek-R1-Distill-Qwen-1.5B

Use this smaller model only for an installation or workflow smoke test unless
the repository also provides a separately trained, model-compatible UHI
adapter:

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --local-dir model/deepseek-qwen-1.5b \
  --local-dir-use-symlinks False
```

### Typical installation time

- environment creation and Python-package installation: approximately
  **10–30 minutes** on a typical broadband-connected desktop;
- repository download: usually **less than 5 minutes**;
- model download: approximately **30–120 minutes**, depending on the selected
  model, network bandwidth, connection stability, and Hugging Face server
  availability. The actual download time may vary substantially.

## Dataset

The data archive is available on Zenodo:
<https://doi.org/10.5281/zenodo.17846841>.

After downloading and preprocessing, `src/Inference.py` expects a directory
with the following structure:

```text
DATASET_PATH/
├── city_metrics/
│   ├── city_metrics_train.csv
│   └── city_metrics_test.csv
├── train/
│   └── UHI_city_<city_id>/
└── valid/
    └── UHI_city_<city_id>/
```

The `city_metrics_*.csv` files must contain an `id` column. Each
`UHI_city_<city_id>` directory must be a Hugging Face Dataset saved with
`datasets.Dataset.save_to_disk`. See `scripts/Preprocess_new.ipynb` for the
preprocessing workflow.

To run the demo, select several city samples from the dataset, place them in
the corresponding `train/` and `valid/` directories, and list their IDs in
`city_metrics/city_metrics_test.csv`.

## Instructions for use

The SFT, inference, PPO, and LIME scripts resolve their default data, model,
adapter, log, and output paths from the repository root. The commands below
assume that they are launched from the repository root. Pass `--dataset_path`
or `--save_path` to override the relevant defaults.

### Supervised fine-tuning (LoRA)

```bash
python src/SFT_trainer.py \
  --model deepseek-8b \
  --dataset_path Dataset
```

Outputs are written under `model_adapters/SFT/<model>`.

### Inference and metrics

```bash
python src/Inference.py \
  --device cuda \
  --model deepseek-8b \
  --adapter_path SFT \
  --checkpoint CHECKPOINT_NAME \
  --dataset_path DATASET_PATH \
  --mode test \
  --save_path inference_results
```

Replace `CHECKPOINT_NAME` and `DATASET_PATH` with the relevant adapter
checkpoint and processed dataset directory. The script produces per-city CSV
files and an aggregate metrics CSV.

### PPO alignment

Use the PPO-specific environment described in the installation guide, then run:

```bash
python src/PPOTrainer.py \
  --device cuda \
  --model deepseek-8b_merged \
  --dataset_path Dataset \
  --batch_size 1 \
  --train_epochs 2
```

PPO adapter checkpoints are saved under
`model_adapters/PPO/<model>/epoch<epoch>`, while TensorBoard logs are written
under `results_PPO/<model>/`.

### Merge a LoRA adapter into the base model

```bash
python src/merge_lora.py \
  --model deepseek-8b \
  --adapter_path SFT \
  --checkpoint CHECKPOINT_NAME
```

This creates a merged model under `model/<model>_merged`.

### Merge multiple adapters

```bash
python src/merge_multiple_adapters.py \
  --model deepseek-8b \
  --adapter_root model_adapters/SFT \
  --merged_name ensemble \
  --output_dir model_adapters/merged \
  --weights 0.5 0.3 0.2
```

Weights are optional. If omitted, adapters under
`model_adapters/SFT/<model>` are merged with equal weights into
`model_adapters/merged/<model>`.

### LIME attributions

```bash
python src/Lime_explanation.py \
  --model deepseek-8b_merged \
  --adapter_path PPO \
  --checkpoint epoch0 \
  --dataset_path DATASET_PATH \
  --save_path Lime_explanation \
  --num_samples 1000
```

The script saves per-sample LIME plots under
`Lime_explanation/City<id>`. By default, it processes the city IDs listed in
`DATASET_PATH/city_metrics/city_metrics_test.csv`. Use
`--city_ids <id1> <id2> ...` to process a specified subset.

### Expected output

A successful run should produce the following outputs:

- **SFT:** LoRA adapter checkpoints under
  `model_adapters/SFT/<model>/`.
- **LoRA adapter merging:** a merged Hugging Face model under
  `model/<model>_merged/`.
- **PPO training:** PPO adapter checkpoints under
  `model_adapters/PPO/<model>/epoch<epoch>` and TensorBoard logs under
  `results_PPO/<model>/`.
- **Explanation-text distillation:** a distilled explanation-model checkpoint
  and the corresponding training logs.
- **Inference:** per-city prediction files and an aggregate metrics file under
  the directory specified by `--save_path`.
- **Explanation generation:** a non-empty factor-level explanation for each
  processed sample.
- **LIME analysis:** per-sample attribution plots under
  `Lime_explanation/City<id>/`.

With `--save_path inference_results`, `--model deepseek-8b`, and
`--adapter_path SFT`, the inference output has the following structure:

```text
inference_results/
└── <model>/
    └── <adapter_name>/
        ├── city_results/
        │   ├── city_<city_id_1>_predictions.csv
        │   ├── city_<city_id_2>_predictions.csv
        │   └── ...
        └── <adapter_name>_<checkpoint>_inference.csv
```

Each per-city prediction file contains:

```text
city_idx, year, month, year_month, true_uhi, predicted_uhi, mae
```

The aggregate inference file contains:

```text
city_idx, mae, mse, rmse
```

## API key handling

`src/sample.py` uses the DeepSeek API. Do not place an API key in source code or
commit it to the repository. Set it as an environment variable:

```bash
export DEEPSEEK_API_KEY="your-key"
```

The script reads `DEEPSEEK_API_KEY` and uses
`https://api.deepseek.com` as the base URL. API access is not required for the
offline inference demo when all necessary model and adapter files have already
been downloaded.

## Estimated runtime on the recommended demo configuration

The following estimates assume the resource-limited demo configuration:
an NVIDIA GeForce RTX 5060 Ti with 16 GB VRAM, 32 GB system RAM, an
8-core or better CPU, and SSD storage. The estimates apply to the 1.5B
workflow and are provided for planning purposes rather than as measured
benchmarks.

| Stage | Estimated runtime |
|---|---:|
| SFT | 8–14 h |
| LoRA adapter merging | 1–5 min |
| PPO training | 28–48 h |
| Explanation-text distillation | 10–30 min |
| Inference and explanation generation | 1–3 h |

## License

The original source code in this repository is available under the MIT License. 