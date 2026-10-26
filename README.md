# UHI-E3LLM

UHI-E3LLM is an end-to-end large language model (LLM)-based framework for forecasting monthly Urban Heat Island (UHI) intensity and generating factor-level explanations. The codebase provides the complete experimental workflows described in our paper, including data preprocessing, supervised fine-tuning (SFT), LoRA adapter fusion, PPO-based alignment, explainability enhancement, inference, baseline comparisons, leave-one-region-out evaluation, ablation studies, uncertainty quantification, and sensitivity analyses.

> [!IMPORTANT]
> **For results that are consistent with the manuscript and for the best
> predictive performance, we strongly recommend
> [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B).**
> The smaller
> [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
> may be used on resource-limited machines to check installation and the
> inference workflow, but its outputs are not expected to reproduce the
> manuscript results or match the performance of the 8B model.

## Repository structure

```text
UHI_E3LLM/
|-- baseline/                   # Four tabular and three time-series baselines
|-- data/                       # Versioned city manifests; downloaded datasets are ignored
|-- docs/                       # Inputs, splits, and manuscript-to-code map
|-- environment/                # Unified locked environment for all workflows
|-- scripts/                    # Executable manuscript workflows
|-- src/                        # UHI-E3LLM training, inference, reward, and parsing
|-- LICENSE
|-- README.md
`-- requirements.txt
```

Detailed documentation:

- [Dataset lineage and splits](docs/DATASET_AND_SPLITS.md)
- [Final model inputs](docs/INPUT_FEATURES.md)
- [Manuscript-to-code map](docs/MANUSCRIPT_MAP.md)

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
| Python | 3.10 |

```bash
conda env create -f environment/environment.yml
conda activate uhi-e3llm
```
## Recommended environment for the demo

A representative normal desktop configuration for the resource-limited demo
is:

- 64-bit Linux, or Windows 11 with WSL2;
- an 8-core or better desktop CPU;
- 32 GB system RAM;
- an NVIDIA GeForce RTX 5060 Ti with 16 GB VRAM;
- at least 50 GB of free SSD space.

This configuration is intended for running the
DeepSeek-R1-Distill-Qwen-1.5B workflow. It can be used to verify
installation, model loading, inference, and output generation.
For results consistent with the manuscript, we strongly recommend using
DeepSeek-R1-Distill-Llama-8B.

## Dataset

The complete processed dataset is published under CC BY 4.0:

- DOI: <https://doi.org/10.5281/zenodo.22161925>
- File: `Dataset.zip`

Download `Dataset.zip` from Zenodo and extract the archive. Place the extracted
`train/`, `valid/`, `city_metrics/`, `regional_folds/`, and
`explanation_samples/` directories directly under this repository's `data/`
directory.

## Base model

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --local-dir model/deepseek-8b \
  --local-dir-use-symlinks False
```

## Released model adapters

The trained LoRA adapters are available from [Kurofu/UHI_E3LLM](https://huggingface.co/Kurofu/UHI_E3LLM).

Download the adapter collection into the code-repository root:

```bash
huggingface-cli download Kurofu/UHI_E3LLM \
  --local-dir Deepseek8b_Adapters \
  --local-dir-use-symlinks False
```

The released artifacts are organized as follows:

```text
Deepseek8b_Adapters/
|-- sft/checkpoint-epoch0/ ... checkpoint-epoch4/
|-- SFT_fusion/
|-- ppo/epoch0/
|-- ppo/epoch1/
|-- distillation/epoch0/
|-- distillation/epoch1/
`-- run_all_stage_inference.sh
```

After downloading the processed dataset and base model, run the evaluation script with:

```bash
conda activate uhi-e3llm
GPU_ID=0 bash Deepseek8b_Adapters/run_all_stage_inference.sh
```

## Typical installation time

- environment creation and Python-package installation: approximately
  **10–30 minutes** on a typical broadband-connected desktop;
- repository download: usually **less than 5 minutes**;
- model download: approximately **30–120 minutes**, depending on the selected
  model, network bandwidth, connection stability, and Hugging Face server
  availability. The actual download time may vary substantially.


## Main pipeline

```bash
conda activate uhi-e3llm

bash scripts/run_sft.sh
bash scripts/run_lora_fusion.sh
bash scripts/run_ppo.sh
bash scripts/run_explanation_distillation.sh
bash scripts/run_main_evaluation.sh
bash scripts/run_efficiency_measurement.sh
```

Training `Total` is the sum of SFT, PPO, and student explanation distillation
for both policy epochs.

## Expected output

With the default paths, the complete pipeline produces the following main
artifacts. Individual output roots can be changed through the environment
variables accepted by the corresponding workflow scripts.

```text
data/explanation_samples/
|-- sample_1.txt
|-- ...
`-- sample_900.txt

model_adapters/
|-- SFT/deepseek-8b/
|   |-- checkpoint-epoch0/
|   |-- ...
|   `-- checkpoint-epoch4/
|-- merged/deepseek-8b/
|-- PPO/deepseek-8b_merged__Global500/
|   |-- epoch0/
|   `-- epoch1/
`-- explanation_distilled/deepseek-8b_merged__Global500/
    |-- epoch0/
    `-- epoch1/

model/
|-- deepseek-8b_merged__Global500/
|-- deepseek-8b_merged__Global500_PPO_epoch0_merged/
`-- deepseek-8b_merged__Global500_PPO_epoch1_merged/

outputs/
|-- lora_fusion/
|-- ppo_train_baseline/
|-- main_ensemble/
|   |-- epoch0/
|   |-- epoch1/
|   `-- ensemble/
|-- baselines/
|-- efficiency/
|   |-- raw/
|   `-- efficiency_measurements.csv
`-- tables/
    |-- table1_main_performance.csv
    |-- table2_leave_one_region_out.csv
    `-- table3_ablation.csv

logs/
`-- <workflow-name>.log
```

## Leave-one-region-out evaluation

```bash
bash scripts/run_leave_one_region_out.sh
```

## Ablation workflows

```bash
bash scripts/run_module_ablations.sh
bash scripts/run_component_ablations.sh
```

## Uncertainty workflows

```bash
bash scripts/run_city_bootstrap.sh
bash scripts/run_independent_calibration.sh
```

## Sensitivity workflows

```bash
bash scripts/run_ppo_penalty_sensitivity.sh
bash scripts/run_lora_similarity_threshold.sh
```

## Table reproduction

```bash
# Table 1: main test-set and city-size comparison
bash scripts/run_tables.sh table1

# Table 2: leave-one-region-out comparison
bash scripts/run_tables.sh table2

# Table 3: module- and component-level ablations
bash scripts/run_tables.sh table3

# Rebuild all three after all experiments are complete
bash scripts/run_tables.sh all
```

Table 1 requires `run_main_evaluation.sh`; Table 2 requires
`run_leave_one_region_out.sh`; Table 3 requires the main evaluation and both
ablation workflows.

The ablation results are reported in Table 3 and are not rendered as a
separate bar-chart panel by `scripts/run_figures.sh`.

## Figure reproduction

The plotting workflow covers the manuscript panels generated directly from
experimental outputs. Fig. 3(c--e), Fig. 5(b,d), and Fig. 6(b--d) were
manually composed using HTML, while Fig. 2(d--f) and Fig. 4(a) were manually
prepared in QGIS.

```bash
GPU_ID=0 bash scripts/run_case_study_outputs.sh
GPU_ID=0 bash scripts/run_main_evaluation.sh
GPU_ID=0 bash scripts/run_leave_one_region_out.sh
GPU_ID=0 bash scripts/run_module_ablations.sh
GPU_ID=0 bash scripts/run_component_ablations.sh
bash scripts/run_city_bootstrap.sh
bash scripts/run_independent_calibration.sh
bash scripts/run_ppo_penalty_sensitivity.sh
EFFICIENCY_RUN_MAIN_EVALUATION=0 bash scripts/run_efficiency_measurement.sh
bash scripts/run_figures.sh
```

## Evaluation workflows

The following table maps the executable workflows to the experiments, tables, and figures reported in the manuscript.

| Workflow | Experiment or analysis | Corresponding manuscript item |
|---|---|---|
| `scripts/run_main_evaluation.sh` | UHI-E3LLM and seven-baseline comparison on the unseen test cities | Table 1; Fig. 3(a) |
| `scripts/run_leave_one_region_out.sh` | Six-region holdout evaluation | Table 2 |
| `scripts/run_module_ablations.sh` | Module-level ablations | Table 3 |
| `scripts/run_component_ablations.sh` | Component-level ablations | Table 3 |
| `scripts/run_case_study_outputs.sh` | Five-city predictions and LIME analysis | Fig. 5(a,c) |
| `scripts/run_city_bootstrap.sh` | City-level bootstrap confidence intervals | Fig. 7(a) |
| `scripts/run_independent_calibration.sh` | Calibration-derived error radii and test-set coverage | Fig. 7(b) |
| `scripts/run_efficiency_measurement.sh` | Training-time and inference-time measurements | Fig. 7(c,d) |
| `scripts/run_ppo_penalty_sensitivity.sh` | PPO invalid-output penalty sensitivity | Fig. 7(e) |
| `scripts/run_lora_similarity_threshold.sh` | Adapter cosine similarities and threshold screening | LoRA adapter-fusion analysis |
| `scripts/run_tables.sh` | Assemble the main, regional, and ablation results | Tables 1–3 |
| `scripts/run_figures.sh` | Generate the statistical figure panels | Figs. 2(a–c,g,h), 3(a,b), 4(b,c), 5(a,c), and 7(a–e) |
| Manual preparation using HTML | Presentation panels | Figs. 3(c–e), 5(b,d), and 6(b–d) |
| Manual preparation using QGIS | Geographical maps | Figs. 2(d–f) and 4(a) |
| No plotting entry in the released repository | Framework illustration and representative ablation output | Figs. 1 and 6(a) |

## API key handling

The released `data/explanation_samples/sample_*.txt` files are consumed
directly by the default distillation workflow and do not require API access.
Regenerating those samples, or constructing fold-specific explanation data,
uses the DeepSeek API. Never store an API key in source code:

```bash
export DEEPSEEK_API_KEY="your-key"
```

API access is not required for numerical inference when the model and adapter
checkpoints are available locally.

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

UHI-E3LLM is released under the MIT License.
