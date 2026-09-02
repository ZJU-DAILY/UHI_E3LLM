#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

BASE_MODEL="${BASE_MODEL:-deepseek-8b}"
MAIN_MODEL="${MAIN_MODEL:-deepseek-8b_merged__Global500}"
SFT_EPOCHS="${SFT_EPOCHS:-5}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-16}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-1300}"
SFT_NUM_WORKERS="${SFT_NUM_WORKERS:-4}"
SFT_SEED="${SFT_SEED:-42}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_UPDATE_EPOCHS="${PPO_UPDATE_EPOCHS:-4}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-3e-6}"
PPO_SEED="${PPO_SEED:-0}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
INVALID_POLICY="${INVALID_POLICY:-error}"
ABLATION_OVERWRITE="${ABLATION_OVERWRITE:-0}"
RUN_MAIN_ENSEMBLE="${RUN_MAIN_ENSEMBLE:-1}"
RUN_EXPLANATION_DISTILLATION="${RUN_EXPLANATION_DISTILLATION:-1}"
ABLATION_ADAPTER_ROOT="${ABLATION_ADAPTER_ROOT:-${ROOT}/model_adapters/ablations/modules}"
ABLATION_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-${OUTPUT_ROOT}/ablations}"
CANONICAL_EXPLANATION_CONTENT="${CANONICAL_EXPLANATION_CONTENT:-${DATA_DIR}/explanation_samples}"

for boolean_name in ABLATION_OVERWRITE RUN_MAIN_ENSEMBLE RUN_EXPLANATION_DISTILLATION; do
  value="${!boolean_name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${boolean_name} must be 0 or 1; got ${value}." >&2
    exit 1
  fi
done

if [[ ! "${SFT_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SFT_EPOCHS must be a positive integer; got ${SFT_EPOCHS}." >&2
  exit 1
fi
numeric_fusion_epochs=""
for ((epoch = 0; epoch < SFT_EPOCHS; epoch++)); do
  numeric_fusion_epochs+="${numeric_fusion_epochs:+ }${epoch}"
done

run_explanation_variant() {
  local name="$1"
  local base_model="$2"
  local policy_adapter="$3"
  local policy_epoch="$4"
  local input_format="$5"
  local data_root="$6"
  local reuse_content="$7"
  local merged_model="${base_model}_PPO_epoch${policy_epoch}_merged"
  local adapter_dir="${ABLATION_ADAPTER_ROOT}/${name}/explanation_distilled/epoch${policy_epoch}"
  if [[ -z "${policy_adapter}" ]]; then
    merged_model="${base_model}"
    adapter_dir="${ABLATION_ADAPTER_ROOT}/${name}/explanation_distilled/final"
  fi

  if [[ "${RUN_EXPLANATION_DISTILLATION}" == "0" ]]; then
    if [[ ! -s "${adapter_dir}/adapter_config.json" ]]; then
      echo "Required explanation adapter is missing: ${adapter_dir}" >&2
      exit 1
    fi
    return
  fi
  env \
    GPU_ID="${GPU_ID}" PYTHON="${PYTHON}" DATA_DIR="${DATA_DIR}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_DIR="${LOG_DIR}" \
    BASE_MODEL="${base_model}" \
    EXPLANATION_POLICY_ADAPTER_DIR="${policy_adapter}" \
    EXPLANATION_POLICY_EPOCH="${policy_epoch}" \
    MERGED_MODEL="${merged_model}" \
    MERGED_MODEL_DIR="${ROOT}/model/${merged_model}" \
    EXPLANATION_DATA_ROOT="${data_root}" \
    EXPLANATION_REUSE_CONTENT_DIR="${reuse_content}" \
    EXPLANATION_INPUT_FORMAT="${input_format}" \
    EXPLANATION_ADAPTER="${name}_epoch${policy_epoch}_explanation" \
    EXPLANATION_LOG_PREFIX="module_${name}_epoch${policy_epoch}" \
    EXPLANATION_ADAPTER_DIR="${adapter_dir}" \
    EXPLANATION_RUN_INFERENCE=0 \
    INVALID_POLICY="${INVALID_POLICY}" \
    bash scripts/run_explanation_distillation.sh
}

# ---------------------------------------------------------------------------
# 1. w/o UHI Knowledge Enhancer
# Only the structured prompt is removed. The same five-epoch SFT, greedy LoRA
# fusion, two-epoch PPO, two-checkpoint ensemble, and explanation distillation
# are retained.
# ---------------------------------------------------------------------------
numeric_name="without_uhi_knowledge"
numeric_sft="${ABLATION_ADAPTER_ROOT}/${numeric_name}/sft"
numeric_fused_adapter="${ABLATION_ADAPTER_ROOT}/${numeric_name}/fused_adapter"
numeric_model="${BASE_MODEL}_${numeric_name}_fused"
numeric_model_dir="${ROOT}/model/${numeric_model}"
numeric_ppo="${ABLATION_ADAPTER_ROOT}/${numeric_name}/ppo"
numeric_distilled_root="${ABLATION_ADAPTER_ROOT}/${numeric_name}/explanation_distilled"

env \
  GPU_ID="${GPU_ID}" PYTHON="${PYTHON}" DATA_DIR="${DATA_DIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_DIR="${LOG_DIR}" \
  MODEL="${BASE_MODEL}" SFT_OUTPUT="${numeric_sft}" \
  SFT_EPOCHS="${SFT_EPOCHS}" SFT_BATCH_SIZE="${SFT_BATCH_SIZE}" \
  SFT_MAX_LENGTH="${SFT_MAX_LENGTH}" SFT_NUM_WORKERS="${SFT_NUM_WORKERS}" \
  SFT_SEED="${SFT_SEED}" SFT_INPUT_FORMAT=numeric_plain \
  SFT_OVERWRITE_OUTPUT="${ABLATION_OVERWRITE}" \
  SFT_RUN_NAME=module_without_uhi_knowledge_sft \
  bash scripts/run_sft.sh

env \
  GPU_ID="${GPU_ID}" PYTHON="${PYTHON}" DATA_DIR="${DATA_DIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_DIR="${LOG_DIR}" \
  MODEL="${BASE_MODEL}" SFT_ADAPTER_ROOT="${numeric_sft}" \
  FUSION_EPOCHS="${numeric_fusion_epochs}" FUSION_INPUT_FORMAT=numeric_plain \
  FUSION_MAX_LENGTH="${SFT_MAX_LENGTH}" \
  OUTPUT_DIR="${numeric_fused_adapter}" MERGED_MODEL_DIR="${numeric_model_dir}" \
  FUSION_REPORT_DIR="${ABLATION_OUTPUT_ROOT}/${numeric_name}_fusion" \
  FUSION_RUN_NAME=module_without_uhi_knowledge_fusion \
  FUSION_OVERWRITE="${ABLATION_OVERWRITE}" \
  bash scripts/run_lora_fusion.sh

env \
  GPU_ID="${GPU_ID}" PYTHON="${PYTHON}" DATA_DIR="${DATA_DIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_DIR="${LOG_DIR}" \
  MODEL="${numeric_model}" PPO_OUTPUT="${numeric_ppo}" PPO_EPOCHS=2 \
  PPO_BATCH_SIZE="${PPO_BATCH_SIZE}" PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
  PPO_UPDATE_EPOCHS="${PPO_UPDATE_EPOCHS}" PPO_LEARNING_RATE="${PPO_LEARNING_RATE}" \
  PPO_MAX_LENGTH="${SFT_MAX_LENGTH}" PPO_INPUT_FORMAT=numeric_plain \
  PPO_BASELINE_OUTPUT="${ABLATION_OUTPUT_ROOT}/${numeric_name}_reward_baseline" \
  PPO_LOG_PREFIX=module_without_uhi_knowledge \
  PPO_SEED="${PPO_SEED}" PPO_OVERWRITE="${ABLATION_OVERWRITE}" \
  bash scripts/run_ppo.sh

if [[ ! -d "${CANONICAL_EXPLANATION_CONTENT}" ]]; then
  echo "Canonical explanation content is required: ${CANONICAL_EXPLANATION_CONTENT}" >&2
  echo "Run scripts/run_explanation_distillation.sh once for the full model first." >&2
  exit 1
fi
for policy_epoch in 0 1; do
  run_explanation_variant \
    "${numeric_name}" "${numeric_model}" "${numeric_ppo}/epoch${policy_epoch}" \
    "${policy_epoch}" numeric_plain \
    "${DATA_DIR}/explanation_samples_numeric_list" "${CANONICAL_EXPLANATION_CONTENT}"
done

env \
  GPU_ID="${GPU_ID}" PYTHON="${PYTHON}" DATA_DIR="${DATA_DIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_DIR="${LOG_DIR}" \
  BASE_MODEL="${numeric_model}" DISTILLED_ADAPTER_ROOT="${numeric_distilled_root}" \
  DISTILLED_OUTPUT_DIR="${ABLATION_OUTPUT_ROOT}/${numeric_name}_epochs" \
  DISTILLED_ENSEMBLE_OUTPUT="${ABLATION_OUTPUT_ROOT}/${numeric_name}" \
  DISTILLED_EPOCHS="0 1" DISTILLED_INPUT_FORMAT=numeric_plain \
  DISTILLED_EXPECTED_CITIES=100 DISTILLED_EXPECTED_SAMPLES_PER_CITY=36 \
  DISTILLED_BATCH_SIZE="${INFERENCE_BATCH_SIZE}" \
  DISTILLED_MAX_LENGTH="${SFT_MAX_LENGTH}" \
  DISTILLED_INVALID_POLICY="${INVALID_POLICY}" \
  DISTILLED_LOG_PREFIX=module_without_uhi_knowledge \
  bash scripts/run_distilled_ensemble_inference.sh

# ---------------------------------------------------------------------------
# 2. w/o Generalizability Enhancer
# Keep the structured prompt and explanation distillation, but remove LoRA
# fusion, PPO, and ensemble inference as one high-level module.
# ---------------------------------------------------------------------------
general_name="without_generalizability"
SINGLE_SFT_EPOCH="${SINGLE_SFT_EPOCH:-4}"
SINGLE_SFT_ADAPTER="${SINGLE_SFT_ADAPTER:-${ROOT}/model_adapters/SFT/${BASE_MODEL}/checkpoint-epoch${SINGLE_SFT_EPOCH}}"
if [[ -z "${SINGLE_SFT_ADAPTER}" || ! -s "${SINGLE_SFT_ADAPTER}/adapter_config.json" ]]; then
  echo "A single SFT adapter is required for ${general_name}." >&2
  exit 1
fi
single_sft_model="${BASE_MODEL}_${general_name}_single_sft"
single_sft_model_dir="${ROOT}/model/${single_sft_model}"
if [[ ! -s "${single_sft_model_dir}/config.json" ]]; then
  run_logged module_without_generalizability_merge env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON}" src/merge_lora.py \
    --model "${BASE_MODEL}" \
    --adapter_dir "${SINGLE_SFT_ADAPTER}" \
    --output_dir "${single_sft_model_dir}"
fi
run_explanation_variant \
  "${general_name}" "${single_sft_model}" "" 0 prompt \
  "${DATA_DIR}/explanation_samples" ""
run_logged module_without_generalizability_inference env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON}" src/Inference.py \
  --model "${single_sft_model}" \
  --adapter_path explanation \
  --adapter_dir "${ABLATION_ADAPTER_ROOT}/${general_name}/explanation_distilled/final" \
  --checkpoint "${general_name}" \
  --dataset_path "${DATA_DIR}" \
  --save_path "${ABLATION_OUTPUT_ROOT}/${general_name}" \
  --device cuda --mode test --expected_cities 100 --expected_samples_per_city 36 \
  --batch_size "${INFERENCE_BATCH_SIZE}" --max_length "${SFT_MAX_LENGTH}" \
  --max_new_tokens 19 --input_format prompt --invalid_policy "${INVALID_POLICY}"

# ---------------------------------------------------------------------------
# 3. w/o Explainability Enhancer
# Keep the complete numerical pipeline and omit only explanation distillation.
# Its numerical output is therefore exactly the main two-checkpoint ensemble.
# ---------------------------------------------------------------------------
if [[ "${RUN_MAIN_ENSEMBLE}" == "1" ]]; then
  env \
    GPU_ID="${GPU_ID}" PYTHON="${PYTHON}" DATA_DIR="${DATA_DIR}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_DIR="${LOG_DIR}" MODEL="${MAIN_MODEL}" \
    ENSEMBLE_OUTPUT_DIR="${ABLATION_OUTPUT_ROOT}/without_explainability_epochs" \
    INFERENCE_LOG_PREFIX=module_without_explainability \
    INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE}" INVALID_POLICY="${INVALID_POLICY}" \
    bash scripts/run_ensemble_inference.sh
fi
run_logged module_without_explainability_result "${PYTHON}" scripts/average_predictions.py \
  --input "${ABLATION_OUTPUT_ROOT}/without_explainability_epochs/epoch0" \
          "${ABLATION_OUTPUT_ROOT}/without_explainability_epochs/epoch1" \
  --output "${ABLATION_OUTPUT_ROOT}/without_explainability" \
  --expected-cities 100 --expected-samples-per-city 36

echo "Strict module-level ablations completed below ${ABLATION_OUTPUT_ROOT}."
