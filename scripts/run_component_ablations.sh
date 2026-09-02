#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

BASE_MODEL="${BASE_MODEL:-deepseek-8b}"
FUSED_MODEL="${FUSED_MODEL:-deepseek-8b_merged__Global500}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-1300}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_UPDATE_EPOCHS="${PPO_UPDATE_EPOCHS:-4}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-3e-6}"
PPO_SEED="${PPO_SEED:-0}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
INVALID_POLICY="${INVALID_POLICY:-error}"
ABLATION_OVERWRITE="${ABLATION_OVERWRITE:-0}"
RUN_EXPLANATION_DISTILLATION="${RUN_EXPLANATION_DISTILLATION:-1}"
ABLATION_ADAPTER_ROOT="${ABLATION_ADAPTER_ROOT:-${ROOT}/model_adapters/ablations/components}"
ABLATION_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-${OUTPUT_ROOT}/ablations}"

for boolean_name in ABLATION_OVERWRITE RUN_EXPLANATION_DISTILLATION; do
  value="${!boolean_name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${boolean_name} must be 0 or 1; got ${value}." >&2
    exit 1
  fi
done

SINGLE_SFT_EPOCH="${SINGLE_SFT_EPOCH:-4}"
SINGLE_SFT_ADAPTER="${SINGLE_SFT_ADAPTER:-${ROOT}/model_adapters/SFT/${BASE_MODEL}/checkpoint-epoch${SINGLE_SFT_EPOCH}}"
if [[ -z "${SINGLE_SFT_ADAPTER}" || ! -s "${SINGLE_SFT_ADAPTER}/adapter_config.json" ]]; then
  echo "No single SFT checkpoint found; run scripts/run_sft.sh first." >&2
  exit 1
fi
MAIN_PPO_ROOT="${MAIN_PPO_ROOT:-${ROOT}/model_adapters/PPO/${FUSED_MODEL}}"
if [[ ! -s "${ROOT}/model/${FUSED_MODEL}/config.json" ]]; then
  echo "Fused SFT model not found: ${ROOT}/model/${FUSED_MODEL}" >&2
  exit 1
fi

run_explanation_variant() {
  local name="$1"
  local base_model="$2"
  local policy_adapter="$3"
  local policy_epoch="$4"
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
    EXPLANATION_DATA_ROOT="${DATA_DIR}/explanation_samples" \
    EXPLANATION_INPUT_FORMAT=prompt \
    EXPLANATION_ADAPTER="${name}_epoch${policy_epoch}_explanation" \
    EXPLANATION_LOG_PREFIX="component_${name}_epoch${policy_epoch}" \
    EXPLANATION_ADAPTER_DIR="${adapter_dir}" \
    EXPLANATION_RUN_INFERENCE=0 \
    INVALID_POLICY="${INVALID_POLICY}" \
    bash scripts/run_explanation_distillation.sh
}

# ---------------------------------------------------------------------------
# 1. w/o LoRA fusion
# Use one SFT adapter, then keep the same two PPO epochs, epoch0/epoch1
# ensemble, and explanation distillation as the full framework.
# ---------------------------------------------------------------------------
without_fusion_name="without_fusion"
single_model="${BASE_MODEL}_${without_fusion_name}_single_sft"
single_model_dir="${ROOT}/model/${single_model}"
if [[ ! -s "${single_model_dir}/config.json" ]]; then
  run_logged component_without_fusion_merge env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON}" src/merge_lora.py \
    --model "${BASE_MODEL}" \
    --adapter_dir "${SINGLE_SFT_ADAPTER}" \
    --output_dir "${single_model_dir}"
fi

without_fusion_ppo="${ABLATION_ADAPTER_ROOT}/${without_fusion_name}/ppo"
env \
  GPU_ID="${GPU_ID}" PYTHON="${PYTHON}" DATA_DIR="${DATA_DIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_DIR="${LOG_DIR}" \
  MODEL="${single_model}" PPO_OUTPUT="${without_fusion_ppo}" PPO_EPOCHS=2 \
  PPO_BATCH_SIZE="${PPO_BATCH_SIZE}" PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
  PPO_UPDATE_EPOCHS="${PPO_UPDATE_EPOCHS}" PPO_LEARNING_RATE="${PPO_LEARNING_RATE}" \
  PPO_MAX_LENGTH="${SFT_MAX_LENGTH}" PPO_INPUT_FORMAT=prompt_task_only \
  PPO_BASELINE_OUTPUT="${ABLATION_OUTPUT_ROOT}/${without_fusion_name}_reward_baseline" \
  PPO_LOG_PREFIX=component_without_fusion \
  PPO_SEED="${PPO_SEED}" PPO_OVERWRITE="${ABLATION_OVERWRITE}" \
  bash scripts/run_ppo.sh

for policy_epoch in 0 1; do
  run_explanation_variant \
    "${without_fusion_name}" "${single_model}" \
    "${without_fusion_ppo}/epoch${policy_epoch}" "${policy_epoch}"
done
env \
  GPU_ID="${GPU_ID}" PYTHON="${PYTHON}" DATA_DIR="${DATA_DIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_DIR="${LOG_DIR}" \
  BASE_MODEL="${single_model}" \
  DISTILLED_ADAPTER_ROOT="${ABLATION_ADAPTER_ROOT}/${without_fusion_name}/explanation_distilled" \
  DISTILLED_OUTPUT_DIR="${ABLATION_OUTPUT_ROOT}/${without_fusion_name}_epochs" \
  DISTILLED_ENSEMBLE_OUTPUT="${ABLATION_OUTPUT_ROOT}/${without_fusion_name}" \
  DISTILLED_EPOCHS="0 1" DISTILLED_INPUT_FORMAT=prompt \
  DISTILLED_EXPECTED_CITIES=100 DISTILLED_EXPECTED_SAMPLES_PER_CITY=36 \
  DISTILLED_BATCH_SIZE="${INFERENCE_BATCH_SIZE}" DISTILLED_MAX_LENGTH="${SFT_MAX_LENGTH}" \
  DISTILLED_INVALID_POLICY="${INVALID_POLICY}" \
  DISTILLED_LOG_PREFIX=component_without_fusion \
  bash scripts/run_distilled_ensemble_inference.sh

# ---------------------------------------------------------------------------
# 2. w/o PPO
# Keep the fused SFT model. The PPO-dependent ensemble has no valid candidates,
# so numerical inference is performed once; explanation distillation remains.
# ---------------------------------------------------------------------------
without_ppo_name="without_ppo"
run_explanation_variant "${without_ppo_name}" "${FUSED_MODEL}" "" 0
run_logged component_without_ppo_inference env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON}" src/Inference.py \
  --model "${FUSED_MODEL}" \
  --adapter_path explanation \
  --adapter_dir "${ABLATION_ADAPTER_ROOT}/${without_ppo_name}/explanation_distilled/final" \
  --checkpoint "${without_ppo_name}" \
  --dataset_path "${DATA_DIR}" \
  --save_path "${ABLATION_OUTPUT_ROOT}/${without_ppo_name}" \
  --device cuda --mode test --expected_cities 100 --expected_samples_per_city 36 \
  --batch_size "${INFERENCE_BATCH_SIZE}" --max_length "${SFT_MAX_LENGTH}" \
  --max_new_tokens 19 --input_format prompt --invalid_policy "${INVALID_POLICY}"

# ---------------------------------------------------------------------------
# 3. w/o ensemble inference
# Keep LoRA fusion and both sequential PPO training epochs, but evaluate only
# the final policy state (epoch1). Explanation distillation also uses epoch1.
# ---------------------------------------------------------------------------
without_ensemble_name="without_ensemble"
if [[ ! -s "${MAIN_PPO_ROOT}/epoch1/adapter_config.json" ]]; then
  echo "Final PPO epoch1 adapter not found: ${MAIN_PPO_ROOT}/epoch1" >&2
  echo "Run scripts/run_ppo.sh for two epochs before this ablation." >&2
  exit 1
fi
run_explanation_variant \
  "${without_ensemble_name}" "${FUSED_MODEL}" "${MAIN_PPO_ROOT}/epoch1" 1
env \
  GPU_ID="${GPU_ID}" PYTHON="${PYTHON}" DATA_DIR="${DATA_DIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_DIR="${LOG_DIR}" \
  BASE_MODEL="${FUSED_MODEL}" \
  DISTILLED_ADAPTER_ROOT="${ABLATION_ADAPTER_ROOT}/${without_ensemble_name}/explanation_distilled" \
  DISTILLED_OUTPUT_DIR="${ABLATION_OUTPUT_ROOT}/${without_ensemble_name}_epoch" \
  DISTILLED_ENSEMBLE_OUTPUT="${ABLATION_OUTPUT_ROOT}/${without_ensemble_name}" \
  DISTILLED_EPOCHS="1" DISTILLED_INPUT_FORMAT=prompt \
  DISTILLED_EXPECTED_CITIES=100 DISTILLED_EXPECTED_SAMPLES_PER_CITY=36 \
  DISTILLED_BATCH_SIZE="${INFERENCE_BATCH_SIZE}" DISTILLED_MAX_LENGTH="${SFT_MAX_LENGTH}" \
  DISTILLED_INVALID_POLICY="${INVALID_POLICY}" \
  DISTILLED_LOG_PREFIX=component_without_ensemble \
  bash scripts/run_distilled_ensemble_inference.sh

echo "Strict component-level ablations completed below ${ABLATION_OUTPUT_ROOT}."
