#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

BASE_MODEL="${BASE_MODEL:-deepseek-8b_merged__Global500}"
DISTILLED_ADAPTER_ROOT="${DISTILLED_ADAPTER_ROOT:-${ROOT}/model_adapters/explanation_distilled/${BASE_MODEL}}"
DISTILLED_MODEL_PREFIX="${DISTILLED_MODEL_PREFIX:-${BASE_MODEL}_PPO_epoch}"
DISTILLED_MODEL_SUFFIX="${DISTILLED_MODEL_SUFFIX:-_merged}"
DISTILLED_EPOCHS="${DISTILLED_EPOCHS:-0 1}"
DISTILLED_OUTPUT_DIR="${DISTILLED_OUTPUT_DIR:-${OUTPUT_ROOT}/main_ensemble}"
DISTILLED_ENSEMBLE_OUTPUT="${DISTILLED_ENSEMBLE_OUTPUT:-${DISTILLED_OUTPUT_DIR}/ensemble}"
DISTILLED_MODE="${DISTILLED_MODE:-test}"
DISTILLED_BATCH_SIZE="${DISTILLED_BATCH_SIZE:-8}"
DISTILLED_MAX_LENGTH="${DISTILLED_MAX_LENGTH:-6000}"
DISTILLED_MAX_NEW_TOKENS="${DISTILLED_MAX_NEW_TOKENS:-19}"
DISTILLED_INPUT_FORMAT="${DISTILLED_INPUT_FORMAT:-prompt}"
DISTILLED_EXPECTED_CITIES="${DISTILLED_EXPECTED_CITIES:-100}"
DISTILLED_EXPECTED_SAMPLES_PER_CITY="${DISTILLED_EXPECTED_SAMPLES_PER_CITY:-36}"
DISTILLED_CITY_LIST="${DISTILLED_CITY_LIST:-}"
DISTILLED_INVALID_POLICY="${DISTILLED_INVALID_POLICY:-error}"
DISTILLED_LOG_PREFIX="${DISTILLED_LOG_PREFIX:-distilled}"
DISTILLED_EFFICIENCY_RECORD="${DISTILLED_EFFICIENCY_RECORD:-}"

read -r -a epochs <<< "${DISTILLED_EPOCHS}"
if (( ${#epochs[@]} == 0 )); then
  echo "DISTILLED_EPOCHS must not be empty." >&2
  exit 1
fi
read -r -a city_ids <<< "${DISTILLED_CITY_LIST}"

inference_inputs=()
for epoch in "${epochs[@]}"; do
  if [[ ! "${epoch}" =~ ^[0-9]+$ ]]; then
    echo "Invalid distilled policy epoch: ${epoch}" >&2
    exit 1
  fi
  model_name="${DISTILLED_MODEL_PREFIX}${epoch}${DISTILLED_MODEL_SUFFIX}"
  adapter_dir="${DISTILLED_ADAPTER_ROOT}/epoch${epoch}"
  if [[ ! -s "${ROOT}/model/${model_name}/config.json" ]]; then
    echo "Distilled base model not found: ${ROOT}/model/${model_name}" >&2
    exit 1
  fi
  if [[ ! -s "${adapter_dir}/adapter_config.json" ]]; then
    echo "Explanation-distillation adapter not found: ${adapter_dir}" >&2
    exit 1
  fi

  command=(env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" src/Inference.py
    --model "${model_name}"
    --adapter_path explanation
    --adapter_dir "${adapter_dir}"
    --checkpoint "distilled_epoch${epoch}"
    --dataset_path "${DATA_DIR}"
    --save_path "${DISTILLED_OUTPUT_DIR}/epoch${epoch}"
    --device cuda
    --mode "${DISTILLED_MODE}"
    --expected_cities "${DISTILLED_EXPECTED_CITIES}"
    --expected_samples_per_city "${DISTILLED_EXPECTED_SAMPLES_PER_CITY}"
    --batch_size "${DISTILLED_BATCH_SIZE}"
    --max_length "${DISTILLED_MAX_LENGTH}"
    --max_new_tokens "${DISTILLED_MAX_NEW_TOKENS}"
    --input_format "${DISTILLED_INPUT_FORMAT}"
    --invalid_policy "${DISTILLED_INVALID_POLICY}")
  if (( ${#city_ids[@]} > 0 )); then
    command+=(--city_ids "${city_ids[@]}")
  fi
  if [[ -n "${DISTILLED_EFFICIENCY_RECORD}" ]]; then
    command+=(
      --efficiency_record "${DISTILLED_EFFICIENCY_RECORD%.json}_epoch${epoch}.json"
      --efficiency_member_epoch "${epoch}"
      --efficiency_batch_size 36)
  fi
  run_logged "${DISTILLED_LOG_PREFIX}_epoch${epoch}" "${command[@]}"
  inference_inputs+=("${DISTILLED_OUTPUT_DIR}/epoch${epoch}")
done

run_logged "${DISTILLED_LOG_PREFIX}_ensemble" "${PYTHON}" \
  scripts/average_predictions.py \
  --input "${inference_inputs[@]}" \
  --output "${DISTILLED_ENSEMBLE_OUTPUT}" \
  --expected-cities "${DISTILLED_EXPECTED_CITIES}" \
  --expected-samples-per-city "${DISTILLED_EXPECTED_SAMPLES_PER_CITY}"
