#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

MODEL="${MODEL:-deepseek-8b_merged__Global500}"
OUT_DIR="${ENSEMBLE_OUTPUT_DIR:-${OUTPUT_ROOT}/main_ensemble}"
PPO_ADAPTER_ROOT="${PPO_ADAPTER_ROOT:-${ROOT}/model_adapters/PPO/${MODEL}}"
INFERENCE_EPOCHS="${INFERENCE_EPOCHS:-0 1}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
INFERENCE_MAX_LENGTH="${INFERENCE_MAX_LENGTH:-1300}"
INFERENCE_MAX_NEW_TOKENS="${INFERENCE_MAX_NEW_TOKENS:-32}"
INFERENCE_MODE="${INFERENCE_MODE:-test}"
INVALID_POLICY="${INVALID_POLICY:-error}"
INFERENCE_INPUT_FORMAT="${INFERENCE_INPUT_FORMAT:-prompt_task_only}"
INFERENCE_EXPECTED_CITIES="${INFERENCE_EXPECTED_CITIES:-100}"
INFERENCE_EXPECTED_SAMPLES_PER_CITY="${INFERENCE_EXPECTED_SAMPLES_PER_CITY:-36}"
INFERENCE_LOG_PREFIX="${INFERENCE_LOG_PREFIX:-}"
inference_log_prefix="${INFERENCE_LOG_PREFIX:+${INFERENCE_LOG_PREFIX}_}"

if [[ "${INFERENCE_INPUT_FORMAT}" != "prompt" && "${INFERENCE_INPUT_FORMAT}" != "prompt_task_only" && "${INFERENCE_INPUT_FORMAT}" != "numeric_plain" ]]; then
  echo "INFERENCE_INPUT_FORMAT must be prompt, prompt_task_only, or numeric_plain." >&2
  exit 1
fi
if [[ "${INFERENCE_MODE}" != "test" && "${INFERENCE_MODE}" != "test_full" && "${INFERENCE_MODE}" != "validation" && "${INFERENCE_MODE}" != "train" ]]; then
  echo "INFERENCE_MODE must be test, test_full, validation, or train; got ${INFERENCE_MODE}." >&2
  exit 1
fi

read -r -a epochs <<< "${INFERENCE_EPOCHS}"
inference_inputs=()
for epoch in "${epochs[@]}"; do
  inference_inputs+=("${OUT_DIR}/epoch${epoch}")
  run_logged "${inference_log_prefix}inference_epoch${epoch}" env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON}" src/Inference.py \
    --model "${MODEL}" \
    --adapter_path PPO \
    --adapter_dir "${PPO_ADAPTER_ROOT}/epoch${epoch}" \
    --checkpoint "epoch${epoch}" \
    --dataset_path "${DATA_DIR}" \
    --save_path "${OUT_DIR}/epoch${epoch}" \
    --device cuda \
    --mode "${INFERENCE_MODE}" \
    --expected_cities "${INFERENCE_EXPECTED_CITIES}" \
    --expected_samples_per_city "${INFERENCE_EXPECTED_SAMPLES_PER_CITY}" \
    --batch_size "${INFERENCE_BATCH_SIZE}" \
    --max_length "${INFERENCE_MAX_LENGTH}" \
    --max_new_tokens "${INFERENCE_MAX_NEW_TOKENS}" \
    --valid_min -5 \
    --valid_max 5 \
    --max_decimal_places 3 \
    --input_format "${INFERENCE_INPUT_FORMAT}" \
    --invalid_policy "${INVALID_POLICY}"
done

run_logged "${inference_log_prefix}ensemble_epoch0_epoch1" "${PYTHON}" scripts/average_predictions.py \
  --input "${inference_inputs[@]}" \
  --output "${OUT_DIR}/ensemble" \
  --expected-cities "${INFERENCE_EXPECTED_CITIES}" \
  --expected-samples-per-city "${INFERENCE_EXPECTED_SAMPLES_PER_CITY}"
