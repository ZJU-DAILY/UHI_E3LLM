#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

# Produce the exact prediction and attribution inputs consumed by Fig. 5.
MODEL="${MODEL:-deepseek-8b_merged__Global500}"
DISTILLED_ADAPTER_ROOT="${DISTILLED_ADAPTER_ROOT:-${ROOT}/model_adapters/explanation_distilled/${MODEL}}"
CASE_CITY_LIST="${CASE_CITY_LIST:-413 661 222 677 864}"
read -r -a CASE_CITY_IDS <<< "${CASE_CITY_LIST}"
CASE_EXPECTED_CITIES="${#CASE_CITY_IDS[@]}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
INFERENCE_MAX_LENGTH="${INFERENCE_MAX_LENGTH:-1300}"
INVALID_POLICY="${INVALID_POLICY:-error}"
RUN_LIME="${RUN_LIME:-1}"
LIME_NUM_SAMPLES="${LIME_NUM_SAMPLES:-1000}"

if [[ "${RUN_LIME}" != "0" && "${RUN_LIME}" != "1" ]]; then
  echo "RUN_LIME must be 0 or 1; got ${RUN_LIME}." >&2
  exit 1
fi

for mode in train test; do
  if [[ "${mode}" == "train" ]]; then
    output_name="train"
    expected_samples=168
  else
    output_name="valid"
    expected_samples=36
  fi
  BASE_MODEL="${MODEL}" DISTILLED_ADAPTER_ROOT="${DISTILLED_ADAPTER_ROOT}" \
  DISTILLED_OUTPUT_DIR="${OUTPUT_ROOT}/case_study/${output_name}_epochs" \
  DISTILLED_ENSEMBLE_OUTPUT="${OUTPUT_ROOT}/case_study/inference_${output_name}" \
  DISTILLED_EPOCHS="0 1" DISTILLED_MODE="${mode}" \
  DISTILLED_CITY_LIST="${CASE_CITY_LIST}" \
  DISTILLED_EXPECTED_CITIES="${CASE_EXPECTED_CITIES}" \
  DISTILLED_EXPECTED_SAMPLES_PER_CITY="${expected_samples}" \
  DISTILLED_BATCH_SIZE="${INFERENCE_BATCH_SIZE}" \
  DISTILLED_MAX_LENGTH="${INFERENCE_MAX_LENGTH}" \
  DISTILLED_INVALID_POLICY="${INVALID_POLICY}" \
  DISTILLED_LOG_PREFIX="case_study_${output_name}" \
    bash scripts/run_distilled_ensemble_inference.sh
done

if [[ "${RUN_LIME}" == "1" ]]; then
  run_logged case_study_lime env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON}" src/Lime_explanation.py \
    --device cuda \
    --model "${MODEL}_PPO_epoch1_merged" \
    --adapter_path explanation \
    --adapter_dir "${DISTILLED_ADAPTER_ROOT}/epoch1" \
    --checkpoint distilled_epoch1 \
    --dataset_path "${DATA_DIR}" \
    --city_ids "${CASE_CITY_IDS[@]}" \
    --save_path "${OUTPUT_ROOT}/lime_explanation" \
    --max_length "${INFERENCE_MAX_LENGTH}" \
    --num_samples "${LIME_NUM_SAMPLES}"

  run_logged build_case_study_lime_figure_data "${PYTHON}" \
    scripts/build_lime_figure_data.py \
    --lime-root "${OUTPUT_ROOT}/lime_explanation" \
    --sample-index 6 \
    --output "${OUTPUT_ROOT}/figures/data/fig5c_lime_importance.csv"
fi

echo "Case-study figure inputs are ready below ${OUTPUT_ROOT}/case_study and ${OUTPUT_ROOT}/lime_explanation."
