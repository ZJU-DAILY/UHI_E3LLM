#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_baseline_workflows.bash"
cd "${ROOT}"

REGION_LIST="${REGION_LIST:-Africa EastAsiaOceania Europe LatinAmerica NorthAmerica SCWAsia}"
read -r -a REGIONS <<< "${REGION_LIST}"
MODEL="${MODEL:-deepseek-8b}"
SFT_EPOCHS="${SFT_EPOCHS:-5}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-16}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
REGIONAL_OVERWRITE="${REGIONAL_OVERWRITE:-0}"

if [[ "${REGIONAL_OVERWRITE}" != "0" && "${REGIONAL_OVERWRITE}" != "1" ]]; then
  echo "REGIONAL_OVERWRITE must be 0 or 1; got ${REGIONAL_OVERWRITE}." >&2
  exit 1
fi

for region in "${REGIONS[@]}"; do
  echo "=== ${region}: UHI-E3LLM (always evaluated before baselines) ==="
  train_manifest="${DATA_DIR}/regional_folds/fold_${region}_train.csv"
  test_manifest="${DATA_DIR}/regional_folds/fold_${region}_test.csv"
  read -r train_cities expected_cities < <("${PYTHON}" \
    scripts/validate_regional_fold.py \
    --region "${region}" \
    --train "${train_manifest}" \
    --test "${test_manifest}" \
    --assignments "${DATA_DIR}/city_metrics/region_assignments_600.csv")

  fold_dir="${OUTPUT_ROOT}/regional/${region}/dataset"
  mkdir -p "${fold_dir}/city_metrics"
  ln -sfn "${DATA_DIR}/train" "${fold_dir}/train"
  ln -sfn "${DATA_DIR}/valid" "${fold_dir}/valid"
  cp "${train_manifest}" \
    "${fold_dir}/city_metrics/city_metrics_train.csv"
  cp "${test_manifest}" \
    "${fold_dir}/city_metrics/city_metrics_test.csv"

  region_adapter_root="${ROOT}/model_adapters/regional/${region}"
  sft_root="${region_adapter_root}/sft"
  fusion_adapter="${region_adapter_root}/fusion"
  ppo_dir="${region_adapter_root}/ppo"
  regional_model="${MODEL}_regional_${region}_merged"

  DATA_DIR="${fold_dir}" GPU_ID="${GPU_ID}" MODEL="${MODEL}" \
  SFT_OUTPUT="${sft_root}" SFT_EPOCHS="${SFT_EPOCHS}" \
  SFT_BATCH_SIZE="${SFT_BATCH_SIZE}" SFT_EXPECTED_CITIES="${train_cities}" \
  SFT_EXPECTED_SAMPLES_PER_CITY=168 SFT_INPUT_FORMAT=prompt \
  SFT_OVERWRITE_OUTPUT="${REGIONAL_OVERWRITE}" \
  SFT_RUN_NAME="sft_region_${region}" \
    bash "${ROOT}/scripts/run_sft.sh"

  DATA_DIR="${fold_dir}" GPU_ID="${GPU_ID}" MODEL="${MODEL}" \
  SFT_ADAPTER_ROOT="${sft_root}" OUTPUT_DIR="${fusion_adapter}" \
  MERGED_MODEL_DIR="${ROOT}/model/${regional_model}" \
  FUSION_REPORT_DIR="${OUTPUT_ROOT}/regional/${region}/lora_fusion" \
  FUSION_EXPECTED_VALIDATION_CITIES="${train_cities}" \
  FUSION_MODE=greedy FUSION_INPUT_FORMAT=prompt \
  FUSION_OVERWRITE="${REGIONAL_OVERWRITE}" \
  FUSION_RUN_NAME="fusion_region_${region}" \
    bash "${ROOT}/scripts/run_lora_fusion.sh"

  DATA_DIR="${fold_dir}" GPU_ID="${GPU_ID}" MODEL="${regional_model}" \
  PPO_OUTPUT="${ppo_dir}" PPO_EPOCHS=2 PPO_BATCH_SIZE="${PPO_BATCH_SIZE}" \
  PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" PPO_BASELINE_MODE=train \
  PPO_EXPECTED_TRAIN_CITIES="${train_cities}" \
  PPO_EXPECTED_BASELINE_CITIES="${train_cities}" \
  PPO_EXPECTED_SAMPLES_PER_CITY=168 \
  PPO_EXPECTED_BASELINE_SAMPLES_PER_CITY=168 \
  PPO_BASELINE_OUTPUT="${OUTPUT_ROOT}/regional/${region}/ppo_train_baseline" \
  PPO_INPUT_FORMAT=prompt_task_only PPO_OVERWRITE="${REGIONAL_OVERWRITE}" \
  PPO_LOG_PREFIX="region_${region}" \
    bash "${ROOT}/scripts/run_ppo.sh"

  regional_explanation_root="${region_adapter_root}/explanation_distilled/${regional_model}"
  regional_explanation_data="${OUTPUT_ROOT}/regional/${region}/explanation_samples"
  for policy_epoch in 0 1; do
    regional_distill_model="${regional_model}_PPO_epoch${policy_epoch}_merged"
    DATA_DIR="${fold_dir}" GPU_ID="${GPU_ID}" PYTHON="${PYTHON}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_DIR="${LOG_DIR}" \
    BASE_MODEL="${regional_model}" \
    EXPLANATION_POLICY_ADAPTER_DIR="${ppo_dir}/epoch${policy_epoch}" \
    EXPLANATION_POLICY_EPOCH="${policy_epoch}" \
    MERGED_MODEL="${regional_distill_model}" \
    MERGED_MODEL_DIR="${ROOT}/model/${regional_distill_model}" \
    EXPLANATION_TRAIN_CITY_METRICS="${train_manifest}" \
    EXPLANATION_EXPECTED_TRAIN_CITIES="${train_cities}" \
    EXPLANATION_EXPECTED_TRAIN_SAMPLES_PER_CITY=168 \
    EXPLANATION_DATA_ROOT="${regional_explanation_data}" \
    EXPLANATION_INPUT_FORMAT=prompt \
    EXPLANATION_ADAPTER="regional_${region}_epoch${policy_epoch}_explanation" \
    EXPLANATION_ADAPTER_DIR="${regional_explanation_root}/epoch${policy_epoch}" \
    EXPLANATION_LOG_PREFIX="region_${region}_epoch${policy_epoch}" \
    EXPLANATION_RUN_INFERENCE=0 \
    INVALID_POLICY=error \
      bash "${ROOT}/scripts/run_explanation_distillation.sh"
  done

  DATA_DIR="${fold_dir}" GPU_ID="${GPU_ID}" \
  BASE_MODEL="${regional_model}" \
  DISTILLED_ADAPTER_ROOT="${regional_explanation_root}" \
  DISTILLED_OUTPUT_DIR="${OUTPUT_ROOT}/regional/${region}" \
  DISTILLED_EPOCHS="0 1" DISTILLED_INPUT_FORMAT=prompt \
  DISTILLED_EXPECTED_CITIES="${expected_cities}" \
  DISTILLED_EXPECTED_SAMPLES_PER_CITY=36 DISTILLED_INVALID_POLICY=error \
  DISTILLED_LOG_PREFIX="region_${region}" \
    bash "${ROOT}/scripts/run_distilled_ensemble_inference.sh"

  echo "=== ${region}: all seven baselines ==="
  run_all_regional_baselines \
    "${region}" \
    "${train_manifest}" \
    "${test_manifest}"
done

run_logged build_leave_one_region_out_table "${PYTHON}" \
  scripts/build_comparison_tables.py regional \
  --data-dir "${DATA_DIR}" \
  --output-root "${OUTPUT_ROOT}"

echo "Table 2 output: ${OUTPUT_ROOT}/tables/table2_leave_one_region_out.csv"
