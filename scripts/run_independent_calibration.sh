#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_baseline_workflows.bash"
cd "${ROOT}"

MODEL="${MODEL:-deepseek-8b_merged__Global500}"
DISTILLED_ADAPTER_ROOT="${DISTILLED_ADAPTER_ROOT:-${ROOT}/model_adapters/explanation_distilled/${MODEL}}"
CALIBRATION_EXPECTED_CITIES="${CALIBRATION_EXPECTED_CITIES:-100}"
INVALID_POLICY="${INVALID_POLICY:-error}"
CALIBRATION_DATASET="${OUTPUT_ROOT}/calibration/dataset"
require_global_baseline_artifacts

run_logged prepare_calibration_dataset "${PYTHON}" scripts/prepare_calibration_dataset.py \
  --data_dir "${DATA_DIR}" \
  --output_dir "${CALIBRATION_DATASET}" \
  --expected_cities "${CALIBRATION_EXPECTED_CITIES}"

DATA_DIR="${CALIBRATION_DATASET}" GPU_ID="${GPU_ID}" \
BASE_MODEL="${MODEL}" DISTILLED_ADAPTER_ROOT="${DISTILLED_ADAPTER_ROOT}" \
DISTILLED_OUTPUT_DIR="${OUTPUT_ROOT}/calibration/distilled_epochs" \
DISTILLED_ENSEMBLE_OUTPUT="${OUTPUT_ROOT}/calibration/ensemble" \
DISTILLED_EPOCHS="0 1" DISTILLED_MODE=test \
DISTILLED_EXPECTED_CITIES="${CALIBRATION_EXPECTED_CITIES}" \
DISTILLED_EXPECTED_SAMPLES_PER_CITY=36 \
DISTILLED_INVALID_POLICY="${INVALID_POLICY}" \
DISTILLED_LOG_PREFIX=calibration \
  bash "${ROOT}/scripts/run_distilled_ensemble_inference.sh"

run_logged calibration_tabular_inference env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON}" baseline/tabular_inference.py \
  --dataset_path "${CALIBRATION_DATASET}" \
  --model_dir "${OUTPUT_ROOT}/baselines/tabular/models" \
  --output_dir "${OUTPUT_ROOT}/calibration/baselines/tabular" \
  --models xgb lgb rf mlp \
  --xgb_device "${XGB_DEVICE:-cuda}" \
  --torch_device "${TORCH_DEVICE:-cuda:0}" \
  --expected_cities "${CALIBRATION_EXPECTED_CITIES}" \
  --expected_samples_per_city 36

REUSE_TIMESERIES_DATA=0 run_calibration_timeseries_inference \
  "${CALIBRATION_DATASET}/city_metrics/city_metrics_test.csv"

declare -A CALIBRATION_PREDICTIONS=(
  [UHI-E3LLM]="${OUTPUT_ROOT}/calibration/ensemble/city_results"
  [XGBoost]="${OUTPUT_ROOT}/calibration/baselines/tabular/predictions/xgb_test_predictions.csv"
  [RandomForest]="${OUTPUT_ROOT}/calibration/baselines/tabular/predictions/rf_test_predictions.csv"
  [LightGBM]="${OUTPUT_ROOT}/calibration/baselines/tabular/predictions/lgb_test_predictions.csv"
  [MLP]="${OUTPUT_ROOT}/calibration/baselines/tabular/predictions/mlp_test_predictions.csv"
  [ARMD]="${OUTPUT_ROOT}/regional/Calibration/baselines/armd/sample_errors"
  [FilterNet]="${OUTPUT_ROOT}/regional/Calibration/baselines/filternet/sample_errors"
  [ST-MTM]="${OUTPUT_ROOT}/regional/Calibration/baselines/st_mtm/sample_errors"
)
declare -A TEST_PREDICTIONS=(
  [UHI-E3LLM]="${OUTPUT_ROOT}/main_ensemble/ensemble/city_results"
  [XGBoost]="${OUTPUT_ROOT}/baselines/tabular/predictions/xgb_test_predictions.csv"
  [RandomForest]="${OUTPUT_ROOT}/baselines/tabular/predictions/rf_test_predictions.csv"
  [LightGBM]="${OUTPUT_ROOT}/baselines/tabular/predictions/lgb_test_predictions.csv"
  [MLP]="${OUTPUT_ROOT}/baselines/tabular/predictions/mlp_test_predictions.csv"
  [ARMD]="${OUTPUT_ROOT}/baselines/armd/sample_errors"
  [FilterNet]="${OUTPUT_ROOT}/baselines/filternet/sample_errors"
  [ST-MTM]="${OUTPUT_ROOT}/baselines/st_mtm/sample_errors"
)

for method in UHI-E3LLM XGBoost RandomForest LightGBM MLP ARMD FilterNet ST-MTM; do
  safe_name="${method//[^A-Za-z0-9]/_}"
  run_logged "independent_calibration_${safe_name}" \
    "${PYTHON}" scripts/calibrate_prediction_intervals.py \
    --calibration_predictions "${CALIBRATION_PREDICTIONS[$method]}" \
    --test_predictions "${TEST_PREDICTIONS[$method]}" \
    --output_dir "${OUTPUT_ROOT}/uncertainty/calibration/${safe_name}" \
    --quantile 0.90 \
    --expected_calibration_cities "${CALIBRATION_EXPECTED_CITIES}" \
    --expected_test_cities 100 \
    --expected_samples_per_city 36 \
    --calibration_city_manifest "${DATA_DIR}/city_metrics/calibration_cities.csv" \
    --test_city_manifest "${DATA_DIR}/city_metrics/city_metrics_test.csv"
done
