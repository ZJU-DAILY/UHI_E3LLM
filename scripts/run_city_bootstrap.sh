#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

declare -A PREDICTIONS=(
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
  path="${PREDICTIONS[$method]}"
  if [[ ! -e "${path}" ]]; then
    echo "Missing predictions for ${method}: ${path}" >&2
    exit 1
  fi
  safe_name="${method//[^A-Za-z0-9]/_}"
  run_logged "city_bootstrap_${safe_name}" "${PYTHON}" scripts/city_bootstrap.py \
    --predictions "${path}" \
    --output_dir "${OUTPUT_ROOT}/uncertainty/bootstrap/${safe_name}" \
    --replicates 1000 \
    --seed 42 \
    --expected_cities 100 \
    --expected_samples_per_city 36
done
