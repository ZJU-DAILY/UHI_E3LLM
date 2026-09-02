#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_baseline_workflows.bash"
cd "${ROOT}"

# This is the sole public entry for the complete test-set comparison. The
# proposed method is deliberately evaluated before any baseline is launched.
MAIN_MODEL="${MAIN_MODEL:-deepseek-8b_merged__Global500}"
MAIN_DISTILLED_ADAPTER_ROOT="${MAIN_DISTILLED_ADAPTER_ROOT:-${ROOT}/model_adapters/explanation_distilled/${MAIN_MODEL}}"

if ! runtime_versions="$("${PYTHON}" -c \
  'import datasets, einops, lightgbm, pandas, peft, scipy, sklearn, torch, transformers, trl, xgboost; print(transformers.__version__, trl.__version__, peft.__version__)')"; then
  echo "Unable to import the main-model and baseline dependencies." >&2
  echo "Activate the uhi-e3llm environment before running this workflow." >&2
  exit 1
fi
read -r transformers_version trl_version peft_version <<< "${runtime_versions}"
if [[ "${transformers_version}" != "4.46.3" || "${trl_version}" != "0.11.2" || "${peft_version}" != "0.19.1" ]]; then
  echo "Expected transformers=4.46.3, trl=0.11.2, and peft=0.19.1; got transformers=${transformers_version}, trl=${trl_version}, and peft=${peft_version}." >&2
  exit 1
fi

echo "[1/4] Evaluating UHI-E3LLM (explanation-distilled epoch0 + epoch1 ensemble)"
BASE_MODEL="${MAIN_MODEL}" \
DISTILLED_ADAPTER_ROOT="${MAIN_DISTILLED_ADAPTER_ROOT}" \
DISTILLED_OUTPUT_DIR="${OUTPUT_ROOT}/main_ensemble" \
DISTILLED_LOG_PREFIX=main_table \
DISTILLED_EPOCHS="0 1" \
DISTILLED_INPUT_FORMAT=prompt \
DISTILLED_EXPECTED_CITIES=100 \
DISTILLED_EXPECTED_SAMPLES_PER_CITY=36 \
DISTILLED_INVALID_POLICY=error \
DISTILLED_EFFICIENCY_RECORD="${EFFICIENCY_RECORD_DIR}/inference_uhi_e3llm.json" \
  bash "${ROOT}/scripts/run_distilled_ensemble_inference.sh"

echo "[2/4] Evaluating tabular baselines"
run_global_tabular_baselines

echo "[3/4] Evaluating time-series baselines"
run_global_timeseries_baselines

echo "[4/4] Validating predictions and building the comparison table"
run_logged collect_main_comparison_predictions "${PYTHON}" \
  "${ROOT}/scripts/figures/figure_data.py" \
  --output-root "${OUTPUT_ROOT}"
run_logged build_main_comparison_table "${PYTHON}" \
  "${ROOT}/scripts/build_comparison_tables.py" main \
  --data-dir "${DATA_DIR}" \
  --output-root "${OUTPUT_ROOT}"

echo "Table 1 output: ${OUTPUT_ROOT}/tables/table1_main_performance.csv"
