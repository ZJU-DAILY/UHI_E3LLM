#!/usr/bin/env bash
# Regenerate the manuscript statistical figures from the workflow outputs.
# Panels whose inputs are not present yet are skipped with a message.
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

FIGURE_DATA_DIR="${FIGURE_DATA_DIR:-${OUTPUT_ROOT}/figures/data}"
FIGURE_DIR="${FIGURE_DIR:-${OUTPUT_ROOT}/figures}"
FIG4C_RAW_DATA_DIR="${FIG4C_RAW_DATA_DIR:-${RAW_DATA_DIR:-${DATA_DIR}/raw}}"
FIG4C_SOURCE="${FIGURE_DATA_DIR}/fig4c_factor_evolution_5cities.csv"
mkdir -p "${FIGURE_DIR}" "${FIGURE_DATA_DIR}"

step() {
  local name="$1"; shift
  if run_logged "figures_${name}" "${PYTHON}" "$@"; then
    echo "[done] ${name}"
    return 0
  else
    echo "[skip] ${name}: inputs missing or step failed (see logs/figures_${name}.log)" >&2
    return 1
  fi
}

step fig2 scripts/figures/plot_fig2_dataset.py \
  --city-metrics-dir "${DATA_DIR}/city_metrics" --out-dir "${FIGURE_DIR}" || true

step fig2h scripts/figures/plot_fig2h_factor_evidence.py \
  --out-dir "${FIGURE_DIR}" || true

if step fig4c_source scripts/figures/build_fig4c_source.py \
  --raw-data "${FIG4C_RAW_DATA_DIR}" \
  --city-ids 864,677,222,661,413 \
  --output "${FIG4C_SOURCE}"; then
  step fig4 scripts/figures/plot_fig4_case_study.py \
    --data-dir "${DATA_DIR}" --out-dir "${FIGURE_DIR}" \
    --fig4c-source "${FIG4C_SOURCE}" || true
else
  echo "[skip] fig4: five-city factor source could not be generated" >&2
fi

step fig5 scripts/figures/plot_fig5_case_study.py \
  --predictions "${OUTPUT_ROOT}/case_study/inference_train" \
                "${OUTPUT_ROOT}/case_study/inference_valid" \
  --out-dir "${FIGURE_DIR}" || true

step fig5c scripts/figures/plot_fig5c_lime.py \
  --input "${FIGURE_DATA_DIR}/fig5c_lime_importance.csv" \
  --out-dir "${FIGURE_DIR}" || true

if step collect scripts/figures/figure_data.py \
  --output-root "${OUTPUT_ROOT}" --figure-data-dir "${FIGURE_DATA_DIR}" \
  --expected-cities 100 --expected-samples-per-city 36; then
  if step full_period scripts/prepare_full_period_figure_data.py \
    --data-dir "${DATA_DIR}" --output-root "${OUTPUT_ROOT}" \
    --output "${FIGURE_DATA_DIR}/method_city_year_metrics_2004_2020.csv" \
    --gpu-id "${GPU_ID}"; then
    step fig3 scripts/figures/plot_fig3_performance.py \
      --figure-data-dir "${FIGURE_DATA_DIR}" --out-dir "${FIGURE_DIR}" \
      --city-metrics "${DATA_DIR}/city_metrics/city_metrics_test.csv" \
      --full-period-city-year "${FIGURE_DATA_DIR}/method_city_year_metrics_2004_2020.csv" || true
  else
    echo "[skip] fig3: full-period source could not be prepared" >&2
  fi
else
  echo "[skip] fig3: current eight-method prediction contract did not validate" >&2
fi

step fig7ab scripts/figures/plot_fig7_uncertainty.py \
  --output-root "${OUTPUT_ROOT}" \
  --figure-data-dir "${FIGURE_DATA_DIR}" || true

step fig7cd scripts/figures/plot_fig7_efficiency.py \
  --measurements "${OUTPUT_ROOT}/efficiency/efficiency_measurements.csv" \
  --out-dir "${FIGURE_DIR}" || true

step fig7e scripts/figures/plot_fig7_lambda.py \
  --summary "${OUTPUT_ROOT}/sensitivity/ppo_lambda_summary.csv" \
  --out-dir "${FIGURE_DIR}" || true

echo "Figures below ${FIGURE_DIR}"
