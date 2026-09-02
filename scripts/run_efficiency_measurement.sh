#!/usr/bin/env bash
# Measure the eight released methods and build the Fig. 7c/7d source table.
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

EFFICIENCY_RAW_DIR="${EFFICIENCY_RAW_DIR:-${OUTPUT_ROOT}/efficiency/raw}"
EFFICIENCY_OUTPUT="${EFFICIENCY_OUTPUT:-${OUTPUT_ROOT}/efficiency/efficiency_measurements.csv}"
EFFICIENCY_RUN_MAIN_EVALUATION="${EFFICIENCY_RUN_MAIN_EVALUATION:-1}"

if [[ "${EFFICIENCY_RUN_MAIN_EVALUATION}" != "0" && "${EFFICIENCY_RUN_MAIN_EVALUATION}" != "1" ]]; then
  echo "EFFICIENCY_RUN_MAIN_EVALUATION must be 0 or 1." >&2
  exit 1
fi

mkdir -p "${EFFICIENCY_RAW_DIR}" "$(dirname "${EFFICIENCY_OUTPUT}")"

for record in training_uhi_sft.json training_uhi_ppo.json; do
  if [[ ! -s "${EFFICIENCY_RAW_DIR}/${record}" ]]; then
    echo "Missing main-model training measurement: ${EFFICIENCY_RAW_DIR}/${record}" >&2
    echo "Run scripts/run_sft.sh and scripts/run_ppo.sh with the same OUTPUT_ROOT before this workflow." >&2
    exit 1
  fi
done

if [[ "${EFFICIENCY_RUN_MAIN_EVALUATION}" == "1" ]]; then
  echo "[1/2] Training/evaluating baselines and benchmarking all inference methods"
  # run_main_evaluation.sh writes baseline training records and the eight-method
  # inference records while producing the ordinary prediction artifacts.
  EFFICIENCY_RECORD_DIR="${EFFICIENCY_RAW_DIR}" \
    bash "${ROOT}/scripts/run_main_evaluation.sh"
else
  echo "[1/2] Reusing raw measurements already produced by the model workflows"
fi

echo "[2/2] Validating and aggregating workflow measurements"
run_logged build_efficiency_measurements "${PYTHON}" \
  "${ROOT}/scripts/build_efficiency_measurements.py" \
  --raw-dir "${EFFICIENCY_RAW_DIR}" \
  --output "${EFFICIENCY_OUTPUT}"

echo "Efficiency table: ${EFFICIENCY_OUTPUT}"
