#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-${ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
GPU_ID="${GPU_ID:-0}"
EFFICIENCY_RECORD_DIR="${EFFICIENCY_RECORD_DIR:-${OUTPUT_ROOT}/efficiency/raw}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${EFFICIENCY_RECORD_DIR}"

run_logged() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  local status=0
  {
    echo "[$(date -Iseconds)] START ${name}"
    echo "root=${ROOT}"
    echo "data_dir=${DATA_DIR}"
    echo "gpu_id=${GPU_ID}"
    echo "python=$(${PYTHON} --version 2>&1)"
    echo "command=$*"
    "$@" && status=0 || status=$?
    if [[ "${status}" -eq 0 ]]; then
      echo "[$(date -Iseconds)] DONE ${name}"
    else
      echo "[$(date -Iseconds)] FAILED ${name} (exit ${status})"
    fi
  } > >(tee "${log_file}") 2>&1
  wait $! 2>/dev/null || true
  return "${status}"
}
