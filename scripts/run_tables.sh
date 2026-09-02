#!/usr/bin/env bash
# Recompute manuscript Tables 1-3 from completed per-sample experiment outputs.
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

target="${1:-all}"
case "${target}" in
  table1) mode=main ;;
  table2) mode=regional ;;
  table3) mode=ablation ;;
  all) mode=all ;;
  *)
    echo "Usage: bash scripts/run_tables.sh [table1|table2|table3|all]" >&2
    exit 2
    ;;
esac

run_logged "build_${target}" "${PYTHON}" scripts/build_comparison_tables.py \
  "${mode}" \
  --data-dir "${DATA_DIR}" \
  --output-root "${OUTPUT_ROOT}"

echo "Manuscript table outputs: ${OUTPUT_ROOT}/tables/"
