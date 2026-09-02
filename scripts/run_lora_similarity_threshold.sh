#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

MODEL="${MODEL:-deepseek-8b}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${ROOT}/model_adapters/SFT/${MODEL}}"
SIMILARITY_EPOCHS="${SIMILARITY_EPOCHS:-0 1 2 3 4}"
SIMILARITY_THRESHOLDS="${SIMILARITY_THRESHOLDS:-0.4 0.5 0.6 0.7 0.8}"
SIMILARITY_OUTPUT_DIR="${SIMILARITY_OUTPUT_DIR:-${OUTPUT_ROOT}/sensitivity/lora_similarity}"
SIMILARITY_OVERWRITE="${SIMILARITY_OVERWRITE:-0}"
SIMILARITY_EXPECTED_TRANSFORMERS_VERSION="${SIMILARITY_EXPECTED_TRANSFORMERS_VERSION:-4.46.3}"
SIMILARITY_EXPECTED_TRL_VERSION="${SIMILARITY_EXPECTED_TRL_VERSION:-0.11.2}"
SIMILARITY_EXPECTED_PEFT_VERSION="${SIMILARITY_EXPECTED_PEFT_VERSION:-0.19.1}"

if [[ "${SIMILARITY_OVERWRITE}" != "0" && "${SIMILARITY_OVERWRITE}" != "1" ]]; then
  echo "SIMILARITY_OVERWRITE must be 0 or 1; got ${SIMILARITY_OVERWRITE}." >&2
  exit 1
fi

if ! runtime_versions="$("${PYTHON}" -c \
  'import peft, transformers, trl; print(transformers.__version__, trl.__version__, peft.__version__)')"; then
  echo "Unable to import transformers, TRL, and PEFT. Activate uhi-e3llm first." >&2
  exit 1
fi
read -r transformers_version trl_version peft_version <<< "${runtime_versions}"
if [[ "${transformers_version}" != "${SIMILARITY_EXPECTED_TRANSFORMERS_VERSION}" ||
      "${trl_version}" != "${SIMILARITY_EXPECTED_TRL_VERSION}" ||
      "${peft_version}" != "${SIMILARITY_EXPECTED_PEFT_VERSION}" ]]; then
  echo "Expected transformers=${SIMILARITY_EXPECTED_TRANSFORMERS_VERSION}, trl=${SIMILARITY_EXPECTED_TRL_VERSION}, and peft=${SIMILARITY_EXPECTED_PEFT_VERSION}; got transformers=${transformers_version}, trl=${trl_version}, and peft=${peft_version}." >&2
  exit 1
fi

read -r -a epoch_indices <<< "${SIMILARITY_EPOCHS}"
if (( ${#epoch_indices[@]} < 2 )); then
  echo "SIMILARITY_EPOCHS must identify at least two SFT checkpoints." >&2
  exit 1
fi
adapters=()
for epoch_index in "${epoch_indices[@]}"; do
  if [[ ! "${epoch_index}" =~ ^[0-9]+$ ]]; then
    echo "Invalid zero-based epoch index: ${epoch_index}" >&2
    exit 1
  fi
  adapter_path="${ADAPTER_ROOT}/checkpoint-epoch${epoch_index}"
  if [[ ! -s "${adapter_path}/adapter_config.json" || ! -s "${adapter_path}/adapter_model.safetensors" ]]; then
    echo "Incomplete SFT adapter: ${adapter_path}" >&2
    exit 1
  fi
  adapters+=("${adapter_path}")
done

read -r -a thresholds <<< "${SIMILARITY_THRESHOLDS}"
if (( ${#thresholds[@]} == 0 )); then
  echo "SIMILARITY_THRESHOLDS must not be empty." >&2
  exit 1
fi

command=("${PYTHON}" scripts/lora_similarity_threshold.py
  --adapters "${adapters[@]}"
  --thresholds "${thresholds[@]}"
  --output_dir "${SIMILARITY_OUTPUT_DIR}")
if [[ "${SIMILARITY_OVERWRITE}" == "1" ]]; then
  command+=(--overwrite)
fi

run_logged lora_similarity_threshold "${command[@]}"
