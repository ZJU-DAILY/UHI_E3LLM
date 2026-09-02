#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

MODEL="${MODEL:-deepseek-8b}"
SFT_ADAPTER_ROOT="${SFT_ADAPTER_ROOT:-${ROOT}/model_adapters/SFT/${MODEL}}"
FUSION_EPOCHS="${FUSION_EPOCHS:-0 1 2 3 4}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/model_adapters/merged/${MODEL}}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-${ROOT}/model/deepseek-8b_merged__Global500}"
FUSION_REPORT_DIR="${FUSION_REPORT_DIR:-${OUTPUT_ROOT}/lora_fusion}"
FUSION_MODE="${FUSION_MODE:-greedy}"
FUSION_FIXED_WEIGHTS="${FUSION_FIXED_WEIGHTS:-}"
FUSION_SIMILARITY_THRESHOLD="${FUSION_SIMILARITY_THRESHOLD:-0.6}"
FUSION_BETA="${FUSION_BETA:-10.0}"
FUSION_MIN_MAE_IMPROVEMENT="${FUSION_MIN_MAE_IMPROVEMENT:-0.0}"
FUSION_BATCH_SIZE="${FUSION_BATCH_SIZE:-24}"
FUSION_MAX_LENGTH="${FUSION_MAX_LENGTH:-1300}"
FUSION_MAX_NEW_TOKENS="${FUSION_MAX_NEW_TOKENS:-19}"
FUSION_NUM_WORKERS="${FUSION_NUM_WORKERS:-0}"
FUSION_VALIDATION_CITY_LIMIT="${FUSION_VALIDATION_CITY_LIMIT:-0}"
FUSION_VALID_MIN="${FUSION_VALID_MIN:--5.0}"
FUSION_VALID_MAX="${FUSION_VALID_MAX:-5.0}"
FUSION_MAX_DECIMAL_PLACES="${FUSION_MAX_DECIMAL_PLACES:-3}"
FUSION_SEED="${FUSION_SEED:-42}"
FUSION_OVERWRITE="${FUSION_OVERWRITE:-0}"
FUSION_INPUT_FORMAT="${FUSION_INPUT_FORMAT:-prompt}"
FUSION_RUN_NAME="${FUSION_RUN_NAME:-main_lora_fusion}"
FUSION_EXPECTED_VALIDATION_CITIES="${FUSION_EXPECTED_VALIDATION_CITIES:-500}"
FUSION_EXPECTED_TRANSFORMERS_VERSION="${FUSION_EXPECTED_TRANSFORMERS_VERSION:-4.46.3}"
FUSION_EXPECTED_TRL_VERSION="${FUSION_EXPECTED_TRL_VERSION:-0.11.2}"
FUSION_EXPECTED_PEFT_VERSION="${FUSION_EXPECTED_PEFT_VERSION:-0.19.1}"

if [[ "${FUSION_MODE}" != "greedy" && "${FUSION_MODE}" != "fixed" ]]; then
  echo "FUSION_MODE must be greedy or fixed; got ${FUSION_MODE}." >&2
  exit 1
fi
if [[ "${FUSION_INPUT_FORMAT}" != "prompt" && "${FUSION_INPUT_FORMAT}" != "numeric_plain" ]]; then
  echo "FUSION_INPUT_FORMAT must be prompt or numeric_plain; got ${FUSION_INPUT_FORMAT}." >&2
  exit 1
fi
if [[ "${FUSION_OVERWRITE}" != "0" && "${FUSION_OVERWRITE}" != "1" ]]; then
  echo "FUSION_OVERWRITE must be 0 or 1; got ${FUSION_OVERWRITE}." >&2
  exit 1
fi
if [[ "${MODEL}" = /* ]]; then
  model_path="$(realpath -m "${MODEL}")"
else
  model_path="${ROOT}/model/${MODEL}"
fi
if [[ ! -s "${model_path}/config.json" ]]; then
  echo "Base model config does not exist: ${model_path}/config.json" >&2
  exit 1
fi

if [[ "${FUSION_MODE}" == "greedy" ]]; then
  for required_path in \
    "${DATA_DIR}/city_metrics/city_metrics_train.csv" \
    "${DATA_DIR}/city_metrics/city_metrics_test.csv"; do
    if [[ ! -s "${required_path}" ]]; then
      echo "Required city manifest does not exist: ${required_path}" >&2
      exit 1
    fi
  done
  if [[ ! -d "${DATA_DIR}/valid" ]]; then
    echo "Validation dataset directory does not exist: ${DATA_DIR}/valid" >&2
    exit 1
  fi
fi

if ! runtime_versions="$("${PYTHON}" -c \
  'import peft, transformers, trl; print(transformers.__version__, trl.__version__, peft.__version__)')"; then
  echo "Unable to import transformers, TRL, and PEFT. Activate uhi-e3llm first." >&2
  exit 1
fi
read -r transformers_version trl_version peft_version <<< "${runtime_versions}"
if [[ "${transformers_version}" != "${FUSION_EXPECTED_TRANSFORMERS_VERSION}" ||
      "${trl_version}" != "${FUSION_EXPECTED_TRL_VERSION}" ||
      "${peft_version}" != "${FUSION_EXPECTED_PEFT_VERSION}" ]]; then
  echo "Expected transformers=${FUSION_EXPECTED_TRANSFORMERS_VERSION}, trl=${FUSION_EXPECTED_TRL_VERSION}, and peft=${FUSION_EXPECTED_PEFT_VERSION}; got transformers=${transformers_version}, trl=${trl_version}, and peft=${peft_version}." >&2
  exit 1
fi
echo "LoRA-fusion runtime: transformers=${transformers_version}, trl=${trl_version}, peft=${peft_version}"

read -r -a epoch_indices <<< "${FUSION_EPOCHS}"
adapter_paths=()
for epoch_index in "${epoch_indices[@]}"; do
  if [[ ! "${epoch_index}" =~ ^[0-9]+$ ]]; then
    echo "Invalid zero-based epoch index: ${epoch_index}" >&2
    exit 1
  fi
  adapter_path="${SFT_ADAPTER_ROOT}/checkpoint-epoch${epoch_index}"
  if [[ ! -s "${adapter_path}/adapter_config.json" || ! -s "${adapter_path}/adapter_model.safetensors" ]]; then
    echo "Incomplete SFT adapter: ${adapter_path}" >&2
    exit 1
  fi
  adapter_paths+=("${adapter_path}")
done

output_adapter_path="$(realpath -m "${OUTPUT_DIR}")"
merged_model_path="$(realpath -m "${MERGED_MODEL_DIR}")"
report_path="$(realpath -m "${FUSION_REPORT_DIR}")"
sft_adapter_root_path="$(realpath -m "${SFT_ADAPTER_ROOT}")"
data_path="$(realpath -m "${DATA_DIR}")"

paths_overlap() {
  local first="$1"
  local second="$2"
  [[ "${first}" == "${second}" ||
     "${first}" == "${second}/"* ||
     "${second}" == "${first}/"* ]]
}

for output_path in "${output_adapter_path}" "${merged_model_path}" "${report_path}"; do
  if [[ "${output_path}" == "/" || "${output_path}" == "${ROOT}" ]]; then
    echo "Refusing unsafe fusion output path: ${output_path}" >&2
    exit 1
  fi
  for input_path in "${model_path}" "${sft_adapter_root_path}" "${data_path}"; do
    if paths_overlap "${output_path}" "${input_path}"; then
      echo "Fusion output overlaps an input path: ${output_path} and ${input_path}" >&2
      exit 1
    fi
  done
done
if paths_overlap "${output_adapter_path}" "${merged_model_path}" ||
   paths_overlap "${output_adapter_path}" "${report_path}" ||
   paths_overlap "${merged_model_path}" "${report_path}"; then
  echo "Fusion adapter, full-model, and report output directories must not overlap." >&2
  exit 1
fi

command=(env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" \
  src/greedy_adapter_fusion.py \
  --model "${MODEL}" \
  --adapter_paths "${adapter_paths[@]}" \
  --dataset_path "${DATA_DIR}" \
  --output_adapter_dir "${OUTPUT_DIR}" \
  --merged_model_dir "${MERGED_MODEL_DIR}" \
  --report_dir "${FUSION_REPORT_DIR}" \
  --merged_name manuscript_fusion \
  --selection_mode "${FUSION_MODE}" \
  --similarity_threshold "${FUSION_SIMILARITY_THRESHOLD}" \
  --beta "${FUSION_BETA}" \
  --min_mae_improvement "${FUSION_MIN_MAE_IMPROVEMENT}" \
  --batch_size "${FUSION_BATCH_SIZE}" \
  --max_length "${FUSION_MAX_LENGTH}" \
  --max_new_tokens "${FUSION_MAX_NEW_TOKENS}" \
  --num_workers "${FUSION_NUM_WORKERS}" \
  --validation_city_limit "${FUSION_VALIDATION_CITY_LIMIT}" \
  --expected_validation_cities "${FUSION_EXPECTED_VALIDATION_CITIES}" \
  --expected_samples_per_city 36 \
  --valid_min "${FUSION_VALID_MIN}" \
  --valid_max "${FUSION_VALID_MAX}" \
  --max_decimal_places "${FUSION_MAX_DECIMAL_PLACES}" \
  --input_format "${FUSION_INPUT_FORMAT}" \
  --seed "${FUSION_SEED}" \
  --device cuda)

if [[ "${FUSION_MODE}" == "fixed" ]]; then
  if [[ -z "${FUSION_FIXED_WEIGHTS}" ]]; then
    echo "FUSION_MODE=fixed requires FUSION_FIXED_WEIGHTS." >&2
    exit 1
  fi
  read -r -a fixed_weights <<< "${FUSION_FIXED_WEIGHTS}"
  command+=(--fixed_weights "${fixed_weights[@]}")
elif [[ -n "${FUSION_FIXED_WEIGHTS}" ]]; then
  echo "FUSION_FIXED_WEIGHTS can only be used with FUSION_MODE=fixed." >&2
  exit 1
fi

if [[ "${FUSION_OVERWRITE}" == "1" ]]; then
  command+=(--overwrite)
fi

run_logged "${FUSION_RUN_NAME}" "${command[@]}"
