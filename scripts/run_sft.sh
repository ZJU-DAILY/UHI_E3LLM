#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

MODEL="${MODEL:-deepseek-8b}"
SFT_OUTPUT="${SFT_OUTPUT:-${ROOT}/model_adapters/SFT/${MODEL}}"
SFT_EPOCHS="${SFT_EPOCHS:-5}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-16}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-1300}"
SFT_NUM_WORKERS="${SFT_NUM_WORKERS:-4}"
SFT_SEED="${SFT_SEED:-42}"
SFT_DATA_SEED="${SFT_DATA_SEED:-}"
SFT_FORCE_RETOKENIZE="${SFT_FORCE_RETOKENIZE:-0}"
SFT_RESUME_FROM_CHECKPOINT="${SFT_RESUME_FROM_CHECKPOINT:-}"
SFT_OVERWRITE_OUTPUT="${SFT_OVERWRITE_OUTPUT:-0}"
SFT_INPUT_FORMAT="${SFT_INPUT_FORMAT:-prompt}"
SFT_RUN_NAME="${SFT_RUN_NAME:-main_sft}"
SFT_EXPECTED_CITIES="${SFT_EXPECTED_CITIES:-500}"
SFT_EXPECTED_SAMPLES_PER_CITY="${SFT_EXPECTED_SAMPLES_PER_CITY:-168}"
SFT_EXPECTED_TRANSFORMERS_VERSION="${SFT_EXPECTED_TRANSFORMERS_VERSION:-4.46.3}"
SFT_EXPECTED_TRL_VERSION="${SFT_EXPECTED_TRL_VERSION:-0.11.2}"

if [[ "${SFT_INPUT_FORMAT}" != "prompt" && "${SFT_INPUT_FORMAT}" != "numeric_plain" ]]; then
  echo "SFT_INPUT_FORMAT must be prompt or numeric_plain; got ${SFT_INPUT_FORMAT}." >&2
  exit 1
fi

for boolean_name in SFT_FORCE_RETOKENIZE SFT_OVERWRITE_OUTPUT; do
  value="${!boolean_name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${boolean_name} must be 0 or 1; got ${value}." >&2
    exit 1
  fi
done

if [[ "${SFT_OVERWRITE_OUTPUT}" == "1" && -n "${SFT_RESUME_FROM_CHECKPOINT}" ]]; then
  echo "SFT_OVERWRITE_OUTPUT and SFT_RESUME_FROM_CHECKPOINT cannot be used together." >&2
  exit 1
fi

if [[ ! -d "${ROOT}/model/${MODEL}" ]]; then
  echo "Base model directory does not exist: ${ROOT}/model/${MODEL}" >&2
  exit 1
fi
if [[ ! -s "${DATA_DIR}/city_metrics/city_metrics_train.csv" ]]; then
  echo "Training-city manifest does not exist: ${DATA_DIR}/city_metrics/city_metrics_train.csv" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}/train" ]]; then
  echo "Training dataset directory does not exist: ${DATA_DIR}/train" >&2
  exit 1
fi

if ! runtime_versions="$("${PYTHON}" -c \
  'import transformers, trl; print(transformers.__version__, trl.__version__)')"; then
  echo "Unable to import transformers and trl. Activate uhi-e3llm first." >&2
  exit 1
fi
read -r transformers_version trl_version <<< "${runtime_versions}"
if [[ "${transformers_version}" != "${SFT_EXPECTED_TRANSFORMERS_VERSION}" || "${trl_version}" != "${SFT_EXPECTED_TRL_VERSION}" ]]; then
  echo "Expected transformers=${SFT_EXPECTED_TRANSFORMERS_VERSION} and trl=${SFT_EXPECTED_TRL_VERSION}; got transformers=${transformers_version} and trl=${trl_version}." >&2
  exit 1
fi
echo "SFT runtime: transformers=${transformers_version}, trl=${trl_version}"

if [[ -d "${SFT_OUTPUT}" ]] &&
   [[ -n "$(find "${SFT_OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] &&
   [[ -z "${SFT_RESUME_FROM_CHECKPOINT}" ]]; then
  if [[ "${SFT_OVERWRITE_OUTPUT}" != "1" ]]; then
    echo "SFT output directory is not empty: ${SFT_OUTPUT}" >&2
    echo "Set SFT_RESUME_FROM_CHECKPOINT to continue or SFT_OVERWRITE_OUTPUT=1 to replace it." >&2
    exit 1
  fi
  output_path="$(realpath -m "${SFT_OUTPUT}")"
  if [[ "${output_path}" == "/" || "${output_path}" == "${ROOT}" ]]; then
    echo "Refusing to remove unsafe SFT_OUTPUT path: ${SFT_OUTPUT}" >&2
    exit 1
  fi
  rm -rf -- "${output_path}"
fi

command=(env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" src/SFT_trainer.py
  --model "${MODEL}" \
  --dataset_path "${DATA_DIR}" \
  --output_dir "${SFT_OUTPUT}" \
  --train_epochs "${SFT_EPOCHS}" \
  --batch_size "${SFT_BATCH_SIZE}" \
  --gradient_accumulation 1 \
  --learning_rate 0.0001 \
  --max_length "${SFT_MAX_LENGTH}" \
  --num_workers "${SFT_NUM_WORKERS}" \
  --seed "${SFT_SEED}" \
  --expected_cities "${SFT_EXPECTED_CITIES}" \
  --expected_samples_per_city "${SFT_EXPECTED_SAMPLES_PER_CITY}" \
  --target_modules q_proj v_proj \
  --input_format "${SFT_INPUT_FORMAT}" \
  --efficiency_record "${EFFICIENCY_RECORD_DIR}/training_uhi_sft.json")

if [[ -n "${SFT_DATA_SEED}" ]]; then
  command+=(--data_seed "${SFT_DATA_SEED}")
fi
if [[ "${SFT_FORCE_RETOKENIZE}" == "1" ]]; then
  command+=(--force_retokenize)
fi

if [[ -n "${SFT_RESUME_FROM_CHECKPOINT}" ]]; then
  command+=(--resume_from_checkpoint "${SFT_RESUME_FROM_CHECKPOINT}")
fi

run_logged "${SFT_RUN_NAME}" "${command[@]}"
