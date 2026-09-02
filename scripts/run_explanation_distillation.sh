#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

BASE_MODEL="${BASE_MODEL:-deepseek-8b_merged__Global500}"
EXPLANATION_POLICY_EPOCHS="${EXPLANATION_POLICY_EPOCHS:-0 1}"
EXPLANATION_INTERNAL_SINGLE_EPOCH="${EXPLANATION_INTERNAL_SINGLE_EPOCH:-0}"
PPO_POLICY_ADAPTER_ROOT="${PPO_POLICY_ADAPTER_ROOT:-${ROOT}/model_adapters/PPO/${BASE_MODEL}}"
EXPLANATION_DISTILLED_ADAPTER_ROOT="${EXPLANATION_DISTILLED_ADAPTER_ROOT:-${ROOT}/model_adapters/explanation_distilled/${BASE_MODEL}}"

# The full model ensembles two independently distilled policies. Explicit
# EXPLANATION_POLICY_ADAPTER_DIR callers (for example, single-policy
# ablations) continue to execute exactly one distillation run.
if [[ "${EXPLANATION_INTERNAL_SINGLE_EPOCH}" == "0" &&
      ! -v EXPLANATION_POLICY_ADAPTER_DIR ]]; then
  read -r -a explanation_policy_epochs <<< "${EXPLANATION_POLICY_EPOCHS}"
  if (( ${#explanation_policy_epochs[@]} == 0 )); then
    echo "EXPLANATION_POLICY_EPOCHS must not be empty." >&2
    exit 1
  fi
  for policy_epoch in "${explanation_policy_epochs[@]}"; do
    if [[ ! "${policy_epoch}" =~ ^[0-9]+$ ]]; then
      echo "Invalid PPO policy epoch: ${policy_epoch}" >&2
      exit 1
    fi
    distill_model="${BASE_MODEL}_PPO_epoch${policy_epoch}_merged"
    env \
      EXPLANATION_INTERNAL_SINGLE_EPOCH=1 \
      EXPLANATION_POLICY_EPOCH="${policy_epoch}" \
      EXPLANATION_POLICY_ADAPTER_DIR="${PPO_POLICY_ADAPTER_ROOT}/epoch${policy_epoch}" \
      MERGED_MODEL="${distill_model}" \
      MERGED_MODEL_DIR="${ROOT}/model/${distill_model}" \
      EXPLANATION_ADAPTER="explanation_distill_global500_ppo_epoch${policy_epoch}" \
      EXPLANATION_ADAPTER_DIR="${EXPLANATION_DISTILLED_ADAPTER_ROOT}/epoch${policy_epoch}" \
      EXPLANATION_LOG_PREFIX="${EXPLANATION_LOG_PREFIX:-explanation}_epoch${policy_epoch}" \
      bash "$0"
  done
  exit 0
fi

# Draw one reproducible simple-random sample from the complete training pool.
EXPLANATION_TRAIN_CITY_METRICS="${EXPLANATION_TRAIN_CITY_METRICS:-${DATA_DIR}/city_metrics/city_metrics_train.csv}"
EXPLANATION_EXPECTED_TRAIN_CITIES="${EXPLANATION_EXPECTED_TRAIN_CITIES:-500}"
EXPLANATION_EXPECTED_TRAIN_SAMPLES_PER_CITY="${EXPLANATION_EXPECTED_TRAIN_SAMPLES_PER_CITY:-168}"
EXPLANATION_SAMPLE_COUNT="${EXPLANATION_SAMPLE_COUNT:-900}"
EXPLANATION_SAMPLING_SEED="${EXPLANATION_SAMPLING_SEED:-42}"
EXPLANATION_FIRST_TARGET="${EXPLANATION_FIRST_TARGET:-2004-01}"
EXPLANATION_LAST_TARGET="${EXPLANATION_LAST_TARGET:-2017-12}"
EXPLANATION_DATA_ROOT="${EXPLANATION_DATA_ROOT:-${DATA_DIR}/explanation_samples}"
EXPLANATION_CONTENT_DIR="${EXPLANATION_DATA_ROOT}/content"
EXPLANATION_SELECTION_MANIFEST="${EXPLANATION_DATA_ROOT}/selection_manifest.json"
EXPLANATION_CONCURRENCY="${EXPLANATION_CONCURRENCY:-1}"
EXPLANATION_API_MODEL="${EXPLANATION_API_MODEL:-deepseek-reasoner}"
EXPLANATION_API_TIMEOUT="${EXPLANATION_API_TIMEOUT:-1800}"
EXPLANATION_API_MAX_ATTEMPTS="${EXPLANATION_API_MAX_ATTEMPTS:-5}"
EXPLANATION_API_RETRY_DELAY="${EXPLANATION_API_RETRY_DELAY:-2}"
EXPLANATION_FORCE_REGENERATE="${EXPLANATION_FORCE_REGENERATE:-0}"
EXPLANATION_DRY_RUN="${EXPLANATION_DRY_RUN:-0}"
EXPLANATION_GENERATE_ONLY="${EXPLANATION_GENERATE_ONLY:-0}"
EXPLANATION_INPUT_FORMAT="${EXPLANATION_INPUT_FORMAT:-prompt}"
EXPLANATION_REUSE_CONTENT_DIR="${EXPLANATION_REUSE_CONTENT_DIR:-}"
EXPLANATION_RUN_INFERENCE="${EXPLANATION_RUN_INFERENCE:-0}"

# Each PPO policy is materialized and distilled independently; model weights
# are never fused across PPO epochs. Ablations may select one policy explicitly
# or pass an empty adapter when the distillation base is already complete.
EXPLANATION_POLICY_EPOCH="${EXPLANATION_POLICY_EPOCH:-1}"
PPO_POLICY_ADAPTER_DIR="${PPO_POLICY_ADAPTER_DIR:-${PPO_POLICY_ADAPTER_ROOT}/epoch${EXPLANATION_POLICY_EPOCH}}"
if [[ -v EXPLANATION_POLICY_ADAPTER_DIR ]]; then
  policy_adapter_dir="${EXPLANATION_POLICY_ADAPTER_DIR}"
else
  policy_adapter_dir="${PPO_POLICY_ADAPTER_DIR}"
fi
if [[ -n "${policy_adapter_dir}" ]]; then
  default_distill_model="${BASE_MODEL}_PPO_epoch${EXPLANATION_POLICY_EPOCH}_merged"
else
  default_distill_model="${BASE_MODEL}"
fi
MERGED_MODEL="${MERGED_MODEL:-${default_distill_model}}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-${ROOT}/model/${MERGED_MODEL}}"

EXPLANATION_ADAPTER="${EXPLANATION_ADAPTER:-explanation_distill_global500_ppo_epoch${EXPLANATION_POLICY_EPOCH}}"
EXPLANATION_ADAPTER_DIR="${EXPLANATION_ADAPTER_DIR:-${EXPLANATION_DISTILLED_ADAPTER_ROOT}/epoch${EXPLANATION_POLICY_EPOCH}}"
EXPLANATION_LOG_PREFIX="${EXPLANATION_LOG_PREFIX:-${EXPLANATION_ADAPTER}}"
explanation_log_prefix="${EXPLANATION_LOG_PREFIX:+${EXPLANATION_LOG_PREFIX}_}"
DISTILL_DEVICE="${DISTILL_DEVICE:-cuda:0}"
DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-1}"
DISTILL_LEARNING_RATE="${DISTILL_LEARNING_RATE:-5e-7}"
DISTILL_MAX_LENGTH="${DISTILL_MAX_LENGTH:-6000}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-1}"
DISTILL_MAX_STEPS="${DISTILL_MAX_STEPS:-0}"
DISTILL_NUM_WORKERS="${DISTILL_NUM_WORKERS:-2}"
DISTILL_RESULT_LOSS_WEIGHT="${DISTILL_RESULT_LOSS_WEIGHT:-1.0}"
DISTILL_EXPLANATION_LOSS_WEIGHT="${DISTILL_EXPLANATION_LOSS_WEIGHT:-0.001}"
DISTILL_TORCH_DTYPE="${DISTILL_TORCH_DTYPE:-bfloat16}"
DISTILL_SEED="${DISTILL_SEED:-42}"
EXPLANATION_INFERENCE_OUTPUT="${EXPLANATION_INFERENCE_OUTPUT:-${OUTPUT_ROOT}/explanation/${EXPLANATION_ADAPTER}}"
EXPLANATION_INFERENCE_BATCH_SIZE="${EXPLANATION_INFERENCE_BATCH_SIZE:-8}"
EXPLANATION_INFERENCE_MAX_NEW_TOKENS="${EXPLANATION_INFERENCE_MAX_NEW_TOKENS:-512}"
EXPLANATION_EXPECTED_TEST_CITIES="${EXPLANATION_EXPECTED_TEST_CITIES:-100}"
EXPLANATION_EXPECTED_TEST_SAMPLES_PER_CITY="${EXPLANATION_EXPECTED_TEST_SAMPLES_PER_CITY:-36}"
INVALID_POLICY="${INVALID_POLICY:-error}"

for boolean_name in EXPLANATION_FORCE_REGENERATE EXPLANATION_DRY_RUN \
  EXPLANATION_GENERATE_ONLY EXPLANATION_RUN_INFERENCE; do
  boolean_value="${!boolean_name}"
  if [[ "${boolean_value}" != "0" && "${boolean_value}" != "1" ]]; then
    echo "${boolean_name} must be 0 or 1; got ${boolean_value}." >&2
    exit 1
  fi
done
if [[ "${EXPLANATION_INPUT_FORMAT}" != "prompt" && "${EXPLANATION_INPUT_FORMAT}" != "numeric_plain" ]]; then
  echo "EXPLANATION_INPUT_FORMAT must be prompt or numeric_plain." >&2
  exit 1
fi
if [[ ! "${DISTILL_MAX_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "DISTILL_MAX_STEPS must be a non-negative integer; got ${DISTILL_MAX_STEPS}." >&2
  exit 1
fi
if [[ ! "${EXPLANATION_SAMPLING_SEED}" =~ ^[0-9]+$ ]]; then
  echo "EXPLANATION_SAMPLING_SEED must be a non-negative integer; got ${EXPLANATION_SAMPLING_SEED}." >&2
  exit 1
fi
for integer_name in EXPLANATION_EXPECTED_TRAIN_CITIES \
  EXPLANATION_EXPECTED_TRAIN_SAMPLES_PER_CITY EXPLANATION_SAMPLE_COUNT \
  EXPLANATION_CONCURRENCY DISTILL_BATCH_SIZE DISTILL_MAX_LENGTH \
  DISTILL_EPOCHS DISTILL_NUM_WORKERS EXPLANATION_EXPECTED_TEST_CITIES \
  EXPLANATION_EXPECTED_TEST_SAMPLES_PER_CITY; do
  integer_value="${!integer_name}"
  if [[ ! "${integer_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${integer_name} must be a positive integer; got ${integer_value}." >&2
    exit 1
  fi
done

if ! runtime_versions="$("${PYTHON}" -c \
  'import datasets, openai, peft, transformers, trl; print(transformers.__version__, trl.__version__, peft.__version__, openai.__version__)')"; then
  echo "Unable to import datasets, OpenAI, Transformers, TRL, and PEFT." >&2
  echo "Activate the UHI-E3LLM environment before running this workflow." >&2
  exit 1
fi
read -r transformers_version trl_version peft_version openai_version <<< "${runtime_versions}"
if [[ "${transformers_version}" != "4.46.3" || "${trl_version}" != "0.11.2" ]]; then
  echo "Expected transformers=4.46.3 and trl=0.11.2; got transformers=${transformers_version} and trl=${trl_version}." >&2
  exit 1
fi
echo "Explanation runtime: transformers=${transformers_version}, trl=${trl_version}, peft=${peft_version}, openai=${openai_version}"

flat_sample_count=0
if [[ -d "${EXPLANATION_DATA_ROOT}" ]]; then
  flat_sample_count="$(find "${EXPLANATION_DATA_ROOT}" -maxdepth 1 -type f -name 'sample_*.txt' | wc -l)"
fi
use_flat_release_samples=0
if [[ ! -s "${EXPLANATION_SELECTION_MANIFEST}" && "${flat_sample_count}" -gt 0 && "${EXPLANATION_FORCE_REGENERATE}" == "0" ]]; then
  if [[ "${flat_sample_count}" -ne "${EXPLANATION_SAMPLE_COUNT}" ]]; then
    echo "Expected ${EXPLANATION_SAMPLE_COUNT} flat explanation files, found ${flat_sample_count}." >&2
    exit 1
  fi
  EXPLANATION_CONTENT_DIR="${EXPLANATION_DATA_ROOT}"
  use_flat_release_samples=1
  echo "Using ${flat_sample_count} released flat explanation samples from ${EXPLANATION_DATA_ROOT}."
fi

generation_command=(env PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  "${PYTHON}" scripts/generate_explanations.py
  --data-dir "${DATA_DIR}"
  --output-dir "${EXPLANATION_DATA_ROOT}"
  --train-city-metrics "${EXPLANATION_TRAIN_CITY_METRICS}"
  --expected-train-cities "${EXPLANATION_EXPECTED_TRAIN_CITIES}"
  --expected-train-samples-per-city "${EXPLANATION_EXPECTED_TRAIN_SAMPLES_PER_CITY}"
  --sample-count "${EXPLANATION_SAMPLE_COUNT}"
  --sampling-seed "${EXPLANATION_SAMPLING_SEED}"
  --first-target "${EXPLANATION_FIRST_TARGET}"
  --last-target "${EXPLANATION_LAST_TARGET}"
  --student-input-format "${EXPLANATION_INPUT_FORMAT}"
  --api-model "${EXPLANATION_API_MODEL}"
  --concurrency "${EXPLANATION_CONCURRENCY}"
  --timeout "${EXPLANATION_API_TIMEOUT}"
  --max-attempts "${EXPLANATION_API_MAX_ATTEMPTS}"
  --retry-delay "${EXPLANATION_API_RETRY_DELAY}")
if [[ -n "${EXPLANATION_REUSE_CONTENT_DIR}" ]]; then
  generation_command+=(--reuse-content-dir "${EXPLANATION_REUSE_CONTENT_DIR}")
fi
if [[ "${EXPLANATION_FORCE_REGENERATE}" == "1" ]]; then
  generation_command+=(--overwrite)
fi
if [[ "${EXPLANATION_DRY_RUN}" == "1" ]]; then
  generation_command+=(--dry-run)
fi
if [[ "${use_flat_release_samples}" == "0" ]]; then
  run_logged "${explanation_log_prefix}generate_explanations" "${generation_command[@]}"
fi

if [[ "${EXPLANATION_DRY_RUN}" == "1" || "${EXPLANATION_GENERATE_ONLY}" == "1" ]]; then
  echo "Explanation generation stage complete; model merge and distillation were not started."
  exit 0
fi

expected_total="${EXPLANATION_SAMPLE_COUNT}"
if [[ "${use_flat_release_samples}" == "0" && ! -s "${EXPLANATION_SELECTION_MANIFEST}" ]]; then
  echo "Explanation selection manifest not found: ${EXPLANATION_SELECTION_MANIFEST}" >&2
  exit 1
fi
if [[ "${use_flat_release_samples}" == "1" ]]; then
  content_total="$(find "${EXPLANATION_CONTENT_DIR}" -maxdepth 1 -type f -name 'sample_*.txt' | wc -l)"
else
  content_total="$(find "${EXPLANATION_CONTENT_DIR}" -type f -name 'sample_*.txt' | wc -l)"
fi
if [[ "${content_total}" -ne "${expected_total}" ]]; then
  echo "Expected ${expected_total} explanation files, found ${content_total}." >&2
  exit 1
fi

if [[ ! -s "${ROOT}/model/${BASE_MODEL}/config.json" ]]; then
  echo "Explanation base model not found: ${ROOT}/model/${BASE_MODEL}" >&2
  exit 1
fi
if [[ -n "${policy_adapter_dir}" ]]; then
  if [[ ! -s "${policy_adapter_dir}/adapter_config.json" ]]; then
    echo "Final policy adapter not found: ${policy_adapter_dir}" >&2
    exit 1
  fi
  if [[ -s "${MERGED_MODEL_DIR}/config.json" ]]; then
    echo "Reusing materialized explanation base: ${MERGED_MODEL_DIR}"
  else
    if [[ -e "${MERGED_MODEL_DIR}" ]]; then
      echo "Incomplete merged-model directory already exists: ${MERGED_MODEL_DIR}" >&2
      echo "Remove it or choose another MERGED_MODEL name before rerunning." >&2
      exit 1
    fi
    run_logged "${explanation_log_prefix}materialize_policy_for_explanation" env \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
      "${PYTHON}" src/merge_lora.py \
      --model "${BASE_MODEL}" \
      --adapter_dir "${policy_adapter_dir}" \
      --output_dir "${MERGED_MODEL_DIR}"
  fi
elif [[ "${MERGED_MODEL_DIR}" != "${ROOT}/model/${BASE_MODEL}" ]]; then
  echo "Without a policy adapter, MERGED_MODEL must resolve to BASE_MODEL." >&2
  exit 1
fi

if [[ -s "${EXPLANATION_ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "Explanation adapter already exists; skipping distillation: ${EXPLANATION_ADAPTER_DIR}"
elif [[ -e "${EXPLANATION_ADAPTER_DIR}" ]]; then
  echo "Incomplete explanation-adapter directory already exists: ${EXPLANATION_ADAPTER_DIR}" >&2
  echo "Remove it or choose another EXPLANATION_ADAPTER name before rerunning." >&2
  exit 1
else
  distillation_command=(env
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1
    TOKENIZERS_PARALLELISM=false "${PYTHON}" src/explanation_distillation.py
    --device "${DISTILL_DEVICE}"
    --data_path "${EXPLANATION_CONTENT_DIR}"
    --model_path "${MERGED_MODEL_DIR}"
    --output_dir "${EXPLANATION_ADAPTER_DIR}"
    --input_format "${EXPLANATION_INPUT_FORMAT}"
    --batch_size "${DISTILL_BATCH_SIZE}"
    --learning_rate "${DISTILL_LEARNING_RATE}"
    --max_length "${DISTILL_MAX_LENGTH}"
    --num_epochs "${DISTILL_EPOCHS}"
    --max_steps "${DISTILL_MAX_STEPS}"
    --num_workers "${DISTILL_NUM_WORKERS}"
    --result_loss_weight "${DISTILL_RESULT_LOSS_WEIGHT}"
    --explanation_loss_weight "${DISTILL_EXPLANATION_LOSS_WEIGHT}"
    --torch_dtype "${DISTILL_TORCH_DTYPE}"
    --seed "${DISTILL_SEED}"
    --expected_samples "${EXPLANATION_SAMPLE_COUNT}")
  if [[ "${use_flat_release_samples}" == "0" ]]; then
    distillation_command+=(--selection_manifest "${EXPLANATION_SELECTION_MANIFEST}")
  fi
  run_logged "${explanation_log_prefix}explanation_distillation" "${distillation_command[@]}"
fi

if [[ ! -s "${EXPLANATION_ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "Distillation finished without a valid adapter: ${EXPLANATION_ADAPTER_DIR}" >&2
  exit 1
fi
if [[ "${EXPLANATION_RUN_INFERENCE}" == "1" ]]; then
  run_logged "explanation_inference_${EXPLANATION_ADAPTER}" env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false "${PYTHON}" src/Inference.py \
    --model "${MERGED_MODEL}" \
    --adapter_path explanation \
    --adapter_dir "${EXPLANATION_ADAPTER_DIR}" \
    --checkpoint "${EXPLANATION_ADAPTER}" \
    --dataset_path "${DATA_DIR}" \
    --save_path "${EXPLANATION_INFERENCE_OUTPUT}" \
    --device cuda \
    --mode test \
    --expected_cities "${EXPLANATION_EXPECTED_TEST_CITIES}" \
    --expected_samples_per_city "${EXPLANATION_EXPECTED_TEST_SAMPLES_PER_CITY}" \
    --batch_size "${EXPLANATION_INFERENCE_BATCH_SIZE}" \
    --max_length "${DISTILL_MAX_LENGTH}" \
    --max_new_tokens "${EXPLANATION_INFERENCE_MAX_NEW_TOKENS}" \
    --input_format "${EXPLANATION_INPUT_FORMAT}" \
    --invalid_policy "${INVALID_POLICY}"
fi
echo "Explanation-distillation workflow complete: ${EXPLANATION_ADAPTER_DIR}"
