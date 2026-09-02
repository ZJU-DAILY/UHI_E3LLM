#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

MODEL="${MODEL:-deepseek-8b_merged__Global500}"
SENSITIVITY_ADAPTER_ROOT="${SENSITIVITY_ADAPTER_ROOT:-${ROOT}/model_adapters/sensitivity}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_UPDATE_EPOCHS="${PPO_UPDATE_EPOCHS:-4}"
PPO_GRADIENT_ACCUMULATION="${PPO_GRADIENT_ACCUMULATION:-1}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-3e-6}"
PPO_MAX_LENGTH="${PPO_MAX_LENGTH:-1300}"
PPO_MAX_NEW_TOKENS="${PPO_MAX_NEW_TOKENS:-24}"
PPO_SEED="${PPO_SEED:-0}"
PPO_EXPECTED_TRAIN_CITIES="${PPO_EXPECTED_TRAIN_CITIES:-500}"
PPO_EXPECTED_SAMPLES_PER_CITY="${PPO_EXPECTED_SAMPLES_PER_CITY:-168}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
INFERENCE_MAX_NEW_TOKENS="${INFERENCE_MAX_NEW_TOKENS:-19}"
INVALID_POLICY="${INVALID_POLICY:-error}"
EVAL_MODE="${EVAL_MODE:-validation}"
EXPECTED_EVAL_CITIES="${EXPECTED_EVAL_CITIES:-500}"
EXPECTED_EVAL_SAMPLES_PER_CITY="${EXPECTED_EVAL_SAMPLES_PER_CITY:-36}"
SENSITIVITY_REWARD_BASELINE_REPORT="${SENSITIVITY_REWARD_BASELINE_REPORT:-${OUTPUT_ROOT}/ppo_train_baseline/${MODEL}/ppo_reward_baseline.json}"
SENSITIVITY_REWARD_BASELINE="${SENSITIVITY_REWARD_BASELINE:-}"
EXPLANATION_DATA_ROOT="${EXPLANATION_DATA_ROOT:-${DATA_DIR}/explanation_samples}"

if ! runtime_versions="$("${PYTHON}" -c \
  'import peft, transformers, trl; print(transformers.__version__, trl.__version__, peft.__version__)')"; then
  echo "Unable to import transformers, TRL, and PEFT. Activate uhi-e3llm first." >&2
  exit 1
fi
read -r transformers_version trl_version peft_version <<< "${runtime_versions}"
if [[ "${transformers_version}" != "4.46.3" || "${trl_version}" != "0.11.2" ]]; then
  echo "Expected transformers=4.46.3 and trl=0.11.2; got transformers=${transformers_version} and trl=${trl_version}." >&2
  exit 1
fi

for value_name in PPO_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_UPDATE_EPOCHS \
  PPO_GRADIENT_ACCUMULATION PPO_MAX_LENGTH PPO_MAX_NEW_TOKENS \
  PPO_EXPECTED_TRAIN_CITIES PPO_EXPECTED_SAMPLES_PER_CITY \
  INFERENCE_BATCH_SIZE INFERENCE_MAX_NEW_TOKENS EXPECTED_EVAL_CITIES \
  EXPECTED_EVAL_SAMPLES_PER_CITY; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer; got ${value}." >&2
    exit 1
  fi
done
if (( PPO_BATCH_SIZE % (PPO_MINI_BATCH_SIZE * PPO_GRADIENT_ACCUMULATION) != 0 )); then
  echo "PPO_BATCH_SIZE must be divisible by PPO_MINI_BATCH_SIZE * PPO_GRADIENT_ACCUMULATION." >&2
  exit 1
fi
if [[ "${EVAL_MODE}" != "validation" ]]; then
  echo "PPO penalty selection must use EVAL_MODE=validation, not ${EVAL_MODE}." >&2
  exit 1
fi
if [[ ! -d "${ROOT}/model/${MODEL}" ]]; then
  echo "Fused PPO base model not found: ${ROOT}/model/${MODEL}" >&2
  echo "Run scripts/run_lora_fusion.sh and scripts/run_ppo.sh first." >&2
  exit 1
fi
if [[ ! -s "${SENSITIVITY_REWARD_BASELINE_REPORT}" ]]; then
  echo "PPO reward-baseline report not found: ${SENSITIVITY_REWARD_BASELINE_REPORT}" >&2
  echo "Run scripts/run_ppo.sh first so all lambda runs reuse its fitted-SFT baseline." >&2
  exit 1
fi
if [[ -z "${SENSITIVITY_REWARD_BASELINE}" ]]; then
  SENSITIVITY_REWARD_BASELINE="$("${PYTHON}" -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["y_bar"])' \
    "${SENSITIVITY_REWARD_BASELINE_REPORT}")"
fi

for lambda in 0 1 5 10 20; do
  adapter_dir="${SENSITIVITY_ADAPTER_ROOT}/lambda_${lambda}"
  inference_dir="${OUTPUT_ROOT}/sensitivity/ppo_lambda_${lambda}"
  if [[ -e "${adapter_dir}" || -L "${adapter_dir}" ]]; then
    echo "Sensitivity adapter already exists: ${adapter_dir}" >&2
    echo "Choose a new SENSITIVITY_ADAPTER_ROOT or remove the incomplete run explicitly." >&2
    exit 1
  fi
  if [[ -e "${inference_dir}" || -L "${inference_dir}" ]]; then
    echo "Sensitivity inference output already exists: ${inference_dir}" >&2
    echo "Choose a new OUTPUT_ROOT or remove the incomplete run explicitly." >&2
    exit 1
  fi
done
if [[ -e "${OUTPUT_ROOT}/sensitivity/ppo_lambda_summary.csv" ]]; then
  echo "Sensitivity summary already exists: ${OUTPUT_ROOT}/sensitivity/ppo_lambda_summary.csv" >&2
  echo "Choose a new OUTPUT_ROOT or remove the previous complete run explicitly." >&2
  exit 1
fi

for lambda in 0 1 5 10 20; do
  adapter_dir="${SENSITIVITY_ADAPTER_ROOT}/lambda_${lambda}"
  inference_dir="${OUTPUT_ROOT}/sensitivity/ppo_lambda_${lambda}"
  run_logged "ppo_lambda_${lambda}" env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON}" src/PPOTrainer.py \
    --model "${MODEL}" \
    --dataset_path "${DATA_DIR}" \
    --save_path "${adapter_dir}" \
    --train_epochs 2 \
    --batch_size "${PPO_BATCH_SIZE}" \
    --mini_batch_size "${PPO_MINI_BATCH_SIZE}" \
    --ppo_epochs "${PPO_UPDATE_EPOCHS}" \
    --gradient_accumulation_steps "${PPO_GRADIENT_ACCUMULATION}" \
    --learning_rate "${PPO_LEARNING_RATE}" \
    --max_length "${PPO_MAX_LENGTH}" \
    --max_new_tokens "${PPO_MAX_NEW_TOKENS}" \
    --reward_baseline "${SENSITIVITY_REWARD_BASELINE}" \
    --reward_baseline_report "${SENSITIVITY_REWARD_BASELINE_REPORT}" \
    --penalty_lambda "${lambda}" \
    --valid_min -5 \
    --valid_max 5 \
    --max_decimal_places 3 \
    --seed "${PPO_SEED}" \
    --device cuda \
    --expected_train_cities "${PPO_EXPECTED_TRAIN_CITIES}" \
    --expected_samples_per_city "${PPO_EXPECTED_SAMPLES_PER_CITY}" \
    --target_modules q_proj v_proj \
    --input_format prompt_task_only \
    --log_with tensorboard \
    --logging_dir "${OUTPUT_ROOT}/sensitivity/tensorboard/lambda_${lambda}"

  distilled_adapter_root="${adapter_dir}/explanation_distilled"
  for policy_epoch in 0 1; do
    distill_model="${MODEL}_lambda_${lambda}_PPO_epoch${policy_epoch}_merged"
    env \
      GPU_ID="${GPU_ID}" PYTHON="${PYTHON}" DATA_DIR="${DATA_DIR}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_DIR="${LOG_DIR}" \
      BASE_MODEL="${MODEL}" \
      EXPLANATION_POLICY_ADAPTER_DIR="${adapter_dir}/epoch${policy_epoch}" \
      EXPLANATION_POLICY_EPOCH="${policy_epoch}" \
      MERGED_MODEL="${distill_model}" \
      MERGED_MODEL_DIR="${ROOT}/model/${distill_model}" \
      EXPLANATION_DATA_ROOT="${EXPLANATION_DATA_ROOT}" \
      EXPLANATION_INPUT_FORMAT=prompt \
      EXPLANATION_ADAPTER="lambda_${lambda}_epoch${policy_epoch}_explanation" \
      EXPLANATION_ADAPTER_DIR="${distilled_adapter_root}/epoch${policy_epoch}" \
      EXPLANATION_LOG_PREFIX="sensitivity_lambda_${lambda}_epoch${policy_epoch}" \
      EXPLANATION_RUN_INFERENCE=0 \
      INVALID_POLICY="${INVALID_POLICY}" \
      bash scripts/run_explanation_distillation.sh
  done

  env \
    GPU_ID="${GPU_ID}" PYTHON="${PYTHON}" DATA_DIR="${DATA_DIR}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_DIR="${LOG_DIR}" \
    BASE_MODEL="${MODEL}" DISTILLED_ADAPTER_ROOT="${distilled_adapter_root}" \
    DISTILLED_MODEL_PREFIX="${MODEL}_lambda_${lambda}_PPO_epoch" \
    DISTILLED_OUTPUT_DIR="${inference_dir}" DISTILLED_EPOCHS="0 1" \
    DISTILLED_MODE="${EVAL_MODE}" \
    DISTILLED_EXPECTED_CITIES="${EXPECTED_EVAL_CITIES}" \
    DISTILLED_EXPECTED_SAMPLES_PER_CITY="${EXPECTED_EVAL_SAMPLES_PER_CITY}" \
    DISTILLED_BATCH_SIZE="${INFERENCE_BATCH_SIZE}" \
    DISTILLED_MAX_LENGTH="${PPO_MAX_LENGTH}" \
    DISTILLED_MAX_NEW_TOKENS="${INFERENCE_MAX_NEW_TOKENS}" \
    DISTILLED_INPUT_FORMAT=prompt DISTILLED_INVALID_POLICY="${INVALID_POLICY}" \
    DISTILLED_LOG_PREFIX="sensitivity_lambda_${lambda}" \
    bash scripts/run_distilled_ensemble_inference.sh
done

run_logged summarize_ppo_lambda_sensitivity "${PYTHON}" \
  scripts/summarize_ppo_sensitivity.py \
  --root "${OUTPUT_ROOT}/sensitivity" \
  --output "${OUTPUT_ROOT}/sensitivity/ppo_lambda_summary.csv" \
  --lambdas 0 1 5 10 20 \
  --expected_cities "${EXPECTED_EVAL_CITIES}" \
  --expected_samples_per_city "${EXPECTED_EVAL_SAMPLES_PER_CITY}"
