#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
cd "${ROOT}"

MODEL="${MODEL:-deepseek-8b_merged__Global500}"
PPO_OUTPUT="${PPO_OUTPUT:-${ROOT}/model_adapters/PPO/${MODEL}}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_UPDATE_EPOCHS="${PPO_UPDATE_EPOCHS:-4}"
PPO_GRADIENT_ACCUMULATION="${PPO_GRADIENT_ACCUMULATION:-1}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-3e-6}"
PPO_MAX_LENGTH="${PPO_MAX_LENGTH:-1300}"
PPO_MAX_NEW_TOKENS="${PPO_MAX_NEW_TOKENS:-24}"
PPO_PENALTY_LAMBDA="${PPO_PENALTY_LAMBDA:-10}"
PPO_VALID_MIN="${PPO_VALID_MIN:--5}"
PPO_VALID_MAX="${PPO_VALID_MAX:-5}"
PPO_MAX_DECIMAL_PLACES="${PPO_MAX_DECIMAL_PLACES:-3}"
PPO_SEED="${PPO_SEED:-0}"
PPO_LOG_WITH="${PPO_LOG_WITH:-tensorboard}"
PPO_DEVICE="${PPO_DEVICE:-cuda}"
PPO_NUM_PROCESSES="${PPO_NUM_PROCESSES:-1}"
PPO_MAIN_PROCESS_PORT="${PPO_MAIN_PROCESS_PORT:-29500}"
PPO_ROLLOUT_WORKER_DEVICES="${PPO_ROLLOUT_WORKER_DEVICES:-}"
PPO_SAVE_EVERY_STEPS="${PPO_SAVE_EVERY_STEPS:-0}"
PPO_EXPECTED_TRAIN_CITIES="${PPO_EXPECTED_TRAIN_CITIES:-500}"
PPO_EXPECTED_SAMPLES_PER_CITY="${PPO_EXPECTED_SAMPLES_PER_CITY:-168}"
PPO_RESUME_CHECKPOINT="${PPO_RESUME_CHECKPOINT:-}"
PPO_RESUME_TRAINING_STATE="${PPO_RESUME_TRAINING_STATE:-}"
PPO_START_EPOCH="${PPO_START_EPOCH:-0}"
PPO_OVERWRITE="${PPO_OVERWRITE:-0}"
PPO_INPUT_FORMAT="${PPO_INPUT_FORMAT:-prompt_task_only}"
PPO_LOG_PREFIX="${PPO_LOG_PREFIX:-}"
ppo_log_prefix="${PPO_LOG_PREFIX:+${PPO_LOG_PREFIX}_}"

# The manuscript reward statistic e_bar (recorded as y_bar in the code) is
# the sample-weighted MAE of the fused SFT model on the 500 training cities'
# 2004-2017 training targets. Fixed mode is retained only for controlled
# reward-baseline ablations.
PPO_BASELINE_MODE="${PPO_BASELINE_MODE:-train}"
PPO_REWARD_BASELINE="${PPO_REWARD_BASELINE:-}"
PPO_BASELINE_OUTPUT="${PPO_BASELINE_OUTPUT:-${OUTPUT_ROOT}/ppo_train_baseline}"
PPO_BASELINE_CHECKPOINT="${PPO_BASELINE_CHECKPOINT:-fused_sft_train}"
PPO_BASELINE_BATCH_SIZE="${PPO_BASELINE_BATCH_SIZE:-24}"
PPO_BASELINE_MAX_NEW_TOKENS="${PPO_BASELINE_MAX_NEW_TOKENS:-${PPO_MAX_NEW_TOKENS}}"
PPO_BASELINE_TORCH_DTYPE="${PPO_BASELINE_TORCH_DTYPE:-bfloat16}"
PPO_EXPECTED_BASELINE_CITIES="${PPO_EXPECTED_BASELINE_CITIES:-500}"
PPO_EXPECTED_BASELINE_SAMPLES_PER_CITY="${PPO_EXPECTED_BASELINE_SAMPLES_PER_CITY:-168}"

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
echo "PPO runtime: transformers=${transformers_version}, trl=${trl_version}, peft=${peft_version}"

for value_name in PPO_EPOCHS PPO_BATCH_SIZE PPO_MINI_BATCH_SIZE \
  PPO_UPDATE_EPOCHS PPO_GRADIENT_ACCUMULATION PPO_MAX_LENGTH \
  PPO_MAX_NEW_TOKENS PPO_EXPECTED_TRAIN_CITIES \
  PPO_EXPECTED_SAMPLES_PER_CITY PPO_BASELINE_BATCH_SIZE \
  PPO_BASELINE_MAX_NEW_TOKENS PPO_EXPECTED_BASELINE_CITIES \
  PPO_EXPECTED_BASELINE_SAMPLES_PER_CITY PPO_NUM_PROCESSES \
  PPO_MAIN_PROCESS_PORT; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer; got ${value}." >&2
    exit 1
  fi
done
if [[ ! "${PPO_START_EPOCH}" =~ ^[0-9]+$ ]]; then
  echo "PPO_START_EPOCH must be a non-negative integer; got ${PPO_START_EPOCH}." >&2
  exit 1
fi
if [[ ! "${PPO_SAVE_EVERY_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "PPO_SAVE_EVERY_STEPS must be a non-negative integer; got ${PPO_SAVE_EVERY_STEPS}." >&2
  exit 1
fi

visible_devices="${GPU_ID}"
if [[ "${PPO_DEVICE}" == "cpu" ]]; then
  visible_devices=""
fi
if (( PPO_NUM_PROCESSES > 1 )); then
  if [[ "${PPO_DEVICE}" != "cuda" ]]; then
    echo "Multi-process PPO requires PPO_DEVICE=cuda." >&2
    exit 1
  fi
  IFS=',' read -r -a visible_device_list <<< "${visible_devices}"
  if (( ${#visible_device_list[@]} < PPO_NUM_PROCESSES )); then
    echo "GPU_ID must list at least ${PPO_NUM_PROCESSES} devices; got ${GPU_ID}." >&2
    exit 1
  fi
fi
if [[ -n "${PPO_ROLLOUT_WORKER_DEVICES}" && "${PPO_NUM_PROCESSES}" != "1" ]]; then
  echo "Parallel rollout workers require PPO_NUM_PROCESSES=1." >&2
  exit 1
fi
if (( PPO_BATCH_SIZE % (PPO_MINI_BATCH_SIZE * PPO_GRADIENT_ACCUMULATION) != 0 )); then
  echo "PPO_BATCH_SIZE must be divisible by PPO_MINI_BATCH_SIZE * PPO_GRADIENT_ACCUMULATION." >&2
  exit 1
fi
if [[ -z "${PPO_RESUME_CHECKPOINT}" && "${PPO_START_EPOCH}" != "0" ]]; then
  echo "PPO_START_EPOCH requires PPO_RESUME_CHECKPOINT." >&2
  exit 1
fi
if [[ -n "${PPO_RESUME_TRAINING_STATE}" && -z "${PPO_RESUME_CHECKPOINT}" ]]; then
  echo "PPO_RESUME_TRAINING_STATE requires PPO_RESUME_CHECKPOINT." >&2
  exit 1
fi
if [[ "${PPO_OVERWRITE}" != "0" && "${PPO_OVERWRITE}" != "1" ]]; then
  echo "PPO_OVERWRITE must be 0 or 1; got ${PPO_OVERWRITE}." >&2
  exit 1
fi
if [[ "${PPO_INPUT_FORMAT}" != "prompt" && "${PPO_INPUT_FORMAT}" != "prompt_task_only" && "${PPO_INPUT_FORMAT}" != "numeric_plain" ]]; then
  echo "PPO_INPUT_FORMAT must be prompt, prompt_task_only, or numeric_plain; got ${PPO_INPUT_FORMAT}." >&2
  exit 1
fi

if [[ "${MODEL}" = /* ]]; then
  model_dir="${MODEL}"
else
  model_dir="${ROOT}/model/${MODEL}"
fi
if [[ ! -d "${model_dir}" ]]; then
  echo "Fused PPO base model not found: ${model_dir}" >&2
  echo "Run scripts/run_lora_fusion.sh before scripts/run_ppo.sh." >&2
  exit 1
fi
if [[ ! -f "${DATA_DIR}/city_metrics/city_metrics_train.csv" ]]; then
  echo "Training-city manifest not found: ${DATA_DIR}/city_metrics/city_metrics_train.csv" >&2
  exit 1
fi

# Protect PPO outputs before spending time on the validation inference.
if [[ "${PPO_EPOCHS}" == "1" ]]; then
  checkpoint_targets=("${PPO_OUTPUT}")
else
  checkpoint_targets=()
  for ((offset = 0; offset < PPO_EPOCHS; offset++)); do
    checkpoint_targets+=("${PPO_OUTPUT}/epoch$((PPO_START_EPOCH + offset))")
  done
fi
if [[ "${PPO_OVERWRITE}" == "0" ]]; then
  for checkpoint_target in "${checkpoint_targets[@]}"; do
    if [[ -e "${checkpoint_target}" || -L "${checkpoint_target}" ]]; then
      echo "PPO checkpoint already exists: ${checkpoint_target}" >&2
      echo "Choose a new PPO_OUTPUT or set PPO_OVERWRITE=1." >&2
      exit 1
    fi
  done
fi

reward_baseline_report=""
case "${PPO_BASELINE_MODE}" in
  train)
    if [[ "${MODEL}" = */* ]]; then
      echo "Train baseline mode requires MODEL to be a model directory name, not a path: ${MODEL}" >&2
      exit 1
    fi
    run_logged "${ppo_log_prefix}ppo_train_baseline_inference" env \
      CUDA_VISIBLE_DEVICES="${visible_devices}" PYTHONUNBUFFERED=1 \
      TOKENIZERS_PARALLELISM=false "${PYTHON}" src/Inference.py \
      --model "${MODEL}" \
      --adapter_path "" \
      --checkpoint "${PPO_BASELINE_CHECKPOINT}" \
      --dataset_path "${DATA_DIR}" \
      --save_path "${PPO_BASELINE_OUTPUT}" \
      --device "${PPO_DEVICE}" \
      --mode train \
      --batch_size "${PPO_BASELINE_BATCH_SIZE}" \
      --max_length "${PPO_MAX_LENGTH}" \
      --max_new_tokens "${PPO_BASELINE_MAX_NEW_TOKENS}" \
      --num_runs 1 \
      --temperature 0 \
      --base_seed "${PPO_SEED}" \
      --valid_min "${PPO_VALID_MIN}" \
      --valid_max "${PPO_VALID_MAX}" \
      --max_decimal_places "${PPO_MAX_DECIMAL_PLACES}" \
      --invalid_policy error \
      --expected_cities "${PPO_EXPECTED_BASELINE_CITIES}" \
      --expected_samples_per_city "${PPO_EXPECTED_BASELINE_SAMPLES_PER_CITY}" \
      --torch_dtype "${PPO_BASELINE_TORCH_DTYPE}" \
      --input_format "${PPO_INPUT_FORMAT}"

    baseline_inference_csv="${PPO_BASELINE_OUTPUT}/${MODEL}/_${PPO_BASELINE_CHECKPOINT}_inference.csv"
    reward_baseline_report="${PPO_BASELINE_OUTPUT}/${MODEL}/ppo_reward_baseline.json"
    run_logged "${ppo_log_prefix}derive_ppo_reward_baseline" "${PYTHON}" \
      scripts/derive_ppo_reward_baseline.py \
      --inference-csv "${baseline_inference_csv}" \
      --output "${reward_baseline_report}" \
      --split train \
      --expected-cities "${PPO_EXPECTED_BASELINE_CITIES}" \
      --expected-samples-per-city "${PPO_EXPECTED_BASELINE_SAMPLES_PER_CITY}" \
      --model "${MODEL}" \
      --dataset-path "${DATA_DIR}" \
      --overwrite
    PPO_REWARD_BASELINE="$("${PYTHON}" -c \
      'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["y_bar"])' \
      "${reward_baseline_report}")"
    ;;
  fixed)
    if [[ -z "${PPO_REWARD_BASELINE}" ]]; then
      echo "PPO_BASELINE_MODE=fixed requires PPO_REWARD_BASELINE." >&2
      exit 1
    fi
    ;;
  *)
    echo "PPO_BASELINE_MODE must be train or fixed; got ${PPO_BASELINE_MODE}." >&2
    exit 1
    ;;
esac
echo "PPO reward baseline: mode=${PPO_BASELINE_MODE}, e_bar=${PPO_REWARD_BASELINE}"

launcher=("${PYTHON}")
if (( PPO_NUM_PROCESSES > 1 )); then
  accelerate_bin="$(command -v accelerate || true)"
  if [[ -z "${accelerate_bin}" ]]; then
    echo "accelerate executable not found in the active uhi-e3llm environment." >&2
    exit 1
  fi
  launcher=("${accelerate_bin}" launch
    --multi_gpu
    --num_processes "${PPO_NUM_PROCESSES}"
    --num_machines 1
    --mixed_precision no
    --dynamo_backend no
    --main_process_port "${PPO_MAIN_PROCESS_PORT}")
fi

command=(env CUDA_VISIBLE_DEVICES="${visible_devices}" PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false "${launcher[@]}" src/PPOTrainer.py
  --model "${MODEL}"
  --dataset_path "${DATA_DIR}"
  --save_path "${PPO_OUTPUT}"
  --train_epochs "${PPO_EPOCHS}"
  --batch_size "${PPO_BATCH_SIZE}"
  --mini_batch_size "${PPO_MINI_BATCH_SIZE}"
  --ppo_epochs "${PPO_UPDATE_EPOCHS}"
  --gradient_accumulation_steps "${PPO_GRADIENT_ACCUMULATION}"
  --learning_rate "${PPO_LEARNING_RATE}"
  --max_length "${PPO_MAX_LENGTH}"
  --max_new_tokens "${PPO_MAX_NEW_TOKENS}"
  --reward_baseline "${PPO_REWARD_BASELINE}"
  --penalty_lambda "${PPO_PENALTY_LAMBDA}"
  --valid_min "${PPO_VALID_MIN}"
  --valid_max "${PPO_VALID_MAX}"
  --max_decimal_places "${PPO_MAX_DECIMAL_PLACES}"
  --seed "${PPO_SEED}"
  --device "${PPO_DEVICE}"
  --expected_train_cities "${PPO_EXPECTED_TRAIN_CITIES}"
  --expected_samples_per_city "${PPO_EXPECTED_SAMPLES_PER_CITY}"
  --target_modules q_proj v_proj
  --input_format "${PPO_INPUT_FORMAT}"
  --log_with "${PPO_LOG_WITH}"
  --start_epoch "${PPO_START_EPOCH}"
  --efficiency_record "${EFFICIENCY_RECORD_DIR}/training_uhi_ppo.json")

if [[ "${PPO_SAVE_EVERY_STEPS}" != "0" ]]; then
  command+=(--save_every_steps "${PPO_SAVE_EVERY_STEPS}")
fi

if [[ -n "${reward_baseline_report}" ]]; then
  command+=(--reward_baseline_report "${reward_baseline_report}")
fi
if [[ -n "${PPO_RESUME_CHECKPOINT}" ]]; then
  command+=(--resume_ppo_checkpoint "${PPO_RESUME_CHECKPOINT}")
fi
if [[ -n "${PPO_RESUME_TRAINING_STATE}" ]]; then
  command+=(--resume_training_state "${PPO_RESUME_TRAINING_STATE}")
fi
if [[ "${PPO_OVERWRITE}" == "1" ]]; then
  command+=(--overwrite)
fi
if [[ -n "${PPO_ROLLOUT_WORKER_DEVICES}" ]]; then
  command+=(--rollout_worker_devices "${PPO_ROLLOUT_WORKER_DEVICES}")
fi

run_logged "${ppo_log_prefix}main_ppo" "${command[@]}"
