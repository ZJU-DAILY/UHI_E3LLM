#!/usr/bin/env bash
# Internal baseline workflow library. Source this file from a public pipeline entry.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This is an internal workflow library; run scripts/run_main_evaluation.sh instead." >&2
  exit 2
fi

_baseline_has_model() {
  local requested="$1"
  [[ " ${TIMESERIES_MODELS:-armd filternet stmtm} " == *" ${requested} "* ]]
}

_baseline_prepare_timeseries_data() {
  local output_dir="$1"
  local train_cities="$2"
  local test_cities="$3"
  local scope
  scope="$(basename "$(dirname "${output_dir}")")"

  if [[ "${REUSE_TIMESERIES_DATA:-0}" == "1" &&
        -d "${output_dir}/all" &&
        -s "${output_dir}/multivariate_24.schema" ]]; then
    echo "Reusing prepared time-series data: ${output_dir}"
    return
  fi

  run_logged "prepare_timeseries_${scope}" "${PYTHON}" \
    "${ROOT}/scripts/prepare_timeseries_baselines.py" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${output_dir}" \
    --train_city_metrics "${train_cities}" \
    --test_city_metrics "${test_cities}"
}

_baseline_global_armd_checkpoint() {
  printf '%s\n' "${OUTPUT_ROOT}/baselines/armd/checkpoints_mv24_12/checkpoint-final.pt"
}

_baseline_global_filternet_checkpoint() {
  local epochs="${FILTERNET_EPOCHS:-5}"
  local setting="UHI500City1passes_${epochs}epochs_PaiFilter_custom_sl12_pl1_embed12_hidden64_bs1024_lr2e-05_drop0.0_manuscript_mv24"
  printf '%s\n' "${OUTPUT_ROOT}/baselines/filternet/checkpoints/${setting}/checkpoint.pth"
}

_baseline_global_stmtm_checkpoint() {
  local epochs="${STMTM_FINETUNE_EPOCHS:-5}"
  local setting="finetune_STMTM_Exchange_M_sl12_ll0_pl1_dm16_df64_nh4_dh64_el1_fc1_dp0.2_hdp0.2_ep${epochs}_bs1024_lr1.6e-05_wd0.05_ks200_sl4_pt0.2_topk3_tau0.1_alp0.5_iter_manuscript_cp1_cities500"
  printf '%s\n' "${OUTPUT_ROOT}/baselines/st_mtm/checkpoints/${setting}/checkpoint.pth"
}

_baseline_require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "Required fitted baseline artifact is missing: ${path}" >&2
    echo "Run scripts/run_main_evaluation.sh first." >&2
    return 1
  fi
}

require_global_baseline_artifacts() {
  local tabular_model_dir="${OUTPUT_ROOT}/baselines/tabular/models"
  local artifact
  for artifact in \
    "${tabular_model_dir}/standard_scaler.joblib" \
    "${tabular_model_dir}/xgb_n1000_depth16_lr001.json" \
    "${tabular_model_dir}/lgb_gbdt_n500_lr0023_leaves255.txt" \
    "${tabular_model_dir}/rf_n500_bootstrap_false.joblib" \
    "${tabular_model_dir}/mlp_384_192_96_epoch2800.pt" \
    "$(_baseline_global_armd_checkpoint)" \
    "$(_baseline_global_filternet_checkpoint)" \
    "$(_baseline_global_stmtm_checkpoint)"; do
    _baseline_require_file "${artifact}"
  done
}

run_global_tabular_baselines() {
  run_logged main_table_tabular_baselines env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON}" "${ROOT}/baseline/tabular_baselines.py" \
    --dataset_path "${DATA_DIR}" \
    --output_dir "${OUTPUT_ROOT}/baselines/tabular" \
    --models xgb lgb rf mlp \
    --seed 42 \
    --n_jobs "${N_JOBS:-64}" \
    --xgb_device "${XGB_DEVICE:-cuda}" \
    --lgb_device "${LGB_DEVICE:-gpu}" \
    --torch_device "${TORCH_DEVICE:-cuda:0}" \
    --efficiency_record_dir "${EFFICIENCY_RECORD_DIR}"
}

run_regional_tabular_baselines() {
  local region="$1"
  run_logged "regional_tabular_${region}" env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON}" "${ROOT}/baseline/tabular_leave_one_region_out.py" \
    --data_dir "${DATA_DIR}" \
    --dataset_root "${OUTPUT_ROOT}/regional/tabular_datasets" \
    --fold_root "${DATA_DIR}/regional_folds" \
    --output_dir "${OUTPUT_ROOT}/regional/tabular_results" \
    --models xgb lgb rf mlp \
    --regions "${region}" \
    --seed 42 \
    --n_jobs "${N_JOBS:-32}" \
    --xgb_device "${XGB_DEVICE:-cuda}" \
    --lgb_device "${LGB_DEVICE:-gpu}" \
    --torch_device "${TORCH_DEVICE:-cuda:0}"
}

run_global_timeseries_baselines() (
  local ts_dir="${TIMESERIES_DATA_DIR:-${OUTPUT_ROOT}/baselines/timeseries_data}"
  local train_cities="${DATA_DIR}/city_metrics/city_metrics_train.csv"
  local test_cities="${DATA_DIR}/city_metrics/city_metrics_test.csv"
  local armd_rounds="${ARMD_ROUNDS:-5}"
  local filternet_epochs="${FILTERNET_EPOCHS:-5}"
  local stmtm_pretrain_epochs="${STMTM_PRETRAIN_EPOCHS:-5}"
  local stmtm_finetune_epochs="${STMTM_FINETUNE_EPOCHS:-5}"
  local stmtm_root pretrain_root pretrain_checkpoint
  local -a limit_args=()
  local -a overwrite_args=()

  [[ -n "${MAX_TRAIN_CITIES:-}" ]] && limit_args+=(--max_train_cities "${MAX_TRAIN_CITIES}")
  [[ -n "${MAX_TEST_CITIES:-}" ]] && limit_args+=(--max_test_cities "${MAX_TEST_CITIES}")
  [[ "${OVERWRITE_OUTPUTS:-1}" == "1" ]] && overwrite_args+=(--overwrite_outputs)

  _baseline_prepare_timeseries_data "${ts_dir}" "${train_cities}" "${test_cities}"

  if _baseline_has_model armd; then
    cd "${ROOT}/baseline/armd"
    run_logged baseline_armd env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" main_new.py \
      --config_path Config/amod2.yaml \
      --save_dir "${OUTPUT_ROOT}/baselines/armd" \
      --gpu 0 \
      --train_city_metrics_path "${train_cities}" \
      --test_city_metrics_path "${test_cities}" \
      --uhi_train_dir "${ts_dir}/train" \
      --uhi_eval_dir "${ts_dir}/all" \
      --total_rounds "${armd_rounds}" \
      --base_lr 2e-5 \
      --batch_size 64 \
      --weight_decay 0.01 \
      --timesteps 12 \
      --sampling_timesteps 12 \
      --ema_decay 0.995 \
      --ema_update_interval 10 \
      --warmup_lr 2e-5 \
      --warmup_steps 500 \
      --feature_size 24 \
      --evaluation_horizon 1 \
      --sample_errors_dir "${OUTPUT_ROOT}/baselines/armd/sample_errors" \
      --efficiency_record_dir "${EFFICIENCY_RECORD_DIR}" \
      "${limit_args[@]}"
    _baseline_require_file "$(_baseline_global_armd_checkpoint)"
  fi

  if _baseline_has_model filternet; then
    cd "${ROOT}/baseline/filternet"
    run_logged baseline_filternet env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id UHI500 \
      --model PaiFilter \
      --data custom \
      --features M \
      --target Amod2 \
      --freq m \
      --seq_len 12 \
      --label_len 0 \
      --pred_len 1 \
      --enc_in 24 \
      --dec_in 24 \
      --c_out 24 \
      --embed_size 12 \
      --hidden_size 64 \
      --batch_size 1024 \
      --learning_rate 2e-5 \
      --weight_decay 0.05 \
      --dropout 0 \
      --train_epochs "${filternet_epochs}" \
      --city_passes 1 \
      --train_city_metrics_path "${train_cities}" \
      --test_city_metrics_path "${test_cities}" \
      --uhi_data_dir "${ts_dir}/all" \
      --checkpoints "${OUTPUT_ROOT}/baselines/filternet/checkpoints" \
      --results_dir "${OUTPUT_ROOT}/baselines/filternet" \
      --sample_errors_dir "${OUTPUT_ROOT}/baselines/filternet/sample_errors" \
      --output_tag manuscript_mv24 \
      --global_training \
      --evaluate_target_only \
      --efficiency_record_dir "${EFFICIENCY_RECORD_DIR}" \
      "${overwrite_args[@]}" \
      "${limit_args[@]}"
    _baseline_require_file "$(_baseline_global_filternet_checkpoint)"
  fi

  if _baseline_has_model stmtm; then
    stmtm_root="${OUTPUT_ROOT}/baselines/st_mtm"
    pretrain_root="${stmtm_root}/pretrain_checkpoints"
    cd "${ROOT}/baseline/st_mtm"

    run_logged baseline_st_mtm_pretrain env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" run.py \
      --task_name pretrain --is_training 1 --model_id UHI500 --model STMTM \
      --data Exchange --features M --target Amod2 --freq m \
      --seq_len 12 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 \
      --d_model 16 --d_ff 64 --d_hidden 64 --n_heads 4 --e_layers 1 \
      --dropout 0.2 --head_dropout 0.2 --fc_dropout 0 \
      --batch_size 256 --num_workers 0 --learning_rate 1.28e-5 --weight_decay 0.05 \
      --kernel_size 200 --seg_len 4 --p_tmask 0.2 --topk 3 --tau 0.1 --alpha 0.5 \
      --train_epochs "${stmtm_pretrain_epochs}" --city_passes 1 \
      --train_city_metrics_path "${train_cities}" \
      --test_city_metrics_path "${test_cities}" \
      --uhi_data_dir "${ts_dir}/all" \
      --pretrain_checkpoints "${pretrain_root}" \
      --output_tag manuscript --global_training \
      --efficiency_record_dir "${EFFICIENCY_RECORD_DIR}" \
      "${overwrite_args[@]}" \
      "${limit_args[@]}"

    pretrain_checkpoint="${pretrain_root}/Exchange/ckpt_best.pth"
    if [[ ! -f "${pretrain_checkpoint}" ]]; then
      echo "ST-MTM pretrain checkpoint was not created: ${pretrain_checkpoint}" >&2
      exit 1
    fi

    run_logged baseline_st_mtm_finetune env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" run.py \
      --task_name finetune --is_training 1 --model_id UHI500 --model STMTM \
      --data Exchange --features M --target Amod2 --freq m \
      --seq_len 12 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 \
      --d_model 16 --d_ff 64 --d_hidden 64 --n_heads 4 --e_layers 1 \
      --dropout 0.2 --head_dropout 0.2 --fc_dropout 0 \
      --batch_size 1024 --num_workers 0 --learning_rate 1.6e-5 --weight_decay 0.05 \
      --kernel_size 200 --seg_len 4 --p_tmask 0.2 --topk 3 --tau 0.1 --alpha 0.5 \
      --component_mode weighted --trend_weight 0.98 --head_init xavier \
      --train_epochs "${stmtm_finetune_epochs}" --city_passes 1 \
      --train_city_metrics_path "${train_cities}" \
      --test_city_metrics_path "${test_cities}" \
      --uhi_data_dir "${ts_dir}/all" \
      --checkpoints "${stmtm_root}/checkpoints" \
      --pretrain_checkpoints "${pretrain_root}" \
      --load_checkpoints "${pretrain_checkpoint}" \
      --results_dir "${stmtm_root}" \
      --sample_errors_dir "${stmtm_root}/sample_errors" \
      --output_tag manuscript --global_training \
      --target_only_metrics \
      --efficiency_record_dir "${EFFICIENCY_RECORD_DIR}" \
      "${overwrite_args[@]}" \
      "${limit_args[@]}"
    _baseline_require_file "$(_baseline_global_stmtm_checkpoint)"
  fi
)

run_calibration_timeseries_inference() (
  local test_cities="$1"
  local region="Calibration"
  local train_cities="${DATA_DIR}/city_metrics/city_metrics_train.csv"
  local ts_dir="${OUTPUT_ROOT}/regional/${region}/timeseries_data"
  local output_dir="${OUTPUT_ROOT}/regional/${region}/baselines"
  local armd_checkpoint filternet_checkpoint stmtm_checkpoint checkpoint
  local -a overwrite_args=()
  [[ "${OVERWRITE_OUTPUTS:-1}" == "1" ]] && overwrite_args+=(--overwrite_outputs)

  armd_checkpoint="$(_baseline_global_armd_checkpoint)"
  filternet_checkpoint="$(_baseline_global_filternet_checkpoint)"
  stmtm_checkpoint="$(_baseline_global_stmtm_checkpoint)"
  for checkpoint in "${armd_checkpoint}" "${filternet_checkpoint}" "${stmtm_checkpoint}"; do
    _baseline_require_file "${checkpoint}"
  done

  _baseline_prepare_timeseries_data "${ts_dir}" "${train_cities}" "${test_cities}"

  cd "${ROOT}/baseline/armd"
  run_logged calibration_armd_inference env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" main_new.py \
    --config_path Config/amod2.yaml \
    --save_dir "${OUTPUT_ROOT}/calibration/baselines/armd_runtime" \
    --gpu 0 \
    --skip_train \
    --load_ckpt_path "${armd_checkpoint}" \
    --train_city_metrics_path "${train_cities}" \
    --test_city_metrics_path "${test_cities}" \
    --uhi_train_dir "${ts_dir}/train" \
    --uhi_eval_dir "${ts_dir}/all" \
    --base_lr 2e-5 --batch_size 64 --weight_decay 0.01 \
    --timesteps 12 --sampling_timesteps 12 \
    --ema_decay 0.995 --ema_update_interval 10 \
    --warmup_lr 2e-5 --warmup_steps 500 \
    --feature_size 24 \
    --evaluation_horizon 1 \
    --sample_errors_dir "${output_dir}/armd/sample_errors" \
    --overwrite_outputs

  cd "${ROOT}/baseline/filternet"
  run_logged calibration_filternet_inference env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" run.py \
    --task_name long_term_forecast --is_training 0 --model_id UHI500 \
    --model PaiFilter --data custom --features M --target Amod2 --freq m \
    --seq_len 12 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 \
    --embed_size 12 --hidden_size 64 --batch_size 1024 --learning_rate 2e-5 \
    --weight_decay 0.05 --dropout 0 --train_epochs "${FILTERNET_EPOCHS:-5}" \
    --city_passes 1 \
    --train_city_metrics_path "${train_cities}" \
    --test_city_metrics_path "${test_cities}" \
    --uhi_data_dir "${ts_dir}/all" \
    --checkpoints "${OUTPUT_ROOT}/baselines/filternet/checkpoints" \
    --load_checkpoint_path "${filternet_checkpoint}" \
    --results_dir "${output_dir}/filternet" \
    --sample_errors_dir "${output_dir}/filternet/sample_errors" \
    --output_tag calibration_inference \
    --evaluate_target_only \
    "${overwrite_args[@]}"

  cd "${ROOT}/baseline/st_mtm"
  run_logged calibration_st_mtm_inference env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" run.py \
    --task_name finetune --is_training 0 --model_id UHI500 --model STMTM \
    --data Exchange --features M --target Amod2 --freq m \
    --seq_len 12 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 \
    --d_model 16 --d_ff 64 --d_hidden 64 --n_heads 4 --e_layers 1 \
    --dropout 0.2 --head_dropout 0.2 --fc_dropout 0 \
    --batch_size 1024 --num_workers 0 --learning_rate 1.6e-5 --weight_decay 0.05 \
    --kernel_size 200 --seg_len 4 --p_tmask 0.2 --topk 3 --tau 0.1 --alpha 0.5 \
    --component_mode weighted --trend_weight 0.98 --head_init xavier \
    --train_epochs "${STMTM_FINETUNE_EPOCHS:-5}" --city_passes 1 \
    --train_city_metrics_path "${train_cities}" \
    --test_city_metrics_path "${test_cities}" \
    --uhi_data_dir "${ts_dir}/all" \
    --checkpoints "${OUTPUT_ROOT}/baselines/st_mtm/checkpoints" \
    --pretrain_checkpoints "${OUTPUT_ROOT}/baselines/st_mtm/pretrain_checkpoints" \
    --load_checkpoints "${stmtm_checkpoint}" \
    --results_dir "${output_dir}/st_mtm" \
    --sample_errors_dir "${output_dir}/st_mtm/sample_errors" \
    --output_tag calibration_inference \
    --target_only_metrics \
    "${overwrite_args[@]}"
)

run_regional_timeseries_baselines() (
  local region="$1"
  local train_cities="$2"
  local test_cities="$3"
  local ts_dir="${OUTPUT_ROOT}/regional/${region}/timeseries_data"
  local output_dir="${OUTPUT_ROOT}/regional/${region}/baselines"
  local armd_rounds="${ARMD_ROUNDS:-5}"
  local filternet_epochs="${FILTERNET_EPOCHS:-5}"
  local stmtm_pretrain_epochs="${STMTM_PRETRAIN_EPOCHS:-5}"
  local stmtm_finetune_epochs="${STMTM_FINETUNE_EPOCHS:-5}"
  local stmtm_root="${output_dir}/st_mtm"
  local pretrain_root="${output_dir}/st_mtm/pretrain_checkpoints"
  local pretrain_checkpoint
  local -a overwrite_args=()
  [[ "${OVERWRITE_OUTPUTS:-1}" == "1" ]] && overwrite_args+=(--overwrite_outputs)

  _baseline_prepare_timeseries_data "${ts_dir}" "${train_cities}" "${test_cities}"

  cd "${ROOT}/baseline/armd"
  run_logged "regional_armd_${region}" env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" main_new.py \
    --config_path Config/amod2.yaml --save_dir "${output_dir}/armd" --gpu 0 \
    --train_city_metrics_path "${train_cities}" --test_city_metrics_path "${test_cities}" \
    --uhi_train_dir "${ts_dir}/train" --uhi_eval_dir "${ts_dir}/all" \
    --total_rounds "${armd_rounds}" --base_lr 2e-5 --batch_size 64 --weight_decay 0.01 \
    --timesteps 12 --sampling_timesteps 12 --ema_decay 0.995 --ema_update_interval 10 \
    --warmup_lr 2e-5 --warmup_steps 500 --feature_size 24 \
    --evaluation_horizon 1 \
    --sample_errors_dir "${output_dir}/armd/sample_errors"

  cd "${ROOT}/baseline/filternet"
  run_logged "regional_filternet_${region}" env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" run.py \
    --task_name long_term_forecast --is_training 1 --model_id "UHI_${region}" \
    --model PaiFilter --data custom --features M --target Amod2 --freq m \
    --seq_len 12 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 \
    --embed_size 12 --hidden_size 64 --batch_size 1024 --learning_rate 2e-5 \
    --weight_decay 0.05 --dropout 0 --train_epochs "${filternet_epochs}" --city_passes 1 \
    --train_city_metrics_path "${train_cities}" --test_city_metrics_path "${test_cities}" \
    --uhi_data_dir "${ts_dir}/all" \
    --checkpoints "${output_dir}/filternet/checkpoints" \
    --results_dir "${output_dir}/filternet" \
    --sample_errors_dir "${output_dir}/filternet/sample_errors" \
    --output_tag "${region}" --global_training \
    --evaluate_target_only \
    "${overwrite_args[@]}"

  cd "${ROOT}/baseline/st_mtm"
  run_logged "regional_st_mtm_pretrain_${region}" env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" run.py \
    --task_name pretrain --is_training 1 --model_id "UHI_${region}" --model STMTM \
    --data Exchange --features M --target Amod2 --freq m \
    --seq_len 12 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 \
    --d_model 16 --d_ff 64 --d_hidden 64 --n_heads 4 --e_layers 1 \
    --dropout 0.2 --head_dropout 0.2 --fc_dropout 0 \
    --batch_size 256 --num_workers 0 --learning_rate 1.28e-5 --weight_decay 0.05 \
    --kernel_size 200 --seg_len 4 --p_tmask 0.2 --topk 3 --tau 0.1 --alpha 0.5 \
    --train_epochs "${stmtm_pretrain_epochs}" --city_passes 1 \
    --train_city_metrics_path "${train_cities}" --test_city_metrics_path "${test_cities}" \
    --uhi_data_dir "${ts_dir}/all" \
    --pretrain_checkpoints "${pretrain_root}" \
    --output_tag "${region}" --global_training "${overwrite_args[@]}"

  pretrain_checkpoint="${pretrain_root}/Exchange/ckpt_best.pth"
  if [[ ! -f "${pretrain_checkpoint}" ]]; then
    echo "ST-MTM pretrain checkpoint was not created: ${pretrain_checkpoint}" >&2
    exit 1
  fi

  run_logged "regional_st_mtm_finetune_${region}" env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" run.py \
    --task_name finetune --is_training 1 --model_id "UHI_${region}" --model STMTM \
    --data Exchange --features M --target Amod2 --freq m \
    --seq_len 12 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 \
    --d_model 16 --d_ff 64 --d_hidden 64 --n_heads 4 --e_layers 1 \
    --dropout 0.2 --head_dropout 0.2 --fc_dropout 0 \
    --batch_size 1024 --num_workers 0 --learning_rate 1.6e-5 --weight_decay 0.05 \
    --kernel_size 200 --seg_len 4 --p_tmask 0.2 --topk 3 --tau 0.1 --alpha 0.5 \
    --component_mode weighted --trend_weight 0.98 --head_init xavier \
    --train_epochs "${stmtm_finetune_epochs}" --city_passes 1 \
    --train_city_metrics_path "${train_cities}" --test_city_metrics_path "${test_cities}" \
    --uhi_data_dir "${ts_dir}/all" \
    --checkpoints "${stmtm_root}/checkpoints" \
    --pretrain_checkpoints "${pretrain_root}" --load_checkpoints "${pretrain_checkpoint}" \
    --results_dir "${stmtm_root}" --sample_errors_dir "${stmtm_root}/sample_errors" \
    --output_tag "${region}" --global_training \
    --target_only_metrics \
    "${overwrite_args[@]}"
)

run_all_global_baselines() {
  run_global_tabular_baselines
  run_global_timeseries_baselines
}

run_all_regional_baselines() {
  local region="$1"
  local train_cities="$2"
  local test_cities="$3"
  run_regional_tabular_baselines "${region}"
  run_regional_timeseries_baselines "${region}" "${train_cities}" "${test_cities}"
}
