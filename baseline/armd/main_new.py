import os
import torch
import numpy as np
import random
import argparse
import warnings
import copy
import shutil
from collections import defaultdict

warnings.filterwarnings("ignore")

from engine.solver import Trainer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
try:
    from gluonts.dataset.repository.datasets import get_dataset
    from gluonts.dataset.multivariate_grouper import MultivariateGrouper
except ModuleNotFoundError:
    get_dataset = None
    MultivariateGrouper = None
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.autoregressive_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Data.build_dataloader import build_dataloader, build_dataloader_cond

import pandas as pd
import re
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from efficiency import synchronize_cuda, write_measurement

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(2023)


def inclusive_month_count(start_period, end_period):
    start_year, start_month = divmod(int(start_period), 100)
    end_year, end_month = divmod(int(end_period), 100)
    if not 1 <= start_month <= 12 or not 1 <= end_month <= 12:
        raise ValueError('Evaluation periods must use YYYYMM with months in 01..12')
    count = (end_year - start_year) * 12 + end_month - start_month + 1
    if count <= 0:
        raise ValueError('Evaluation end must not precede evaluation start')
    return count

class Args_Example:
    def __init__(self, config_path, save_dir, gpu):
        self.config_path = config_path
        self.save_dir = save_dir
        self.gpu = gpu
        os.makedirs(self.save_dir, exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process configuration and directories.")
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the configuration file.')
    parser.add_argument('--save_dir', type=str, default='./forecasting_exp_round_robin_test',
                        help='Directory to save the final round-robin trained model.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Specify which GPU to use.')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip all training rounds and directly evaluate.')
    parser.add_argument('--load_ckpt_path', type=str, default=None,
                        help='Optional checkpoint path to load before evaluation (when skipping training).')
    parser.add_argument('--train_city_metrics_path', type=str,
                        default='../../nature/Final_dataset/city_metrics/city_metrics_train.csv',
                        help='CSV containing training city ids for regional UHI training.')
    parser.add_argument('--test_city_metrics_path', type=str,
                        default='../../nature/Final_dataset/city_metrics/city_metrics_test.csv',
                        help='CSV containing test city ids for regional UHI evaluation.')
    parser.add_argument('--uhi_train_dir', type=str, default='../Extracted_City_Data/train',
                        help='Directory containing training UHI_city_<id>.csv files.')
    parser.add_argument('--uhi_eval_dir', type=str, default='../Extracted_City_Data',
                        help='Directory containing evaluation UHI_city_<id>.csv files.')
    parser.add_argument('--total_rounds', type=int, default=100,
                        help='Number of round-robin training passes over the training city list.')
    parser.add_argument('--sample_errors_dir', type=str, default=None,
                        help='Optional directory for per-city sample-level prediction errors.')
    parser.add_argument('--overwrite_outputs', action='store_true',
                        help='Replace the requested sample-level evaluation output directory.')
    parser.add_argument('--max_train_cities', type=int, default=None,
                        help='Optional training-city limit for smoke tests.')
    parser.add_argument('--max_test_cities', type=int, default=None,
                        help='Optional test-city limit for smoke tests.')
    parser.add_argument('--evaluation_horizon', type=int, default=1,
                        help='Number of leading forecast horizons to evaluate; use 1 for next-month UHI.')
    parser.add_argument('--evaluation_start', type=int, default=201801,
                        help='First target month to evaluate, formatted as YYYYMM.')
    parser.add_argument('--evaluation_end', type=int, default=202012,
                        help='Last target month to evaluate, formatted as YYYYMM.')
    parser.add_argument('--base_lr', type=float, default=None,
                        help='Optional override for solver.base_lr.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Optional override for dataloader.batch_size.')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Optional Adam weight decay override.')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Optional diffusion-step override.')
    parser.add_argument('--sampling_timesteps', type=int, default=None,
                        help='Optional sampling-step override.')
    parser.add_argument('--feature_size', type=int, default=None,
                        help='Optional multivariate feature-count override.')
    parser.add_argument('--ema_decay', type=float, default=None,
                        help='Optional EMA decay override.')
    parser.add_argument('--ema_update_interval', type=int, default=None,
                        help='Optional EMA update interval override.')
    parser.add_argument('--warmup_lr', type=float, default=None,
                        help='Optional scheduler warmup learning-rate override.')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Optional scheduler warmup-step override.')
    parser.add_argument('--efficiency_record_dir', type=str, default='')
    parser.add_argument('--efficiency_batch_size', type=int, default=36)
    parser.add_argument('--efficiency_warmup_calls', type=int, default=5)
    parser.add_argument('--efficiency_measured_calls', type=int, default=5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse arguments and load base configuration
    args_parsed = parse_arguments()
    expected_months_per_city = inclusive_month_count(
        args_parsed.evaluation_start, args_parsed.evaluation_end
    )
    if (
        args_parsed.overwrite_outputs
        and args_parsed.sample_errors_dir
        and os.path.exists(args_parsed.sample_errors_dir)
    ):
        shutil.rmtree(args_parsed.sample_errors_dir)
    args = Args_Example(args_parsed.config_path, args_parsed.save_dir, args_parsed.gpu)
    args.efficiency_record_dir = args_parsed.efficiency_record_dir
    args.efficiency_batch_size = args_parsed.efficiency_batch_size
    args.efficiency_warmup_calls = args_parsed.efficiency_warmup_calls
    args.efficiency_measured_calls = args_parsed.efficiency_measured_calls
    configs = load_yaml_config(args.config_path)
    if args_parsed.base_lr is not None:
        configs['solver']['base_lr'] = args_parsed.base_lr
    if args_parsed.batch_size is not None:
        configs['dataloader']['batch_size'] = args_parsed.batch_size
    if args_parsed.weight_decay is not None:
        configs['solver']['weight_decay'] = args_parsed.weight_decay
    if args_parsed.timesteps is not None:
        configs['model']['params']['timesteps'] = args_parsed.timesteps
    if args_parsed.sampling_timesteps is not None:
        configs['model']['params']['sampling_timesteps'] = args_parsed.sampling_timesteps
    if args_parsed.feature_size is not None:
        configs['model']['params']['feature_size'] = args_parsed.feature_size
    if args_parsed.ema_decay is not None:
        configs['solver']['ema']['decay'] = args_parsed.ema_decay
    if args_parsed.ema_update_interval is not None:
        configs['solver']['ema']['update_interval'] = args_parsed.ema_update_interval
    if args_parsed.warmup_lr is not None:
        configs['solver']['scheduler']['params']['warmup_lr'] = args_parsed.warmup_lr
    if args_parsed.warmup_steps is not None:
        configs['solver']['scheduler']['params']['warmup'] = args_parsed.warmup_steps

    cities_df = pd.read_csv(args_parsed.train_city_metrics_path)
    city_id_list = cities_df['id'].astype(int).tolist()
    if args_parsed.max_train_cities is not None:
        if args_parsed.max_train_cities < 1:
            raise ValueError('--max_train_cities must be positive')
        city_id_list = city_id_list[:args_parsed.max_train_cities]
    dataset_paths = [os.path.join(args_parsed.uhi_train_dir, f"UHI_city_{city_id}.csv") for city_id in city_id_list]
    missing_paths = [path for path in dataset_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(f'Missing training series: {missing_paths[:3]}')
    print(f"Found {len(city_id_list)} cities to train.")

    TOTAL_ROUNDS = args_parsed.total_rounds

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Use GPU: physical cuda:{args.gpu} mapped to cuda:0")
    else:
        device = torch.device('cpu')
        print("Use CPU")
    model = instantiate_from_config(configs['model']).to(device)
    model.fast_sampling = True
    print("Model instantiated. Starting round-robin training...")

    run_configs = copy.deepcopy(configs)
    run_configs['dataloader']['train_dataset']['params']['data_root'] = dataset_paths[0]
    run_configs['solver']['results_folder'] = os.path.join(args.save_dir, 'checkpoints_mv24')
    run_configs['solver']['max_epochs'] = 1

    dataloader_info = build_dataloader(run_configs, args)
    dataloader = dataloader_info['dataloader']

    trainer = Trainer(config=run_configs, args=args, model=model, dataloader={'dataloader':dataloader})

    if args_parsed.skip_train:
        print("skip_train enabled: bypassing training loops.")
        if args_parsed.load_ckpt_path and os.path.exists(args_parsed.load_ckpt_path):
            print(f"Loading checkpoint from {args_parsed.load_ckpt_path}")
            ckpt = torch.load(args_parsed.load_ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get('model', ckpt))
            if 'ema' in ckpt:
                try:
                    trainer.ema.load_state_dict(ckpt['ema'])
                except Exception as e:
                    print(f"Warning: failed to load EMA state ({e}); using model weights only.")
        else:
            raise FileNotFoundError('--skip_train requires an existing --load_ckpt_path')
    else:
        synchronize_cuda()
        training_started = time.perf_counter()
        for round_num in range(1, TOTAL_ROUNDS + 1):
            print(f"\n{'='*30} STARTING ROUND {round_num}/{TOTAL_ROUNDS} {'='*30}")

            for dataset_path in dataset_paths:
                print(f"\n--- Round {round_num} | Training on: {dataset_path} ---")

                run_configs = copy.deepcopy(configs)
                run_configs['dataloader']['train_dataset']['params']['data_root'] = dataset_path
                run_configs['solver']['results_folder'] = os.path.join(args.save_dir, 'checkpoints_mv24')
                run_configs['solver']['max_epochs'] = 1

                dataloader_info = build_dataloader(run_configs, args)
                dataloader = dataloader_info['dataloader']

                trainer.set_dataloader(dataloader)

                # One round means one complete pass over every window from the
                # current city, while optimizer/scheduler/EMA state persists.
                trainer.train(num_steps=len(dataloader))

        trainer.save('final', verbose=True)
        synchronize_cuda()
        training_seconds = time.perf_counter() - training_started
        if args_parsed.efficiency_record_dir:
            write_measurement(
                os.path.join(args_parsed.efficiency_record_dir, 'training_armd.json'),
                kind='training', method='ARMD', stage='train',
                value=training_seconds, unit='seconds',
                note='optimization and final checkpoint write; excludes data/model loading and test inference',
                metadata={
                    'rounds': TOTAL_ROUNDS,
                    'training_cities': len(dataset_paths),
                    'batch_size': configs['dataloader']['batch_size'],
                },
            )

        print(f"\n{'='*60}")
        print("ALL TRAINING ROUNDS COMPLETED.")
        print(f"The final model is saved in: {args.save_dir}")
        print(f"{'='*60}\n")


    print("Starting evaluation of the final model on all test sets...")

    seq_len = configs['model']['params']['seq_length']
    feat_num = configs['model']['params']['feature_size']

    results = []
    yearly_results = []

    cities_df = pd.read_csv(args_parsed.test_city_metrics_path)
    city_id_list = cities_df['id'].astype(int).tolist()
    if args_parsed.max_test_cities is not None:
        if args_parsed.max_test_cities < 1:
            raise ValueError('--max_test_cities must be positive')
        city_id_list = city_id_list[:args_parsed.max_test_cities]
    dataset_paths = [os.path.join(args_parsed.uhi_eval_dir, f"UHI_city_{city_id}.csv") for city_id in city_id_list]
    missing_paths = [path for path in dataset_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(f'Missing test series: {missing_paths[:3]}')

    all_abs_errors = []
    all_sq_errors = []

    for dataset_path in dataset_paths:
        print(f"\n--- Evaluating on: {dataset_path} ---")

        eval_configs = copy.deepcopy(configs)
        eval_configs['dataloader']['test_dataset']['params']['data_root'] = dataset_path

        args.mode = 'predict'
        args.pred_len = int(configs['model']['params'].get('forecast_horizon', seq_len))

        test_dataloader_info = build_dataloader_cond(eval_configs, args)
        test_dataloader = test_dataloader_info['dataloader']
        test_dataset = test_dataloader_info['dataset']

        sample, real_, ts_targets = trainer.sample_forecast(test_dataloader, shape=[seq_len, feat_num])


        sample_arr = np.asarray(sample)
        real_arr = np.asarray(real_)
        sample_arr = test_dataset.scaler.inverse_transform(
            sample_arr.reshape(-1, feat_num)
        ).reshape(sample_arr.shape)
        real_arr = test_dataset.scaler.inverse_transform(
            real_arr.reshape(-1, feat_num)
        ).reshape(real_arr.shape)

        evaluation_horizon = args_parsed.evaluation_horizon
        native_horizon = getattr(args, 'pred_len', seq_len)
        if not 1 <= evaluation_horizon <= native_horizon:
            raise ValueError(
                f'--evaluation_horizon must be between 1 and {native_horizon}'
            )
        pred_target = sample_arr[:, :evaluation_horizon, -1]
        true_target = real_arr[:, :evaluation_horizon, -1]
        if ts_targets is not None:
            years_arr = np.asarray(ts_targets)
            if years_arr.ndim == 3:
                years = years_arr[:, :evaluation_horizon, 0]
            else:
                years = years_arr[:, :evaluation_horizon]
        else:
            years = np.tile(np.arange(pred_target.shape[1]), (pred_target.shape[0], 1))

        min_len = min(pred_target.shape[1], true_target.shape[1], years.shape[1])
        pred_target = pred_target[:, :min_len]
        true_target = true_target[:, :min_len]
        years = years[:, :min_len]

        prediction_frame = pd.DataFrame({
            'target_year_month': years.reshape(-1).astype(int),
            'pred': pred_target.reshape(-1),
            'true': true_target.reshape(-1),
        })
        prediction_frame = (
            prediction_frame.groupby('target_year_month', as_index=False)
            .agg(pred=('pred', 'mean'), true=('true', 'mean'))
        )
        prediction_frame = prediction_frame[
            prediction_frame['target_year_month'].between(
                args_parsed.evaluation_start, args_parsed.evaluation_end
            )
        ].copy()
        pred_eval = prediction_frame['pred'].to_numpy()
        true_eval = prediction_frame['true'].to_numpy()
        abs_err = np.abs(pred_eval - true_eval)
        sq_err = (pred_eval - true_eval) ** 2
        all_abs_errors.extend(abs_err.tolist())
        all_sq_errors.extend(sq_err.tolist())
        year_abs = defaultdict(float)
        year_sq = defaultdict(float)
        year_cnt = defaultdict(int)
        for period, abs_value, sq_value in zip(
            prediction_frame['target_year_month'], abs_err, sq_err
        ):
            year = int(period) // 100
            year_abs[year] += abs_value
            year_sq[year] += sq_value
            year_cnt[year] += 1

        mse = float(sq_err.mean())
        mae = float(abs_err.mean())

        city_name = os.path.splitext(os.path.basename(dataset_path))[0]
        match = re.search(r'\d+', city_name)
        city_id = int(match.group()) if match else city_name

        rmse = float(np.sqrt(mse))
        print(f"  - Results for city {city_id} -> MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

        if args_parsed.sample_errors_dir:
            os.makedirs(args_parsed.sample_errors_dir, exist_ok=True)
            sample_df = pd.DataFrame({
                'city_id': city_id,
                'sample_index': np.arange(len(pred_eval)),
                'target_year_month': prediction_frame['target_year_month'].to_numpy(),
                'true': true_eval,
                'pred': pred_eval,
                'error': pred_eval - true_eval,
                'abs_error': abs_err,
                'sq_error': sq_err,
            })
            sample_df.to_csv(
                os.path.join(
                    args_parsed.sample_errors_dir,
                    f'city_{city_id}.csv',
                ),
                index=False,
            )

        results.append({
            'city_id': city_id,
            'mse': float(mse),
            'mae': float(mae),
            'rmse': rmse,
            'num_samples': len(pred_eval),
        })
        for y in sorted(year_cnt.keys()):
            yearly_results.append({
                'city_id': city_id,
                'year': y,
                'mae': year_abs[y] / year_cnt[y],
                'mse': year_sq[y] / year_cnt[y]
            })

    if results:
        expected_samples = expected_months_per_city * len(dataset_paths)
        if len(all_abs_errors) != expected_samples:
            raise RuntimeError(
                f'Expected {expected_samples} test samples '
                f'({expected_months_per_city} per city for '
                f'{args_parsed.evaluation_start}-{args_parsed.evaluation_end}), '
                f'but evaluated {len(all_abs_errors)}'
            )
        overall_mae = float(np.mean(all_abs_errors))
        overall_mse = float(np.mean(all_sq_errors))
        overall_rmse = float(np.sqrt(overall_mse))
        print(
            f"\nOverall {args_parsed.evaluation_start}-{args_parsed.evaluation_end} "
            f"metrics -> MAE: {overall_mae:.6f}, "
            f"MSE: {overall_mse:.6f}, RMSE: {overall_rmse:.6f}"
        )
    else:
        print("\nNo evaluation results are available for aggregate metrics.")

    # Save city-level results
    results_df = pd.DataFrame(results)
    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, 'evaluation_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved evaluation results to: {csv_path}")

    # Save yearly metrics per city
    if yearly_results:
        yearly_df = pd.DataFrame(yearly_results)
        yearly_csv = os.path.join(args.save_dir, 'evaluation_yearly_metrics.csv')
        yearly_df.to_csv(yearly_csv, index=False)
        print(f"Saved yearly metrics to: {yearly_csv}")

    if results:
        summary_df = pd.DataFrame([{
            'model': 'ARMD',
            'mae': overall_mae,
            'mse': overall_mse,
            'rmse': overall_rmse,
            'num_cities': len(dataset_paths),
            'num_samples': len(all_abs_errors),
            'evaluation_start': args_parsed.evaluation_start,
            'evaluation_end': args_parsed.evaluation_end,
            'evaluation_horizon': args_parsed.evaluation_horizon,
        }])
        summary_path = os.path.join(args.save_dir, 'summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(summary_df.to_string(index=False))
        print(f"Saved aggregate metrics to: {summary_path}")
