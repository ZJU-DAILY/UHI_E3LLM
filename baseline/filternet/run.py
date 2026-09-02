import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np
import pandas as pd
import shutil
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from efficiency import synchronize_cuda, write_measurement

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='ETTm1', help='model id')
    parser.add_argument('--model', type=str, default='PaiFilter', help='model name, options: [PaiFilter, TexFilter]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../../dataset/ETT-small', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='1', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--evaluate_target_only', action='store_true',
                        help='Train all requested channels but report test metrics only for the final target channel.')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=2, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='1: channel dependence 0: channel independence for FreTS model')

    # FilterNet
    parser.add_argument('--embed_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--dropout', type=float, default=0, help='dropout')

    # UHI regional CV overrides. Defaults keep the original global split behavior.
    parser.add_argument('--train_city_metrics_path', type=str,
                        default='../../nature/Final_dataset/city_metrics/city_metrics_train.csv',
                        help='CSV containing training city ids for iterative UHI training.')
    parser.add_argument('--test_city_metrics_path', type=str,
                        default='../../nature/Final_dataset/city_metrics/city_metrics_test.csv',
                        help='CSV containing test city ids for UHI evaluation.')
    parser.add_argument('--uhi_data_dir', type=str, default='../Extracted_City_Data',
                        help='Directory containing UHI_city_<id>.csv files.')
    parser.add_argument('--output_tag', type=str, default='',
                        help='Optional suffix added to checkpoint/result folders, e.g. cv_Africa.')
    parser.add_argument('--train_iterative', action='store_true',
                        help='Train iteratively over train_city_metrics_path before testing.')
    parser.add_argument('--global_training', action='store_true',
                        help='Train on a shuffled concatenation of all city datasets.')
    parser.add_argument('--overwrite_outputs', action='store_true',
                        help='Remove this setting checkpoint/results/test_results folders before running.')
    parser.add_argument('--load_checkpoint_path', type=str, default=None,
                        help='Optional explicit checkpoint used for evaluation.')
    parser.add_argument('--sample_errors_dir', type=str, default=None,
                        help='Optional directory for per-city sample-level prediction errors.')
    parser.add_argument('--evaluation_start', type=int, default=201801,
                        help='First target month to evaluate, formatted as YYYYMM.')
    parser.add_argument('--evaluation_end', type=int, default=202012,
                        help='Last target month to evaluate, formatted as YYYYMM.')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory for aggregate evaluation-period metrics.')
    parser.add_argument('--max_train_cities', type=int, default=None,
                        help='Optional training-city limit for smoke tests.')
    parser.add_argument('--max_test_cities', type=int, default=None,
                        help='Optional test-city limit for smoke tests.')
    parser.add_argument('--efficiency_record_dir', type=str, default='')
    parser.add_argument('--efficiency_batch_size', type=int, default=36)
    parser.add_argument('--efficiency_warmup_calls', type=int, default=5)
    parser.add_argument('--efficiency_measured_calls', type=int, default=5)


    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1,
                        help='Internal epochs for each city visit.')
    parser.add_argument('--city_passes', type=int, default=15,
                        help='Complete round-robin passes over the training cities.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Adam weight decay')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    parser.add_argument('--seg_len', type=int, default=48, help='segment length of time series')


    args = parser.parse_args()
    args.uhi_data_dir = os.path.abspath(args.uhi_data_dir)
    args.use_gpu = torch.cuda.is_available()
    args.use_multi_gpu = False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.data == 'custom' and args.freq == 'h':
        args.freq = 'm'

    Exp = Exp_Long_Term_Forecast

    cities_df = pd.read_csv(args.train_city_metrics_path)
    city_id_list = cities_df['id'].astype(int).tolist()
    if args.max_train_cities is not None:
        if args.max_train_cities < 1:
            raise ValueError('--max_train_cities must be positive')
        city_id_list = city_id_list[:args.max_train_cities]
    dataset_paths = [os.path.join(args.uhi_data_dir, f"UHI_city_{city_id}.csv") for city_id in city_id_list]
    missing_paths = [path for path in dataset_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(f'Missing training series: {missing_paths[:3]}')
    print(f"Found {len(city_id_list)} training cities.")

    NUM_PASSES = args.city_passes

    exp = Exp(args)
    setting = 'UHI{}City{}passes_{}epochs_{}_{}_sl{}_pl{}_embed{}_hidden{}_bs{}_lr{}_drop{}'.format(
        len(city_id_list),
        NUM_PASSES,
        args.train_epochs,
        args.model,
        args.data,
        args.seq_len,
        args.pred_len,
        args.embed_size,
        args.hidden_size,
        args.batch_size,
        args.learning_rate,
        args.dropout
    )
    if args.output_tag:
        setting = '{}_{}'.format(setting, args.output_tag)
    if args.overwrite_outputs:
        for folder in [
            os.path.join(args.checkpoints, setting),
            os.path.join('./results', setting),
            os.path.join('./test_results', setting),
        ]:
            if os.path.exists(folder):
                print(f'Removing existing output folder: {folder}')
                shutil.rmtree(folder)
        for folder in [args.results_dir, args.sample_errors_dir]:
            if folder and os.path.exists(folder):
                print(f'Removing existing output folder: {folder}')
                shutil.rmtree(folder)
    print('>>>>>> Created single model instance for iterative training: {} <<<<<<'.format(setting))

    did_train = args.global_training or args.train_iterative
    if did_train:
        synchronize_cuda()
        training_started = time.perf_counter()
    if args.global_training:
        exp.args.global_data_paths = dataset_paths
        exp.args.data_path = dataset_paths[0]
        exp.train(setting)
    elif args.train_iterative:
        for city_pass in range(1, NUM_PASSES + 1):
            print(f'\n================ City pass {city_pass}/{NUM_PASSES} ================')
            for idx, data_file in enumerate(dataset_paths, 1):
                exp.args.data_path = data_file
                print(f'[Pass {city_pass}] Training on ({idx}/{len(dataset_paths)}): {data_file}')
                exp.train(setting)
                torch.cuda.empty_cache()
    else:
        print('train_iterative is disabled; testing will load an existing checkpoint if present.')
    if did_train:
        synchronize_cuda()
        training_seconds = time.perf_counter() - training_started
        if args.efficiency_record_dir:
            write_measurement(
                os.path.join(args.efficiency_record_dir, 'training_filternet.json'),
                kind='training', method='FilterNet', stage='train',
                value=training_seconds, unit='seconds',
                note='optimization, in-loop validation, and checkpoint writes; excludes data/model loading and test inference',
                metadata={
                    'epochs': args.train_epochs,
                    'training_cities': len(city_id_list),
                    'batch_size': args.batch_size,
                },
            )

    test_df = pd.read_csv(args.test_city_metrics_path)
    test_city_id_list = test_df['id'].astype(int).tolist()
    if args.max_test_cities is not None:
        if args.max_test_cities < 1:
            raise ValueError('--max_test_cities must be positive')
        test_city_id_list = test_city_id_list[:args.max_test_cities]
    test_dataset_paths = [os.path.join(args.uhi_data_dir, f"UHI_city_{city_id}.csv") for city_id in test_city_id_list]
    missing_paths = [path for path in test_dataset_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(f'Missing test series: {missing_paths[:3]}')
    print(f"\nFound {len(test_city_id_list)} test cities.")

    load = 0
    aggregate_year_abs = {}
    aggregate_year_sq = {}
    aggregate_year_count = {}
    for idx, data_file in enumerate(test_dataset_paths, 1):
        exp.args.data_path = data_file
        print(f'[Test] ({idx}/{len(test_dataset_paths)}) Testing on: {data_file}')
        if load == 0:
            ya, ys, yc = exp.test(setting, test=1)
            load = 1
        else:
            ya, ys, yc = exp.test(setting, test=1, load=1)
        for k, v in ya.items():
            aggregate_year_abs[k] = aggregate_year_abs.get(k, 0.0) + v
        for k, v in ys.items():
            aggregate_year_sq[k] = aggregate_year_sq.get(k, 0.0) + v
        for k, v in yc.items():
            aggregate_year_count[k] = aggregate_year_count.get(k, 0) + v
        torch.cuda.empty_cache()

    if aggregate_year_count:
        results_folder = args.results_dir or os.path.join('./test_results', setting)
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        yearly_all_path = os.path.join(results_folder, 'yearly_metrics_all_cities.csv')
        with open(yearly_all_path, 'w') as f:
            f.write('year,mae,mse,rmse,num_samples\n')
            expected_years = sorted(aggregate_year_count.keys())
            for year in expected_years:
                mae_year = aggregate_year_abs[year] / aggregate_year_count[year]
                mse_year = aggregate_year_sq[year] / aggregate_year_count[year]
                f.write(
                    f"{year},{mae_year:.6f},{mse_year:.6f},"
                    f"{np.sqrt(mse_year):.6f},{aggregate_year_count[year]}\n"
                )
        print(f'Saved aggregated yearly metrics to {yearly_all_path}')

        num_samples = sum(aggregate_year_count.values())
        start_year, start_month = divmod(args.evaluation_start, 100)
        end_year, end_month = divmod(args.evaluation_end, 100)
        expected_per_city = (
            (end_year - start_year) * 12 + end_month - start_month + 1
        )
        if expected_per_city <= 0:
            raise ValueError('evaluation_end must not precede evaluation_start')
        expected_samples = expected_per_city * len(test_dataset_paths)
        if num_samples != expected_samples:
            raise RuntimeError(
                f'Expected {expected_samples} test samples '
                f'({expected_per_city} per city for '
                f'{args.evaluation_start}-{args.evaluation_end}), '
                f'but evaluated {num_samples}'
            )
        mae = sum(aggregate_year_abs.values()) / num_samples
        mse = sum(aggregate_year_sq.values()) / num_samples
        summary = pd.DataFrame([{
            'model': 'FilterNet',
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'num_cities': len(test_dataset_paths),
            'num_samples': num_samples,
            'evaluation_start': args.evaluation_start,
            'evaluation_end': args.evaluation_end,
            'training_loss_channels': 'all' if args.features == 'M' else 'target',
            'evaluation_channels': 'target_only' if args.evaluate_target_only else args.features,
        }])
        summary_path = os.path.join(results_folder, 'summary.csv')
        summary.to_csv(summary_path, index=False)
        print(summary.to_string(index=False))
        print(f'Saved aggregate metrics to {summary_path}')
