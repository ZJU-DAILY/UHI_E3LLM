import argparse
import torch
from exp.exp_stmtm import Exp_STMTM
import random
import numpy as np
import os
import pandas as pd
from collections import defaultdict
import shutil
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from efficiency import synchronize_cuda, write_measurement
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

#%%
parser = argparse.ArgumentParser(description='TSMTM')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='pretrain', help='task name, options:[pretrain, finetune]')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='STMTM', help='model id')
parser.add_argument('--model', type=str, required=True, default='STMTM', help='model name')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='datasets/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./outputs/checkpoints/', help='location of model fine-tuning checkpoints')
parser.add_argument('--pretrain_checkpoints', type=str, default='./outputs/pretrain_checkpoints/', help='location of model pre-training checkpoints')
parser.add_argument('--transfer_checkpoints', type=str, default='ckpt_best.pth', help='checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]')
parser.add_argument('--load_checkpoints', type=str, default=None, help='location of model checkpoints')

# UHI data paths and regional evaluation options.
parser.add_argument('--train_city_metrics_path', type=str,
                    default='../../nature/Final_dataset/city_metrics/city_metrics_train.csv',
                    help='CSV containing training city ids for iterative UHI training.')
parser.add_argument('--test_city_metrics_path', type=str,
                    default='../../nature/Final_dataset/city_metrics/city_metrics_test.csv',
                    help='CSV containing test city ids for UHI evaluation.')
parser.add_argument('--uhi_data_dir', type=str, default='../../Extracted_City_Data',
                    help='Directory containing UHI_city_<id>.csv files. Absolute paths are recommended.')
parser.add_argument('--output_tag', type=str, default='',
                    help='Optional suffix added to checkpoint/result folders, e.g. cv_Africa.')
parser.add_argument('--train_iterative', action='store_true',
                    help='For finetune mode, train iteratively over train_city_metrics_path before testing.')
parser.add_argument('--global_training', action='store_true',
                    help='Train on a shuffled concatenation of all city datasets.')
parser.add_argument('--overwrite_outputs', action='store_true',
                    help='Remove this setting checkpoint/test output folders before running.')
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

# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model') #
parser.add_argument('--n_heads', type=int, default=8, help='num of heads') #
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers') #
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn') #
parser.add_argument('--d_hidden', type=int, default=128, help='dimension of freq-fc')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout') #
parser.add_argument('--fc_dropout', type=float, default=0, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout') #
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]') #
parser.add_argument('--activation', type=str, default='gelu', help='activation') #
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder') #
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--target_only_metrics', action='store_true',
                    help='Train all channels but evaluate only the final target channel.')
parser.add_argument('--freeze_backbone', action='store_true',
                    help='Freeze embeddings and encoder; train only the prediction head.')
parser.add_argument('--disable_layer_norm', action='store_true',
                    help='Replace LayerNorm modules in the trend encoder with identities.')
parser.add_argument('--component_mode', choices=['gated', 'average', 'weighted', 'trend', 'seasonal'], default='gated',
                    help='Select learned gating, fixed fusion, or one native ST-MTM branch.')
parser.add_argument('--trend_weight', type=float, default=0.5,
                    help='Trend weight used by component_mode=weighted; seasonal weight is 1-w.')
parser.add_argument('--head_init', choices=['default', 'xavier', 'kaiming'], default='default',
                    help='Initialization scheme for the native linear prediction head.')

# optimization
parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers') #
parser.add_argument('--itr', type=int, default=1, help='experiments times') #
parser.add_argument('--train_epochs', type=int, default=1,
                    help='Internal epochs for each city visit.')
parser.add_argument('--city_passes', type=int, default=10,
                    help='Complete round-robin passes over the training cities.')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data') #
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate') #
parser.add_argument('--weight_decay', type=float, default=0.0, help='Adam weight decay')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience') #
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate') #
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu') #
parser.add_argument('--gpu', type=int, default=0, help='gpu') #
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

# Pre-train
parser.add_argument('--kernel_size', type=int, default=100, help='kernel size for extracting trend') #
parser.add_argument('--seg_len', type=int, default=25, help='mask segment length') #
parser.add_argument('--p_tmask', type=float, default=0.2, help='masking ratio of trend component') #
parser.add_argument('--topk', type=int, default=3, help='number of masked series of seasonal component') #

# contrastive learning
parser.add_argument('--tau', default=0.1, type=float)
parser.add_argument('--alpha', default=0.5, type=float)

# reproduce
parser.add_argument('--seed', default=2024, type=int)

args = parser.parse_args()
args.uhi_data_dir = os.path.abspath(args.uhi_data_dir)
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
print('torch.cuda.device_count()', torch.cuda.device_count())
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
print('Args in experiment:')
print(args)

# set reasonable default freq for Exchange (monthly UHI)
if args.data == 'Exchange' and args.freq == 'h':
    args.freq = 'm'


random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)



Exp = Exp_STMTM

# helper to build the same setting string (with an '_iter' tag)
def build_setting(prefix=''):
    setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_df{}_nh{}_dh{}_el{}_fc{}_dp{}_hdp{}_ep{}_bs{}_lr{}_wd{}_ks{}_sl{}_pt{}_topk{}_tau{}_alp{}_iter'.format(
        args.task_name,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.d_ff,
        args.n_heads,
        args.d_hidden,
        args.e_layers,
        args.factor,
        args.dropout,
        args.head_dropout,
        args.train_epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.kernel_size,
        args.seg_len,
        args.p_tmask,
        args.topk,
        args.tau,
        args.alpha,
    )
    if args.output_tag:
        setting = '{}_{}'.format(setting, args.output_tag)
    setting = '{}_cp{}_cities{}'.format(setting, args.city_passes, len(train_paths))
    return setting

# City passes and model-internal epochs are intentionally separate. Mixing them
# would turn a 10-pass run into 10 x 10 internal epochs for every city.
NUM_PASSES = args.city_passes

# Load the explicit train/test city manifests used by the release pipeline.
def load_city_dataset_paths(train_csv_rel=None, test_csv_rel=None):
    train_csv_rel = train_csv_rel or args.train_city_metrics_path
    test_csv_rel = test_csv_rel or args.test_city_metrics_path
    cities_df = pd.read_csv(train_csv_rel)
    test_df = pd.read_csv(test_csv_rel)
    city_id_list = cities_df['id'].astype(int).tolist()
    test_city_id_list = test_df['id'].astype(int).tolist()
    if args.max_train_cities is not None:
        if args.max_train_cities < 1:
            raise ValueError('--max_train_cities must be positive')
        city_id_list = city_id_list[:args.max_train_cities]
    if args.max_test_cities is not None:
        if args.max_test_cities < 1:
            raise ValueError('--max_test_cities must be positive')
        test_city_id_list = test_city_id_list[:args.max_test_cities]
    train_paths = [os.path.join(args.uhi_data_dir, f"UHI_city_{city_id}.csv") for city_id in city_id_list]
    test_paths = [os.path.join(args.uhi_data_dir, f"UHI_city_{city_id}.csv") for city_id in test_city_id_list]
    print(f"Found {len(train_paths)} training cities from {train_csv_rel}.")
    print(f"Found {len(test_paths)} test cities from {test_csv_rel}.")
    missing = [path for path in train_paths + test_paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f'Missing UHI series: {missing[:3]}')
    return train_paths, test_paths

train_paths, test_paths = load_city_dataset_paths()

if args.task_name == 'pretrain':
    for ii in range(args.itr):
        setting = build_setting()
        if args.overwrite_outputs:
            for folder in [
                os.path.join(args.pretrain_checkpoints, args.data),
                os.path.join('./outputs/test_results', args.output_tag or args.data),
            ]:
                if os.path.exists(folder):
                    print(f'Removing existing output folder: {folder}')
                    shutil.rmtree(folder)
        # create one experiment instance (keep model instance across iterative city training)
        exp = Exp(args)
        print('>>>>>>> Created pretrain experiment instance : {} >>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

        synchronize_cuda()
        training_started = time.perf_counter()
        if args.global_training:
            exp.args.global_data_paths = train_paths
            exp.pretrain()
        else:
            # iterate epochs (cumulative training across cities)
            for city_pass in range(1, NUM_PASSES + 1):
                print(f'\n================ Pretrain city pass {city_pass}/{NUM_PASSES} ================')
                for idx, data_file in enumerate(train_paths, 1):
                    exp.args.data_path = data_file
                    print(f'[Pretrain pass {city_pass}] ({idx}/{len(train_paths)}) Training on: {data_file}')
                    exp.pretrain()
                    torch.cuda.empty_cache()
        synchronize_cuda()
        training_seconds = time.perf_counter() - training_started
        if args.efficiency_record_dir:
            write_measurement(
                os.path.join(args.efficiency_record_dir, 'training_stmtm_pretrain.json'),
                kind='training', method='ST-MTM', stage='pretrain',
                value=training_seconds, unit='seconds',
                note='self-supervised optimization and checkpoint writes; excludes data/model loading',
                metadata={
                    'epochs': args.train_epochs,
                    'training_cities': len(train_paths),
                    'batch_size': args.batch_size,
                },
            )

elif args.task_name == 'finetune':
    for ii in range(args.itr):
        setting = build_setting()
        if args.overwrite_outputs:
            for folder in [
                os.path.join(args.checkpoints, setting),
                os.path.join('./outputs/test_results', args.output_tag or args.data),
            ]:
                if os.path.exists(folder):
                    print(f'Removing existing output folder: {folder}')
                    shutil.rmtree(folder)
            for folder in [args.results_dir, args.sample_errors_dir]:
                if folder and os.path.exists(folder):
                    print(f'Removing existing output folder: {folder}')
                    shutil.rmtree(folder)
        ckpt_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        if args.load_checkpoints:
            if not os.path.isfile(args.load_checkpoints):
                raise FileNotFoundError(
                    f'Explicit pretrain/fine-tune checkpoint not found: {args.load_checkpoints}'
                )
            print(f'Loading explicitly requested checkpoint from: {args.load_checkpoints}')
        elif os.path.exists(ckpt_path):
            args.load_checkpoints = ckpt_path
            print(f'Loading finetune checkpoint from: {ckpt_path}')
        else:
            print(f'Finetune checkpoint not found at {ckpt_path}, testing with randomly initialized weights.')
        exp = Exp(args)
        print('>>>>>>> Created finetune experiment instance : {} >>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

        did_train = args.global_training or args.train_iterative
        if did_train:
            synchronize_cuda()
            training_started = time.perf_counter()
        if args.global_training:
            exp.args.global_data_paths = train_paths
            exp.args.data_path = train_paths[0]
            exp.train(setting)
        elif args.train_iterative:
            for city_pass in range(1, NUM_PASSES + 1):
                print(f'\n================ Finetune city pass {city_pass}/{NUM_PASSES} ================')
                for idx, data_file in enumerate(train_paths, 1):
                    exp.args.data_path = data_file
                    print(f'[Finetune pass {city_pass}] ({idx}/{len(train_paths)}) Training on: {data_file}')
                    exp.train(setting)
                    torch.cuda.empty_cache()
        else:
            print('train_iterative is disabled; testing will use the loaded/random initialized model.')
        if did_train:
            synchronize_cuda()
            training_seconds = time.perf_counter() - training_started
            if args.efficiency_record_dir:
                write_measurement(
                    os.path.join(args.efficiency_record_dir, 'training_stmtm_finetune.json'),
                    kind='training', method='ST-MTM', stage='finetune',
                    value=training_seconds, unit='seconds',
                    note='supervised optimization, in-loop validation, and checkpoint writes; excludes test inference',
                    metadata={
                        'epochs': args.train_epochs,
                        'training_cities': len(train_paths),
                        'batch_size': args.batch_size,
                    },
                )

        # after finetune across all cities, test on test city list if available
        if test_paths:
            aggregate_year_abs = defaultdict(float)
            aggregate_year_sq = defaultdict(float)
            aggregate_year_count = defaultdict(int)
            for idx, data_file in enumerate(test_paths, 1):
                exp.args.data_path = data_file
                print(f'[Finetune Test] ({idx}/{len(test_paths)}) Testing on: {data_file}')
                mae, mse, year_abs_sums, year_sq_sums, year_counts = exp.test(setting)
                for year, val in year_abs_sums.items():
                    aggregate_year_abs[year] += val
                for year, val in year_sq_sums.items():
                    aggregate_year_sq[year] += val
                for year, cnt in year_counts.items():
                    aggregate_year_count[year] += cnt
                torch.cuda.empty_cache()
            if aggregate_year_count:
                folder_path = args.results_dir or './outputs/test_results/{}/'.format(args.output_tag or args.data)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                yearly_file = os.path.join(folder_path, 'yearly_metrics_all_cities.csv')
                with open(yearly_file, 'w') as yf:
                    yf.write('year,mae,mse,rmse,num_samples\n')
                    expected_years = sorted(aggregate_year_count.keys()) or list(range(2003, 2021))
                    for year in expected_years:
                        if aggregate_year_count.get(year, 0) > 0:
                            mae_year = aggregate_year_abs[year] / aggregate_year_count[year]
                            mse_year = aggregate_year_sq[year] / aggregate_year_count[year]
                            yf.write(
                                f"{year},{mae_year:.6f},{mse_year:.6f},"
                                f"{np.sqrt(mse_year):.6f},{aggregate_year_count[year]}\n"
                            )
                        else:
                            yf.write(f"{year},nan,nan,nan,0\n")
                print(f'Saved aggregated yearly MAE/MSE to {yearly_file}')

                num_samples = sum(aggregate_year_count.values())
                start_year, start_month = divmod(args.evaluation_start, 100)
                end_year, end_month = divmod(args.evaluation_end, 100)
                expected_per_city = (
                    (end_year - start_year) * 12 + end_month - start_month + 1
                )
                if expected_per_city <= 0:
                    raise ValueError('evaluation_end must not precede evaluation_start')
                expected_samples = expected_per_city * len(test_paths)
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
                    'model': 'ST-MTM',
                    'mae': mae,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'num_cities': len(test_paths),
                    'num_samples': num_samples,
                    'evaluation_start': args.evaluation_start,
                    'evaluation_end': args.evaluation_end,
                }])
                summary_path = os.path.join(folder_path, 'summary.csv')
                summary.to_csv(summary_path, index=False)
                print(summary.to_string(index=False))
                print(f'Saved aggregate metrics to {summary_path}')

# %%
