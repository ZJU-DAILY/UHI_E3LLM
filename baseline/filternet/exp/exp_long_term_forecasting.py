from data_provider.data_factory import data_provider, data_provider_many
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas
import re
from collections import defaultdict
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from efficiency import benchmark_model_call, write_measurement

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if getattr(self.args, 'global_data_paths', None) and flag in {'train', 'val'}:
            return data_provider_many(self.args, self.args.global_data_paths, flag)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                efficiency_dir = getattr(self.args, 'efficiency_record_dir', '')
                if efficiency_dir and not getattr(self, '_efficiency_recorded', False):
                    target_count = self.args.efficiency_batch_size
                    indices = torch.arange(target_count, device=batch_x.device) % batch_x.shape[0]
                    bench_x = batch_x.index_select(0, indices)
                    bench_x_mark = batch_x_mark.index_select(0, indices)
                    bench_y_mark = batch_y_mark.index_select(0, indices)
                    bench_dec = dec_inp.index_select(0, indices)

                    def benchmark_forward():
                        result = self.model(bench_x, bench_x_mark, bench_dec, bench_y_mark)
                        return result[0] if self.args.output_attention else result

                    latency = benchmark_model_call(
                        benchmark_forward,
                        sample_count=target_count,
                        warmup_calls=self.args.efficiency_warmup_calls,
                        measured_calls=self.args.efficiency_measured_calls,
                    )
                    write_measurement(
                        os.path.join(efficiency_dir, 'inference_filternet.json'),
                        kind='inference', method='FilterNet', stage='inference',
                        value=latency, unit='ms_per_sample',
                        note='model execution only; excludes loading, scaling, metrics, and file I/O',
                        metadata={
                            'batch_size': target_count,
                            'warmup_calls': self.args.efficiency_warmup_calls,
                            'measured_calls': self.args.efficiency_measured_calls,
                            'timing_protocol': 'independent_repetitions_mean',
                        },
                    )
                    self._efficiency_recorded = True
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
        print('Model {} : params: {:4f}M'.format(self.model._get_name(), para * 4 / 1024 / 1024))

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, load=0):
        test_data, test_loader = self._get_data(flag='test')
        if test and not load:
            print('loading model')
            checkpoint_path = (
                self.args.load_checkpoint_path
                if getattr(self.args, 'load_checkpoint_path', None)
                else os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            )
            print(f'loading checkpoint: {checkpoint_path}')
            self.model.load_state_dict(torch.load(checkpoint_path))

        preds = []
        trues = []
        year_abs_sums = defaultdict(float)
        year_sq_sums = defaultdict(float)
        year_counts = defaultdict(int)
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if getattr(self.args, 'evaluate_target_only', False):
                    outputs = outputs[:, :, -1:]
                    batch_y = batch_y[:, :, -1:]
                else:
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                years = batch_y_mark[:, -self.args.pred_len:, 0].detach().cpu().numpy()
                abs_err = np.abs(pred - true).mean(axis=-1)
                sq_err = ((pred - true) ** 2).mean(axis=-1)
                for b in range(years.shape[0]):
                    for t in range(years.shape[1]):
                        y = int(years[b, t])
                        if 2018 <= y <= 2020:
                            year_abs_sums[y] += abs_err[b, t]
                            year_sq_sums[y] += sq_err[b, t]
                            year_counts[y] += 1
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        dates = pandas.to_datetime(
            pandas.read_csv(
                os.path.join(test_data.root_path, test_data.data_path),
                usecols=['date'],
            )['date']
        )
        target_periods = []
        for sample_index in range(preds.shape[0]):
            for step in range(preds.shape[1]):
                target_date = dates.iloc[sample_index + self.args.seq_len + step]
                target_periods.extend(
                    [target_date.year * 100 + target_date.month] * preds.shape[2]
                )
        target_periods = np.asarray(target_periods)
        pred_flat = preds.reshape(-1)
        true_flat = trues.reshape(-1)
        evaluation_mask = (
            (target_periods >= self.args.evaluation_start)
            & (target_periods <= self.args.evaluation_end)
        )
        pred_eval = pred_flat[evaluation_mask]
        true_eval = true_flat[evaluation_mask]

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(pred_eval, true_eval)
        print('mse:{}, mae:{}'.format(mse, mae))

        sample_errors_dir = getattr(self.args, 'sample_errors_dir', None)
        if sample_errors_dir:
            os.makedirs(sample_errors_dir, exist_ok=True)
            match = re.search(r'(\d+)', os.path.basename(self.args.data_path))
            if not match:
                raise ValueError(
                    f'Cannot parse city ID from data path: {self.args.data_path}'
                )
            city_id = int(match.group(1))
            periods_eval = target_periods[evaluation_mask]
            sample_df = pandas.DataFrame(
                {
                    'city_id': city_id,
                    'sample_index': np.arange(len(pred_eval)),
                    'target_year_month': periods_eval,
                    'true': true_eval,
                    'pred': pred_eval,
                    'error': pred_eval - true_eval,
                    'abs_error': np.abs(pred_eval - true_eval),
                    'sq_error': (pred_eval - true_eval) ** 2,
                }
            )
            sample_df.to_csv(
                os.path.join(sample_errors_dir, f'city_{city_id}.csv'),
                index=False,
            )
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        metrics_dict = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe
        }
        metrics_df = pandas.DataFrame([metrics_dict])
        metrics_csv_path = os.path.join(folder_path, 'metrics_15epochs.csv')
        if os.path.exists(metrics_csv_path):
            metrics_df.to_csv(metrics_csv_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(metrics_csv_path, mode='w', header=True, index=False)

        city_name = os.path.basename(self.args.data_path)
        per_year_mae = {year: year_abs_sums[year] / year_counts[year] for year in year_counts}
        per_year_mse = {year: year_sq_sums[year] / year_counts[year] for year in year_counts}
        expected_years = sorted(year_counts.keys()) or list(range(2003, 2021))

        yearly_csv = os.path.join(folder_path, 'yearly_metrics.csv')
        need_header = not os.path.exists(yearly_csv)
        with open(yearly_csv, 'a') as yf:
            if need_header:
                yf.write('city,year,mae,mse\n')
            for year in expected_years:
                mae_val = per_year_mae.get(year, float('nan'))
                mse_val = per_year_mse.get(year, float('nan'))
                yf.write(f"{city_name},{year},{mae_val:.6f},{mse_val:.6f}\n")

        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)

        return year_abs_sums, year_sq_sums, year_counts

        return
