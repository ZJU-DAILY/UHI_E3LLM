from data_provider.data_factory import data_provider, data_provider_many
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, transfer_weights, show_series, show_matrix
from utils.metrics import metric
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
import re
from collections import OrderedDict, defaultdict
try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    from torch.utils.tensorboard import SummaryWriter
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from efficiency import benchmark_model_call, synchronize_cuda, write_measurement

warnings.filterwarnings('ignore')


class Exp_STMTM(Exp_Basic):
    def __init__(self, args):
        super(Exp_STMTM, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")
        self.training_seconds = 0.0

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        print(model)

        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))
            print(self.device)
            print('torch.cuda.device_count()', torch.cuda.device_count())
            model = transfer_weights(self.args.load_checkpoints, model, device=self.device)

        if getattr(self.args, 'freeze_backbone', False):
            for name, parameter in model.named_parameters():
                parameter.requires_grad = name.startswith('pred_head.')
            trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            print(f'Frozen backbone; trainable parameters: {trainable}')

        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model

    def _get_data(self, flag):
        if getattr(self.args, 'global_data_paths', None) and flag in {'train', 'val'}:
            return data_provider_many(self.args, self.args.global_data_paths, flag)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def pretrain(self):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        self.train_show = next(iter(train_loader))
        self.valid_show = next(iter(vali_loader))

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,
                                                                     T_max=self.args.train_epochs)

        min_vali_loss = None
        synchronize_cuda()
        training_started = time.perf_counter()
        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            train_loss, train_mse_loss, train_cont_loss = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler)
            vali_loss, vali_mse_loss, vali_cont_loss = self.valid_one_epoch(vali_loader)


            end_time = time.time()
            print(
                "Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f} ({4:.3f}, {5:.3f}) Val Loss: {6:.4f} ({7:.3f}, {8:.3f})"
                .format(epoch, model_scheduler.get_lr()[0], end_time - start_time,
                        train_loss, train_mse_loss, train_cont_loss,
                        vali_loss, vali_mse_loss, vali_cont_loss))

            loss_scalar_dict = {
                'train_loss': train_loss,
                'train_mse_loss' : train_mse_loss,
                'train_cont_loss' : train_cont_loss,
                'vali_loss': vali_loss,
                'vali_mse_loss': vali_mse_loss,
                'vali_cont_loss': vali_cont_loss
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, vali_loss, epoch))

                min_vali_loss = vali_loss
                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'embedding' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best.pth"))

            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'embedding' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

        synchronize_cuda()
        self.training_seconds += time.perf_counter() - training_started

    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):

        train_loss = []
        train_cl_loss = []
        train_mse_loss = []

        self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()

            batch_x = batch_x.float().to(self.device)
            loss, mse_loss, cont_loss = self.model(batch_x, batch_x_mark)

            loss.backward()
            model_optim.step()

            train_loss.append(loss.item())
            train_mse_loss.append(mse_loss.item())
            train_cl_loss.append(cont_loss.item())

        model_scheduler.step()

        train_loss = np.average(train_loss)
        train_mse_loss = np.average(train_mse_loss)
        train_cl_loss = np.average(train_cl_loss)

        return train_loss, train_mse_loss, train_cl_loss

    def valid_one_epoch(self, vali_loader):
        valid_loss = []
        valid_cl_loss = []
        valid_mse_loss = []

        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            loss, mse_loss, cont_loss = self.model(batch_x, batch_x_mark)

            valid_loss.append(loss.item())
            valid_mse_loss.append(mse_loss.item())
            valid_cl_loss.append(cont_loss.item())

        valid_loss = np.average(valid_loss)
        valid_mse_loss = np.average(valid_mse_loss)
        valid_cl_loss = np.average(valid_cl_loss)

        self.model.train()
        return valid_loss, valid_mse_loss, valid_cl_loss

    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        synchronize_cuda()
        training_started = time.perf_counter()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            start_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                loss.backward()
                model_optim.step()

                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)

            end_time = time.time()
            print(
            "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f}".format(
                epoch + 1, train_steps, end_time - start_time, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        synchronize_cuda()
        self.training_seconds += time.perf_counter() - training_started
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        self.lr = model_optim.param_groups[0]['lr']

        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark)

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

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        year_abs_sums = defaultdict(float)
        year_sq_sums = defaultdict(float)
        year_counts = defaultdict(int)
        folder_path = './outputs/test_results/{}/'.format(getattr(self.args, 'output_tag', '') or self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                efficiency_dir = getattr(self.args, 'efficiency_record_dir', '')
                if efficiency_dir and not getattr(self, '_efficiency_recorded', False):
                    target_count = self.args.efficiency_batch_size
                    indices = torch.arange(target_count, device=batch_x.device) % batch_x.shape[0]
                    bench_x = batch_x.index_select(0, indices)
                    bench_x_mark = batch_x_mark.index_select(0, indices)
                    latency = benchmark_model_call(
                        lambda: self.model(bench_x, bench_x_mark),
                        sample_count=target_count,
                        warmup_calls=self.args.efficiency_warmup_calls,
                        measured_calls=self.args.efficiency_measured_calls,
                    )
                    write_measurement(
                        os.path.join(efficiency_dir, 'inference_stmtm.json'),
                        kind='inference', method='ST-MTM', stage='inference',
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
                outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                if test_data.scale:
                    pred_shape = pred.shape
                    true_shape = true.shape
                    pred = test_data.inverse_transform(
                        pred.reshape(-1, pred_shape[-1])
                    ).reshape(pred_shape)
                    true = test_data.inverse_transform(
                        true.reshape(-1, true_shape[-1])
                    ).reshape(true_shape)
                if getattr(self.args, 'target_only_metrics', False):
                    pred = pred[..., -1:]
                    true = true[..., -1:]

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        dates = pd.to_datetime(
            pd.read_csv(
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
        errors = (pred_eval - true_eval).reshape(-1, preds.shape[-1])
        years = target_periods[evaluation_mask].reshape(-1, preds.shape[-1])[:, 0] // 100
        absolute_errors = np.abs(errors).mean(axis=-1)
        squared_errors = (errors ** 2).mean(axis=-1)
        for year in np.unique(years):
            selected = years == year
            year_abs_sums[int(year)] = float(absolute_errors[selected].sum())
            year_sq_sums[int(year)] = float(squared_errors[selected].sum())
            year_counts[int(year)] = int(selected.sum())

        mae, mse, rmse, mape, mspe = metric(pred_eval, true_eval)
        print('{0}->{1}, mse:{2:.3f}, mae:{3:.3f}'.format(self.args.seq_len, self.args.pred_len, mse, mae))

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
            sample_df = pd.DataFrame(
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
        f = open(folder_path + "score_raw.txt", 'a')
        if self.args.pred_len==96:
            f.write(setting+'\n')
        f.write('{0}->{1}, {2:.3f}, {3:.3f} \n'.format(self.args.seq_len, self.args.pred_len, mse, mae))
        f.close()
        per_year_mae = {year: year_abs_sums[year] / year_counts[year] for year in year_counts}
        per_year_mse = {year: year_sq_sums[year] / year_counts[year] for year in year_counts}
        expected_years = sorted(year_counts.keys()) or list(range(2003, 2021))
        yearly_path = os.path.join(folder_path, "yearly_mae.txt")
        with open(yearly_path, 'a') as yf:
            city_name = os.path.basename(self.args.data_path)
            yf.write(setting + '\n')
            for year in expected_years:
                mae_val = per_year_mae.get(year, float('nan'))
                yf.write(f"{city_name},{year},{mae_val:.6f}\n")

        csv_path = os.path.join(folder_path, "yearly_metrics.csv")
        need_header = not os.path.exists(csv_path)
        with open(csv_path, 'a') as cf:
            if need_header:
                cf.write("city,year,mae,mse\n")
            city_name = os.path.basename(self.args.data_path)
            for year in expected_years:
                mae_val = per_year_mae.get(year, float('nan'))
                mse_val = per_year_mse.get(year, float('nan'))
                cf.write(f"{city_name},{year},{mae_val:.6f},{mse_val:.6f}\n")

        return mae, mse, year_abs_sums, year_sq_sums, year_counts
