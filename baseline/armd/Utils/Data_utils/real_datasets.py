import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from Models.autoregressive_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self,
        name,
        data_root,
        window=64,
        proportion=0.8,
        save2npy=True,
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate',
        distribution='geometric',
        mean_mask_length=3
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.rawdata, self.scaler, self.raw_timestamps = self.read_data(data_root, self.name)


        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = False

        self.data = self.__normalize(self.rawdata)
        train, inference, train_ts, test_ts = self.__getsamples(self.data, proportion, seed, self.raw_timestamps, self.period)

        self.samples = train if period == 'train' else inference
        self.timestamps = train_ts if period == 'train' else test_ts
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed, timestamps, period):
        if period == 'test' and self.pred_len is not None:
            # ARMD consumes a 12-month context and emits a native 12-month
            # forecast. Build one window for every available next-month target;
            # the unavailable tail is padding and is never used as model input.
            context_length = self.window - int(self.pred_len)
            if context_length <= 0:
                raise ValueError('predict_length must be smaller than window')
            sample_num = max(self.len - context_length, 0)
        else:
            sample_num = self.sample_num_total

        x = np.zeros((sample_num, self.window, self.var_num))
        ts_win = np.zeros((sample_num, self.window))
        for i in range(sample_num):
            observed = data[i:min(i + self.window, self.len), :]
            observed_ts = timestamps[i:min(i + self.window, self.len)]
            x[i, :len(observed), :] = observed
            ts_win[i, :len(observed_ts)] = observed_ts
            if len(observed) < self.window:
                x[i, len(observed):, :] = observed[-1]
                ts_win[i, len(observed_ts):] = observed_ts[-1]

        if period == 'test':
            train_data, test_data = x[:0], x
            train_ts, test_ts = ts_win[:0], ts_win
        else:
            train_data, test_data = self.divide(x, proportion, seed)
            train_ts, test_ts = self.divide(ts_win, proportion, seed)

        if self.save2npy:
            if train_data.size > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
                if self.auto_norm:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
                else:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)
            if test_data.size > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
                if self.auto_norm:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                else:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)

        return train_data, test_data, train_ts, test_ts

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)

    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)

        timestamps = None
        if 'date' in df.columns[0].lower():
            dates = pd.to_datetime(df.iloc[:, 0], errors='raise')
            order = np.argsort(dates.to_numpy())
            df = df.iloc[order].reset_index(drop=True)
            dates = dates.iloc[order].reset_index(drop=True)
            if dates.duplicated().any():
                raise ValueError(f'Duplicate monthly timestamps in {filepath}')
            expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq='MS')
            if len(expected) != len(dates) or not np.array_equal(
                expected.values, dates.values
            ):
                raise ValueError(f'Non-contiguous monthly series in {filepath}')
            timestamps = (dates.dt.year * 100 + dates.dt.month).values
            df.drop(df.columns[0], axis=1, inplace=True)
            fit_mask = dates <= pd.Timestamp('2017-12-01')
            if int(fit_mask.sum()) != 180:
                raise ValueError(
                    f'Expected 180 historical months (2003-01 to 2017-12) in '
                    f'{filepath}; found {int(fit_mask.sum())}'
                )
        else:
            # fallback: create dummy consecutive years to avoid breaking downstream aggregation
            timestamps = np.arange(len(df))
            df.drop(df.columns[0], axis=1, inplace=True)
            fit_mask = np.ones(len(df), dtype=bool)

        data = df.values
        scaler = StandardScaler()
        scaler = scaler.fit(data[np.asarray(fit_mask)])
        return data, scaler, timestamps

    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            ts = self.timestamps[ind, :] if self.timestamps is not None else np.zeros(self.window)
            return torch.from_numpy(x).float(), torch.from_numpy(m), torch.from_numpy(ts).float()
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num


class fMRIDataset(CustomDataset):
    def __init__(
        self,
        proportion=1.,
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
