from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_PEMS
from torch.utils.data import ConcatDataset, DataLoader

import torch
import numpy as np

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Weather': Dataset_Custom,
    'Electricity': Dataset_Custom,
    'Exchange' : Dataset_Custom,
    'PEMS08' : Dataset_PEMS,
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True if flag=='train' else False
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq


    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=args.task_name == 'pretrain',
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )


    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    print(flag, len(data_set), len(data_loader))
    return data_set, data_loader


def data_provider_many(args, data_paths, flag):
    """Build one loader without allowing windows to cross city boundaries."""
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    datasets = [
        Data(
            root_path=args.root_path,
            data_path=path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=args.task_name == 'pretrain',
            timeenc=timeenc,
            freq=args.freq,
            seasonal_patterns=args.seasonal_patterns,
        )
        for path in data_paths
    ]
    combined = ConcatDataset(datasets)
    loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=flag == 'train',
        num_workers=args.num_workers,
        drop_last=False,
    )
    print(flag, len(combined), len(loader), f'across {len(datasets)} cities')
    return combined, loader
