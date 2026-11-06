import torch
import numpy as np
import torch.nn.functional as F


def costume_collate(data, max_len=None, mask_compensation=False):

    batch_size = len(data)
    features, masks = zip(*data)

    lengths = [X.shape[0] for X in features]
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])
    target_masks = torch.zeros_like(
        X, dtype=torch.bool)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]

    targets = X.clone()
    X = X * target_masks
    if mask_compensation:
        X = compensate_masking(X, target_masks)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)
    return X, targets, target_masks, padding_masks


def compensate_masking(X, mask):

    num_active = torch.sum(mask, dim=-1).unsqueeze(-1)
    num_active = torch.max(num_active, torch.ones(num_active.shape, dtype=torch.int16))
    return X.shape[-1] * X / num_active


def padding_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':
        if mode == 'separate':
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)
        else:
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask


def geom_noise_mask_single(L, lm, masking_ratio):
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    p = [p_m, p_u]

    state = int(np.random.rand() > masking_ratio)
    for i in range(L):
        keep_mask[i] = state
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask
