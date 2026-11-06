import torch as t
import torch
import torch.nn as nn
import numpy as np

def divide_no_nan(a, b):
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, n_pos, tau):
        super(ContrastiveLoss, self).__init__()
        self.bs = batch_size
        self.n_pos = n_pos
        self.tau = tau

    def forward(self, anc, pos_list):
        batch_size = anc.size(0)
        anc = torch.reshape(anc, (batch_size, -1, anc.size(-1))).transpose(0, 1)
        pos = torch.stack(pos_list, dim=1)
        pos = torch.reshape(pos, (batch_size, -1, self.n_pos, pos.size(-1))).transpose(0, 1)
        z = torch.cat([anc.unsqueeze(2), pos], dim=2)
        z = z.reshape(z.size(0), -1, z.size(-1))

        n_sample = z.size(1)
        sim = torch.matmul(z, z.transpose(-2, -1)) / self.tau
        logits = torch.triu(sim, diagonal=1)[:, :, 1:] + torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits = - torch.log_softmax(logits, dim=-1)

        i = torch.arange(0, n_sample, step=self.n_pos+1)
        j = torch.arange(n_sample).unfold(0, size=self.n_pos, step=self.n_pos+1)
        logits = logits[:, i.unsqueeze(1), j]

        loss = logits.mean(dim=-1)
        return loss.mean()


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target), t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class AutomaticWeightedLoss(nn.Module):

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
