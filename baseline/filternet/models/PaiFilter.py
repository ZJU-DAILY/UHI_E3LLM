import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02

        self.embed_size = self.seq_len
        self.hidden_size = configs.hidden_size

        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )


    def circular_convolution(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = x.permute(0, 2, 1)

        x = self.circular_convolution(x, self.w.to(x.device))  # B, N, D

        x = self.fc(x)
        x = x.permute(0, 2, 1)

        return x
