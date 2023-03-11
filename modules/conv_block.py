import torch.nn as nn
import torch.nn.functional as F
from modules.conv_norm import ConvNorm


class Conv1DBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, bias=True, drop_rate=0.0):
        super().__init__()

        self.subsample = int(c_in // c_out)
        self.seq = nn.Sequential(
            ConvNorm(c_in, c_out, kernel_size=kernel_size, bias=bias, stride=1),
            nn.BatchNorm1d(c_out),
            nn.ELU(),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        y = self.seq(x)
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x, kernel_size=self.subsample)
        x = x.transpose(1, 2)
        return x + y