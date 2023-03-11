import torch
import torch.nn as nn


class InstanceNorm1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-5

    def forward(self, x, mask=None):
        if mask is None:
            device = 'cpu' if x.get_device() < 0 else f'cuda:{x.get_device()}'
            mask = torch.zeros(x.size(0), x.size(1)).bool().to(device)

        mask = ~mask
        mask = mask.float().unsqueeze(-1)  # (N,L,1)

        mean = (torch.sum(x * mask, 1) / torch.sum(mask, 1))  # (N,C)
        mean = mean.detach()

        var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask) ** 2  # (N,L,C)
        var = (torch.sum(var_term, 1) / torch.sum(mask, 1))  # (N,C)
        var = var.detach()

        mean_reshaped = mean.unsqueeze(1).expand_as(x)  # (N, L, C)
        var_reshaped = var.unsqueeze(1).expand_as(x)  # (N, L, C)

        ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + self.eps)  # (N, L, C)
        return ins_norm