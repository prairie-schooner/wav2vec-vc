import torch.nn as nn
import torch.nn.functional as F
import torch


class LayerWeightedSum(nn.Module):
    '''
        This code is cited from S3PRL project
        (https://github.com/s3prl/s3prl)
    '''
    def __init__(self, layer_norm=True):
        super().__init__()
        self.layer_norm = layer_norm

    def forward(self, x_list, layer_weight):
        n_layers = len(x_list)
        stacked_hs = torch.stack(x_list, dim=0)  # n_layers, B, T, C
        if self.layer_norm:
            stacked_hs = F.layer_norm(stacked_hs, (stacked_hs.shape[-1],))

        _, *origin_size = stacked_hs.size()
        stacked_hs = stacked_hs.view(n_layers, -1)  # n_layers, B * T * C
        weighted_hs = (layer_weight.unsqueeze(-1) * stacked_hs).sum(dim=0)
        weighted_hs = weighted_hs.view(*origin_size)
        return weighted_hs