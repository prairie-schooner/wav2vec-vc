import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.transpose import Transpose
from modules.conv_block import Conv1DBlock
from modules.instance_norm import InstanceNorm1D
from modules.layer_weighted_sum import LayerWeightedSum


class SpeakerEncoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, layer_weight_path=None):
        super().__init__()

        self.layer_weight = nn.Parameter(torch.randn(13), requires_grad=False)
        if layer_weight_path is not None:
            speaker_layer_weight = F.softmax(torch.load(layer_weight_path['speaker']), dim=0)
            speaker_layer_weight = speaker_layer_weight.clone().detach().float()
            self.layer_weight = nn.Parameter(speaker_layer_weight, requires_grad=False)

        self.layer_weighted_sum = LayerWeightedSum()
        self.linear = nn.Linear(c_in, c_h, bias=False)
        self.conv1d_block = nn.Sequential(
            Transpose((1, 2)),
            Conv1DBlock(c_h, c_out, kernel_size=9, bias=False),
            Transpose((1, 2))
        )

    def statistics_pooling(self, x, mask):
        eps = 1e-5

        mask = ~mask

        mask = mask.float().unsqueeze(-1)  # (N,L,1)
        mean = (torch.sum(x * mask, 1) / torch.sum(mask, 1))  # (N,C)

        var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask) ** 2  # (N,L,C)
        var = (torch.sum(var_term, 1) / torch.sum(mask, 1))  # (N,C)
        std = torch.sqrt(var + eps)
        return mean, std

    def forward(self, w2v, w2v_mask=None):
        y = self.layer_weighted_sum(w2v, self.layer_weight)
        y = self.linear(y)
        y = self.conv1d_block(y)

        mn, sd = self.statistics_pooling(y, w2v_mask)

        return mn, sd


class ContentEncoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, layer_weight_path=None):
        super().__init__()

        self.layer_weight = nn.Parameter(torch.randn(13), requires_grad=False)
        if layer_weight_path is not None:
            content_layer_weight = F.softmax(torch.load(layer_weight_path['content']), dim=0)
            content_layer_weight = content_layer_weight.clone().detach().float()
            self.layer_weight = nn.Parameter(content_layer_weight, requires_grad=False)

        self.layer_weighted_sum = LayerWeightedSum()
        self.instance_norm_1d = InstanceNorm1D()

        self.linear = nn.Linear(c_in, c_h, bias=False)

        self.conv1d_block = nn.Sequential(
            Transpose((1, 2)),
            Conv1DBlock(c_h, c_h // 2, kernel_size=3, bias=False),
            Transpose((1, 2))
        )
        self.bottleneck = nn.Linear(c_h // 2, c_out)

    def forward(self, w2v, w2v_mask=None):
        y = self.layer_weighted_sum(w2v, self.layer_weight)
        y = self.linear(y)
        y = self.conv1d_block(y)

        y = self.instance_norm_1d(y, w2v_mask)
        enc = self.bottleneck(y)

        return enc


class Decoder(nn.Module):
    def __init__(
            self, c_content, c_speaker, c_h, c_out
    ):
        super().__init__()

        self.linear = nn.Linear(c_content, c_speaker)
        self.instance_norm_1d = InstanceNorm1D()

        self.linear_block = nn.Sequential(
            nn.Linear(c_speaker, c_h),
            Transpose((1, 2)),
            nn.BatchNorm1d(c_h),
            Transpose((1, 2)),
            nn.ReLU(),
            nn.Linear(c_h, c_h)
        )

        self.conv1d_block = nn.Sequential(
            Transpose((1, 2)),
            Conv1DBlock(c_in=c_speaker, c_out=c_h, kernel_size=5),
            Transpose((1, 2))
        )

        self.gru = nn.GRU(input_size=c_h, hidden_size=c_h, num_layers=1, batch_first=True)
        self.out_layer = nn.Linear(c_h, c_out)

    def forward(self, enc_content, enc_speaker, mask):

        enc_content = self.linear(enc_content)
        enc_content = self.instance_norm_1d(enc_content, mask)

        mn, sd = enc_speaker
        y = enc_content * sd.unsqueeze(1) + mn.unsqueeze(1)

        y = self.linear_block(y)
        y = self.conv1d_block(y)

        y, _ = self.gru(y)
        y = self.out_layer(y)

        return y


class Wav2vecVC(nn.Module):
    def __init__(self, layer_weight_path=None):
        super().__init__()

        c_w2v = 768

        c_h_content = 512
        c_h_speaker = 512
        c_h_decoder = 512

        c_content = 4
        c_speaker = 512
        c_mel = 80

        self.speaker_encoder = SpeakerEncoder(c_in=c_w2v, c_h=c_h_speaker, c_out=c_speaker,
                                              layer_weight_path=layer_weight_path)
        self.content_encoder = ContentEncoder(c_in=c_w2v, c_h=c_h_content, c_out=c_content,
                                              layer_weight_path=layer_weight_path)
        self.decoder = Decoder(c_speaker=c_speaker, c_content=c_content, c_h=c_h_decoder, c_out=c_mel)

    def forward(self, w2vs, w2v_mask):
        enc_s = self.speaker_encoder(w2vs, w2v_mask)
        enc_c = self.content_encoder(w2vs, w2v_mask)

        dec = self.decoder(enc_c, enc_s, w2v_mask)

        return dec

    def inference(self, source_w2vs, target_w2vs, source_w2v_mask, target_w2v_mask):
        enc_s = self.speaker_encoder(target_w2vs, w2v_mask=target_w2v_mask)
        enc_c = self.content_encoder(source_w2vs, w2v_mask=source_w2v_mask)

        dec = self.decoder(enc_c, enc_s, source_w2v_mask)

        return dec
