#!/usr/bin/env python

# wujian@2019
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_complex.tensor import ComplexTensor
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
# import torchaudio

from espnet2.layers.stft import Stft 
from torch.utils.checkpoint import checkpoint
EPSILON = th.finfo(th.float32).eps

class Conv2dBlock(nn.Module):
    """
    2D convolutional blocks used in VoiceFilter
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(5, 5),
                 dilation=(1, 1)):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=1,
                              dilation=dilation,
                              padding=tuple(
                                  d * (k - 1) // 2
                                  for k, d in zip(kernel_size, dilation)))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        x: B x F x T
        """
        # import pdb;pdb.set_trace()
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

STFT = Stft()

class Frontend(nn.Module):
    """
    Reference from
        VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking
    """

    def __init__(self,
                 frame_len,
                 frame_hop,
                 speaker_num=1955,
                 round_pow_of_two=True,
                 embedding_dim=256,
                 log_mag=False,
                 mvn_mag=False,
                 lstm_dim=400,
                 transformer_dim=256,
                 linear_dim=600,
                 l2_norm=True,
                 bidirectional=False,
                 non_linear="relu"):
        super(Frontend, self).__init__()

        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "tanh": th.tanh
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError(
                "Unsupported non-linear function: {}".format(non_linear))
        N = 2**math.ceil(
            math.log2(frame_len)) if round_pow_of_two else frame_len
        num_bins = N // 2 + 1

        self.cnn_f = Conv2dBlock(1, 64, kernel_size=(7, 1))
        self.cnn_t = Conv2dBlock(64, 64, kernel_size=(1, 7))
        blocks = []
        for d in range(5):
            blocks.append(
                Conv2dBlock(64, 64, kernel_size=(5, 5), dilation=(1, 2**d)))
        self.cnn_tf = nn.Sequential(*blocks)
        self.proj = Conv2dBlock(64, 8, kernel_size=(1, 1))
        self.lstm = nn.LSTM(8 * num_bins,
                            lstm_dim,
                            batch_first=True,
                            bidirectional=bidirectional)
        # self.linear_transformer = nn.Linear(8 * num_bins, transformer_dim)
        # self.transformer = EncoderLayer(
        #         size=transformer_dim,
        #         self_attn=MultiHeadedAttention(4, transformer_dim, 0.0),
        #         feed_forward=PositionwiseFeedForward(256, 1024, 0.1),
        #         dropout_rate=0.1,
        #         normalize_before=True,
        #         concat_after=False
        #         )
        self.mask = nn.Sequential(
            nn.Linear(lstm_dim * 2 if bidirectional else lstm_dim, linear_dim),
            nn.ReLU(), nn.Linear(linear_dim, num_bins))
        # self.mask = nn.Sequential(
        #     nn.Linear(transformer_dim, linear_dim),
        #     nn.ReLU(), nn.Linear(linear_dim, num_bins))
        self.non_linear = supported_nonlinear[non_linear]
        self.embedding_dim = embedding_dim
        self.l2_norm = l2_norm
        self.log_mag = log_mag
        self.bn = nn.BatchNorm1d(num_bins) if mvn_mag else None

        # 类似FiLM的操作
        self.linear1_emb = nn.Linear(embedding_dim, 128)
        self.linear2_emb = nn.Linear(128, 8 * num_bins)
        
    def flatten_parameters(self):
        self.lstm.flatten_parameters()

    def check_args(self, x, e):
        if x.dim() != e.dim():
            raise RuntimeError(
                "{} got invalid input dim: x/e = {:d}/{:d}".format(
                    self.__name__, x.dim(), e.dim()))
        if e.size(-1) != self.embedding_dim:
            raise RuntimeError("input embedding dim do not match with "
                               "network's, {:d} vs {:d}".format(
                                   e.size(-1), self.embedding_dim))

    def forward(self, r, i, e, max_norm, return_mask=False):
        """
        x: B x T x F
        e: B * N x D
        max_norm: B x 1
        """
        if e.dim() == 1:
            e = th.unsqueeze(e, 0)
        if self.l2_norm:
            e = (e + EPSILON) / th.norm((e + EPSILON), 2, dim=1, keepdim=True)
        # B, T, F = x.shape
        # B x F x T
        # x = x.transpose(1, 2)
        # r = x.real
        # i = x.imag
        mag = (r**2 + i**2)**0.5
        ang = th.atan2(i, r)
        #import pdb;pdb.set_trace()

        # clip
        y = th.clamp(mag, min=EPSILON)
        # apply log
        if self.log_mag:
            y = th.log(y)
        # apply bn
        if self.bn:
            y = self.bn(y)

        B, _, T = mag.shape
        # B x 1 x F x T
        y = th.unsqueeze(y, 1)
        # B x D => B x D x T
        e = th.unsqueeze(e, 2).expand(-1, -1, T)
        # B x T x D
        e = th.transpose(e, 1, 2)

        # y = self.cnn_f(y)
        y = checkpoint(self.cnn_f, y)
        # m = checkpoint(self.forward_checkpoint, y, e, mag)
        # y = self.cnn_t(y)
        y = checkpoint(self.cnn_t, y)
        # y = self.cnn_tf(y)
        y = checkpoint(self.cnn_tf, y)
        # B x C x F x T
        y = self.proj(y)
        # B x CF x T
        y = y.view(B, -1, T)

        # B x T x CF
        e = self.linear1_emb(e)
        e_trans = self.linear2_emb(e)

        # B x T x CF
        y = th.transpose(y, 1, 2)
        f = y * e_trans

        f, _ = self.lstm(f)
        # f = self.linear_transformer(f)
        # f, _ = self.transformer(f, None)
        # B x T x F
        m = self.non_linear(self.mask(f))
        if return_mask:
            return m
        # B x F x T
        m = th.transpose(m, 1, 2)
        # B x T x F
        r = (m * mag * th.cos(ang)).transpose(1, 2)
        i = (m * mag * th.sin(ang)).transpose(1, 2)
        x = ComplexTensor(r, i)
        wav, _ = STFT.inverse(x)
        wav = wav / th.max(th.abs(wav), dim=1)[0].unsqueeze(1) * max_norm
        # torchaudio.backend.sox_io_backend.save("/home/work_nfs5_ssd/yhliang/workspace/espnet_meeting/Eval-far-R8003_M8001_MS801-N_SPK8001-0060645-0061659_wav.wav", wav.cpu(), 16000)
        # exit()
        # import pdb;pdb.set_trace()
        # wav = wav / th.max(th.abs(wav), dim=1)[0].unsqueeze(1) * max_norm
        # torchaudio.backend.sox_io_backend.save("/home/work_nfs5_ssd/yhliang/workspace/espnet_meeting/0.wav", wav.cpu(), 16000)
        # exit()

        return wav
    def forward_checkpoint(self, y, e, mag):
        B, _, T = mag.shape
        y = self.cnn_t(y)
        y = self.cnn_tf(y)
        # B x C x F x T
        y = self.proj(y)
        # B x CF x T
        y = y.view(B, -1, T)

        # B x T x CF
        e = self.linear1_emb(e)
        e_trans = self.linear2_emb(e)

        # B x T x CF
        y = th.transpose(y, 1, 2)
        f = y * e_trans

        f, _ = self.lstm(f)
        # f = self.linear_transformer(f)
        # f, _ = self.transformer(f, None)
        # B x T x F
        m = self.non_linear(self.mask(f))
        return m

def run():
    x = th.rand(1, 49600)
    e = th.rand(1, 256)

    nnet = VoiceFilter(frame_len=320,
                       frame_hop=160)
    print(nnet)
    s = nnet(x, e, return_mask=True)
    print(s.squeeze().shape)


if __name__ == "__main__":
    run()
