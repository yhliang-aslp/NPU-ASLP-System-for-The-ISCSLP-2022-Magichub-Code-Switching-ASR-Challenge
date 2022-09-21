# Conformer Target Separator
from collections import OrderedDict
from typing import List
from typing import Tuple
from typing import Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.conformer.encoder import (
    Encoder as ConformerEncoder,  # noqa: H301
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet2.enh.separator.abs_separator import AbsSeparator
import pdb
EPSILON = torch.finfo(torch.float32).eps


class Conv2dBlock(torch.nn.Module):
    """
    2D convolutional blocks used in VoiceFilter
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(5, 5),
                 dilation=(1, 1)):
        super(Conv2dBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=1,
                              dilation=dilation,
                              padding=tuple(
                                  d * (k - 1) // 2
                                  for k, d in zip(kernel_size, dilation)))
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        x: N x F x T
        """
        x = self.bn(self.conv(x))
        return torch.nn.functional.relu(x)



class VoiceFilter(AbsSeparator):
    def __init__(
        self,
        input_dim: int= 257,
        embedding_dim: int = 512,
        log_mag: bool = False,
        mvn_mag: bool = False,
        lstm_dim: int = 400,
        linear_dim: int = 600,
        l2_norm: bool = True,
        bidirectional: bool = False,
        non_linear: str = "relu",
        num_spk: int = 1,
    ):
        super().__init__()

        self._num_spk = num_spk
        self.speaker_down = torch.nn.Linear(512,256)
        embedding_dim = 256
        out_channel= 32
        self.cnn_f = Conv2dBlock(1, out_channel, kernel_size=(7, 1))
        self.cnn_t = Conv2dBlock(out_channel, out_channel, kernel_size=(1, 7))
        blocks = []
        for d in range(5):
            blocks.append(
                Conv2dBlock(out_channel, out_channel, kernel_size=(5, 5), dilation=(1, 2**d)))
        self.cnn_tf = torch.nn.Sequential(*blocks)
        self.proj = Conv2dBlock(out_channel, 4, kernel_size=(1, 1))
        self.lstm = torch.nn.LSTM(4*input_dim + embedding_dim,
                            lstm_dim,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.mask = torch.nn.Sequential(
            torch.nn.Linear(lstm_dim * 2 if bidirectional else lstm_dim, linear_dim),
            torch.nn.ReLU(), torch.nn.Linear(linear_dim, input_dim))
        self.embedding_dim = embedding_dim
        self.l2_norm = l2_norm
        self.log_mag = log_mag
        self.bn = torch.nn.BatchNorm1d(input_dim) if mvn_mag else None
        if non_linear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting non_linear={}".format(non_linear))
        self.non_linear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[non_linear]
    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        xvectors: torch.Tensor,
        rt_mask: bool = False,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
            xvectors (torch.Tensor): target speaker embedding [B, nb_spk, spk_embed_dim]

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, nb_spk, T, N), ...]
            ilens (torch.Tensor): (B,)
        """
        # if complex spectrum,
        assert isinstance(input, ComplexTensor)
        #N x F x T
        mag = torch.transpose(input.real, 1, 2)
        ang = input.imag
        e = self.speaker_down(xvectors)
        #e = xvectors
        # clip
        y = torch.clamp(mag, min=EPSILON)

        # apply log
        if self.log_mag:
            y = torch.log(y)
        # apply bn
        if self.bn:
            y = self.bn(y)

        N, _, T = mag.shape
        # N x 1 x F x T
        y = torch.unsqueeze(y, 1)
        # N x D => N x D x T
        if len(xvectors.shape) == 2:
            e = torch.unsqueeze(e, 2).repeat(1, 1, T)
        else:
            e = torch.transpose(e, 1, 2).repeat(1, 1, T)
        y = self.cnn_f(y)
        y = self.cnn_t(y)
        y = self.cnn_tf(y)
        # N x C x F x T
        y = self.proj(y)
        # N x CF x T
        y = y.view(N, -1, T)
        # N x (CF+D) x T
        f = torch.cat([y, e], 1)
        # N x T x (CF+D)
        f = torch.transpose(f, 1, 2)
        f, _ = self.lstm(f)
        # N x T x F
        m = self.non_linear(self.mask(f))
        masks = []
        masks.append(m)
        # N x F x T
        m = torch.transpose(m, 1, 2)
        mag_mask = mag * m

        masked = [ComplexTensor(torch.transpose(mag_mask, 1, 2),ang)]
        if rt_mask:
            return masked, ilens, masks
        else:
            return masked, ilens
        
    @property
    def num_spk(self):
        return self._num_spk
