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

class ConformerTargetSeparatorLinear(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 1,
        spk_embed_dim: int = 512,
        adim: int = 384,
        aheads: int = 4,
        layers_total: int = 6,
        layers_tgt: int = 3,
        linear_units: int = 1536,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        normalize_before: bool = False,
        concat_after: bool = False,
        dropout_rate: float = 0.1,
        input_layer: str = "linear",
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        nonlinear: str = "relu",
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        conformer_activation_type: str = "swish",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        conformer_enc_kernel_size: int = 7,
        padding_idx: int = -1,
    ):
        """Conformer separator.

        Args:
            input_dim: input feature dimension
            num_spk: number of speakers
            adim (int): Dimension of attention.
            spk_embed_dim (int): Dimension of speaker embedding.
            aheads (int): The number of heads of multi head attention.
            linear_units (int): The number of units of position-wise feed forward.
            layers_total (int): The total number of transformer blocks.
            layers_tgt (int): The number of transformer blocks after concate speaker embeddings.
            dropout_rate (float): Dropout rate.
            input_layer (Union[str, torch.nn.Module]): Input layer type.
            attention_dropout_rate (float): Dropout rate in attention.
            positional_dropout_rate (float): Dropout rate after adding
                                             positional encoding.
            normalize_before (bool): Whether to use layer_norm before the first block.
            concat_after (bool): Whether to concat attention layer's input and output.
                if True, additional linear will be applied.
                i.e. x -> x + linear(concat(x, att(x)))
                if False, no additional linear will be applied. i.e. x -> x + att(x)
            conformer_pos_enc_layer_type(str): Encoder positional encoding layer type.
            conformer_self_attn_layer_type (str): Encoder attention layer type.
            conformer_activation_type(str): Encoder activation function type.
            positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
            positionwise_conv_kernel_size (int): Kernel size of
                                                 positionwise conv1d layer.
            use_macaron_style_in_conformer (bool): Whether to use macaron style for
                                                   positionwise layer.
            use_cnn_in_conformer (bool): Whether to use convolution module.
            conformer_enc_kernel_size(int): Kernerl size of convolution module.
            padding_idx (int): Padding idx for input_layer=embed.
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
        """
        super().__init__()

        self._num_spk = num_spk
        self.conformer = ConformerEncoder(
            idim=input_dim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=linear_units,
            num_blocks=layers_total-layers_tgt,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            input_layer=input_layer,
            normalize_before=normalize_before,
            concat_after=concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            macaron_style=use_macaron_style_in_conformer,
            pos_enc_layer_type=conformer_pos_enc_layer_type,
            selfattention_layer_type=conformer_self_attn_layer_type,
            activation_type=conformer_activation_type,
            use_cnn_module=use_cnn_in_conformer,
            cnn_module_kernel=conformer_enc_kernel_size,
            padding_idx=padding_idx,
        )
        self.conformer_tgt = ConformerEncoder(
            idim=adim + spk_embed_dim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=linear_units,
            num_blocks=layers_tgt,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            input_layer='linear',
            normalize_before=normalize_before,
            concat_after=concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            macaron_style=use_macaron_style_in_conformer,
            pos_enc_layer_type=conformer_pos_enc_layer_type,
            selfattention_layer_type=conformer_self_attn_layer_type,
            activation_type=conformer_activation_type,
            use_cnn_module=use_cnn_in_conformer,
            cnn_module_kernel=conformer_enc_kernel_size,
            padding_idx=padding_idx,
        )
        self._input_layer = input_layer
        if input_layer == 'linear':
            self.linear = torch.nn.Linear(adim, input_dim)
        elif input_layer == 'conv2d':
            self.transpose_1 = torch.nn.ConvTranspose1d(adim, adim, 3, 2)
            self.transpose_2 = torch.nn.ConvTranspose1d(adim, input_dim, 3, 2)
        else:
            raise ValueError("Not supporting input_layer={}".format(input_layer))

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))
        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]
    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        xvectors: torch.Tensor,
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
        if isinstance(input, ComplexTensor):
            feature = abs(input)
        else:
            feature = input

        input_length = input.shape[1]
        if xvectors.dim() == 2:
            xvectors = xvectors.unsqueeze(1)

        # prepare pad_mask for transformer
        pad_mask = make_non_pad_mask(ilens).unsqueeze(1).to(feature.device)
        # B*T*N -> B*T/4*N
        x, f_mask = self.conformer(feature, pad_mask)
        masks = []
        for i in range(self._num_spk):
            xvec = xvectors[:, i]
            xvec = xvec.unsqueeze(1).repeat(1,input_length,1)
            y, _ = self.conformer_tgt(torch.cat([x,xvec],axis=-1), f_mask)
            y = self.linear(y)
            mask = self.nonlinear(y)
            masks.append(mask)

        masked = [input * mask for mask in masks]

        return masked, ilens

    @property
    def num_spk(self):
        return self._num_spk
