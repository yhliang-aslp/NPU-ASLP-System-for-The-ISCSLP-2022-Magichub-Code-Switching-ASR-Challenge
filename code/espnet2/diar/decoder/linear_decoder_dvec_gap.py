import torch

from espnet2.diar.decoder.abs_decoder import AbsDecoder

import pdb


class LinearDecoderGAP(AbsDecoder):
    """Linear decoder for speaker diarization"""

    def __init__(
        self,
        encoder_output_size: int,
        num_spk: int = 2,
        attention_dim=256,
        num_heads=8,
    ):
        super().__init__()
        self._num_spk = num_spk
        self.linear_decoder = torch.nn.Linear(encoder_output_size, num_spk)
        self.avg_pooling = torch.nn.AdaptiveAvgPool2d((1,encoder_output_size))


    def forward(self, input: torch.Tensor, re_embed: bool=False):
        """Forward.

        Args:
            input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        """
        #pdb.set_trace()
        input = self.avg_pooling(input).squeeze()
        #pdb.set_trace()
        output = self.linear_decoder(input)
        if re_embed == False:
            return output
        else:
            return input, output
    @property
    def num_spk(self):
        return self._num_spk
