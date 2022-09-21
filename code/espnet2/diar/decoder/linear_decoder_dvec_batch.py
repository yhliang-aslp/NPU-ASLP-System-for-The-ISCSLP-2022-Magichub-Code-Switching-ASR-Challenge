import torch

from espnet2.diar.decoder.abs_decoder import AbsDecoder

import pdb



class LinearDecoderMeanBatch(AbsDecoder):
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
        #self.mha = MultiHeadAttentionPooling(
        #    in_dim=encoder_output_size,
        #    att_dim=attention_dim,
        #    num_heads=num_heads
        #)

    def forward(self, input: torch.Tensor,  length: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        """
        ret = input.data.new(*input.size()).fill_(0)
        for i, l in enumerate(length):
            ret[i, :l] = input[i, :l]
        input = ret.sum(dim=1)/length.unsqueeze(1)
        output = self.linear_decoder(input)
        return output

    @property
    def num_spk(self):
        return self._num_spk
