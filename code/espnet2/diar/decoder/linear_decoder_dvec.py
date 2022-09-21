import torch

from espnet2.diar.decoder.abs_decoder import AbsDecoder

import pdb
class MultiHeadAttentionPooling(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 att_dim=128,
                 num_heads=4):
        super(MultiHeadAttentionPooling, self).__init__()
        self.num_heads = num_heads

        self.linear1 = torch.nn.Linear(in_dim, att_dim, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(att_dim, num_heads, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

        if num_heads != 1:
            self.final_linear = torch.nn.Linear(in_dim * num_heads, in_dim)

    def forward(self, inputs):
        out = self.sigmoid(self.linear1(inputs))
        out = self.softmax(self.linear2(out))

        # inputs: (b, t, d)
        # out: (b, t, n)
        out = torch.einsum('btc,bth->bch', inputs, out)
        if self.num_heads == 1:
            return out.squeeze()
        else:
            return self.final_linear(out.view(out.shape[0], -1))


class LinearDecoderMean(AbsDecoder):
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

    def forward(self, input: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        """
        #pdb.set_trace()
        #input = self.mha(input)
        input = input.mean(dim=1)
        #pdb.set_trace()
        output = self.linear_decoder(input)
        return output

    @property
    def num_spk(self):
        return self._num_spk
