import torch

from espnet2.diar.decoder.abs_decoder import AbsDecoder
import torch.nn as nn

import pdb

class StaticsPooling(nn.Module):
    expansion = 2

    def __init__(self):
        super(StaticsPooling, self).__init__()

    def forward(self, inputs):
        mean = inputs.mean(dim=1)
        std = inputs.std(dim=1)
        return torch.cat((mean, std), dim=1)


class AttentionStatisticPooling(nn.Module):
    expansion = 2

    def __init__(self, hidden_size, attention_size=128, head_num=4):
        super(AttentionStatisticPooling, self).__init__()
        self.key = nn.Linear(hidden_size, attention_size, bias=True)
        self.tanh = nn.Tanh()
        self.value = nn.Linear(attention_size, head_num, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        """

        :param inputs: B \times T \times D
        :return: B \times D*2
        """
        outputs = self.key(inputs)
        outputs = self.tanh(outputs)
        outputs = self.value(outputs)
        outputs = self.softmax(outputs)
        outputs = outputs.transpose(-1, -2)

        mean = torch.sum(torch.bmm(outputs, inputs), dim=1)
        residuals = torch.sum(torch.bmm(outputs, inputs ** 2), dim=1) - mean ** 2
        stdv = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, stdv], dim=1)



class LinearDecoderAttentionStatistic(AbsDecoder):
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
        self.fc = nn.Linear(encoder_output_size*2, encoder_output_size)
        self.mha = AttentionStatisticPooling(
            hidden_size=encoder_output_size,
            attention_size=attention_dim,
            head_num=num_heads
        )

    def forward(self, input: torch.Tensor, re_embed: bool=False):
        """Forward.

        Args:
            input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        """
        #pdb.set_trace()
        input = self.mha(input)
        #pdb.set_trace()
        input = self.fc(input)
        #pdb.set_trace()
        output = self.linear_decoder(input)
        if re_embed == False:
            return output
        else:
            return input, output
    @property
    def num_spk(self):
        return self._num_spk
