from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from wenet.transformer.decoder import LTM_TransformerDecoder
from wenet.utils.common import (IGNORE_ID, add_sos_eos)
from wenet.utils.mask import make_pad_mask
import sys


class intern_TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        att_unit: int = 256,
        attention_heads: int = 2,
        linear_units: int = 1024,
        num_blocks: int = 4,
        positional_dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.reverse_weight = 0
        self.decoder = LTM_TransformerDecoder(
            vocab_size=vocab_size,
            encoder_output_size=att_unit,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            dropout_rate = 0.05,
            positional_dropout_rate = 0.1,
            self_attention_dropout_rate = 0.0,
            src_attention_dropout_rate = 0.0,
            input_layer = "embed",
            use_output_layer= True,
            normalize_before= True,
            concat_after= False,
        )
        
    def forward(self, ys_pad: torch.Tensor, ys_pad_lens: torch.Tensor) -> torch.Tensor:
        """Compute LM loss value from buffer sequences.

        Args:
            text (torch.Tensor): Input ids. (batch, len)
            text_len (torch.Tensor): Target ids. (batch, len)

        """
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(ys_pad, ys_pad_lens)
        return decoder_out

class Wenet_intern_LanguageModel(torch.nn.Module):
    def __init__(self, intern_lm: torch.nn.Module, vocab_size: int, ignore_id: int = IGNORE_ID):
        super().__init__()
        self.intern_lm = intern_lm
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.ignore_id = ignore_id

    def nll(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood(nll)

        Normally, this function is called in batchify_nll.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # text: (Batch, Length) -> x, y: (Batch, Length + 1)
        x, t = add_sos_eos(text, self.sos , self.eos, self.ignore_id)
        x_lengths = text_lengths + 1

        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y = self.intern_lm(x, x_lengths)

        # 3. Calc negative log likelihood
        # nll: (B, L)
        nll = F.cross_entropy(y.transpose(1, 2), t, reduction="none", ignore_index=self.ignore_id)
        return nll, x_lengths

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        nll, y_lengths = self.nll(text, text_lengths)
        ntokens = y_lengths.sum()
        loss = nll.sum() / ntokens
        return loss

    def score(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        eos_flag = 1,
        softmax_flag = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # text: (Batch, Length) -> x, y: (Batch, Length + 1)
        if eos_flag == 1:
            x, t = add_sos_eos(text, self.sos , self.eos, self.ignore_id)
            x_lengths = text_lengths + 1
        else:
            x = text
            x_lengths = text_lengths

        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y = self.intern_lm(x, x_lengths)
        if softmax_flag == 1:
            y = torch.softmax(y, dim=2)
        return y


def init_intern_lm_model(configs):
    # import pdb;pdb.set_trace()
    vocab_size = configs['vocab_size']
    intern_lm_type = configs.get('intern_lm')
      
    intern_lm = intern_TransformerLM(
                        vocab_size,
                        **configs['decoder_conf']
                        )

    wenet_lm = Wenet_intern_LanguageModel(
                                intern_lm=intern_lm,
                                vocab_size=vocab_size,
                                ignore_id=IGNORE_ID
                                )
    return wenet_lm
