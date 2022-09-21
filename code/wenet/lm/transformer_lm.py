from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.encoder import TransformerEncoder
from wenet.utils.common import (IGNORE_ID, add_sos_eos)
from wenet.utils.mask import make_pad_mask
import sys


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pos_enc: str = None,
        embed_unit: int = 128,
        att_unit: int = 256,
        head: int = 2,
        unit: int = 1024,
        layer: int = 4,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_unit) # embed input is not onehot
        self.encoder = TransformerEncoder(
            input_size=embed_unit,
            output_size=att_unit,
            attention_heads=head,
            linear_units=unit,
            num_blocks=layer,
            dropout_rate=dropout_rate,
            input_layer="linear",
            pos_enc_layer_type=pos_enc,
            use_dynamic_chunk=False,
            use_dynamic_left_chunk = False,
            static_chunk_size = 1,
        )
        self.decoder = nn.Linear(att_unit, vocab_size)
        
    def forward(self, text: torch.Tensor, text_len: torch.Tensor) -> torch.Tensor:
        """Compute LM loss value from buffer sequences.

        Args:
            text (torch.Tensor): Input ids. (batch, len)
            text_len (torch.Tensor): Target ids. (batch, len)

        """
        x = self.embed(text)
        h, _ = self.encoder(x, text_len)
        y = self.decoder(h) # B*L*Vocab_size
        return y


class WenetLanguageModel(torch.nn.Module):
    def __init__(self, lm: torch.nn.Module, vocab_size: int, ignore_id: int = IGNORE_ID):
        super().__init__()
        self.lm = lm
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
        y = self.lm(x, x_lengths)

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
        y = self.lm(x, x_lengths)
        if softmax_flag == 1:
            y = torch.softmax(y, dim=2)
        return y


def init_lm_model(configs):
    # import pdb;pdb.set_trace()
    vocab_size = configs['vocab_size']

    lm_type = configs.get('lm')
    if lm_type != 'transformer':
        print("Error: only support transformer LM")
        sys.exit()
      
    lm = TransformerLM(
                        vocab_size,
                        **configs['lm_conf']
                        )

    wenet_lm = WenetLanguageModel(
                                lm=lm,
                                vocab_size=vocab_size,
                                ignore_id=IGNORE_ID
                                )
    return wenet_lm
