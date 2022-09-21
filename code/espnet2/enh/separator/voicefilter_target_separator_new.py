# Conformer Target Separator
from collections import OrderedDict
from typing import List
from typing import Tuple
from typing import Union
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as tf
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.conformer.encoder import (
    Encoder as ConformerEncoder,  # noqa: H301
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet2.enh.separator.abs_separator import AbsSeparator
import pdb
EPSILON = torch.finfo(torch.float32).eps
def parse_1dstr(sstr: str) -> List[int]:
    return list(map(int, sstr.split(",")))

def parse_2dstr(sstr: str) -> List[List[int]]:
    return [parse_1dstr(tok) for tok in sstr.split(";")]





class VoiceFilter_new(AbsSeparator):
    def __init__(
        self,
        input_dim: int= 257,
        embedding_dim: int = 512,
        rnn_layers=2,
        rnn_units=256,
        dropout=0.0,
        K="5,2;5,2;5,2;5,2;5,2;5,2",
        S="2,1;2,1;2,1;2,1;2,1;2,1",
        C="2,16,32,64,128,256,256",
        P="2,2,2,2,2,2",
        O="1,0;1,0;1,0;1,0;1,0;1,0",
        bidirectional: bool = True,
        num_spk: int = 1,
    ):
        super().__init__()
        K = parse_2dstr(K)
        S = parse_2dstr(S)
        C = parse_1dstr(C)
        P = parse_1dstr(P)
        O = parse_2dstr(O)
        self._num_spk = num_spk
        #self.speaker_down = torch.nn.Linear(512,256)
        #embedding_dim = 256
        out_channel= 32
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skipcnn = nn.ModuleList()
        self.skipcnn_batchnorm = nn.ModuleList()
        
        for idx in range(len(C) - 1):
            self.encoder.append(
                    nn.Sequential(
                        nn.ConstantPad2d([1, 0, 0, 0], value=0.),
                        nn.Conv2d(C[idx],
                                  C[idx+1],
                                  kernel_size=K[idx],
                                  stride=S[idx],
                                  padding=(P[idx], 0),
                        ),
                        nn.BatchNorm2d(C[idx+1]),
                        nn.LeakyReLU(),
                    )
                )
            self.skipcnn.append(
                nn.Conv2d(C[idx+1],
                          C[idx+1],
                          kernel_size=(1, 1),
                          stride=(1, 1),
                          padding=(0, 0),
                ),
            )
            self.skipcnn_batchnorm.append(
                nn.BatchNorm2d(C[idx+1]),
            )
        hidden_dim = 512//(2**(len(C)))
        self.enhance = nn.LSTM(input_size=hidden_dim*C[-1]+embedding_dim,
                               hidden_size=rnn_units,
                               num_layers=rnn_layers,
                               dropout=dropout,
                               bidirectional=bidirectional,
                               batch_first=True,
                               )
        fac = 2 if bidirectional else 1
        self.transform = nn.Linear(rnn_units*fac, hidden_dim*C[-1])        
        for idx in range(len(C) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(C[idx]*2,
                                           C[idx-1],
                                           kernel_size=K[idx-1],
                                           stride=S[idx-1],
                                           padding=(P[idx-1], 0),
                                           output_padding=O[idx-1],
                        ),
                        nn.BatchNorm2d(C[idx-1]),
                        nn.LeakyReLU(),
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(C[idx]*2,
                                           C[idx-1],
                                           kernel_size=K[idx-1],
                                           stride=S[idx-1],
                                           padding=(P[idx-1], 0),
                                           output_padding=O[idx-1],
                        ),
                    )
                )    
 
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
        r_spec = torch.transpose(input.real, 1, 2)[:, 1:]
        i_spec = torch.transpose(input.imag, 1, 2)[:, 1:]
        emb = xvectors
        
        
        encoder_out = []
        # N x 2 x F x T
        out = th.stack((r_spec, i_spec), dim=1)

        for idx in range(len(self.encoder)):
            # N x C' x F' x T
            out = self.encoder[idx](out)
            out1 = th.relu(out)
            out1 = out1 + self.skipcnn[idx](out1)
            out1 = self.skipcnn_batchnorm[idx](out1)
            encoder_out.append(out1)

        N, C, F, T = out.shape
        # N x T x C x F
        out = out.permute(0, 3, 1, 2).contiguous()
        # N x T x CF
        out = th.reshape(out, [N, T, -1])

        # N x E => N x 1 x E => N x T x E
        if len(xvectors.shape) == 2:
            emb = torch.unsqueeze(emb, 1).repeat(1, T,1)
        else:
            emb = emb.repeat(1, T, 1)
        
        # N x T x (CF + E)
        out = th.cat([out, emb], dim=2)
        # N x T x H
        out, _ = self.enhance(out)
        # N x T x CF
        out = self.transform(out)
        # N x T x C x F
        out = th.reshape(out, [N, T, C, -1])
        # N x C x F x T
        out = out.permute(0, 2, 3, 1)

        for idx in range(len(self.decoder)):
            # N x C' x F' x T
            out = th.cat([out, encoder_out[-1-idx]], dim=1)
            out = self.decoder[idx](out)
            out = out[..., :-1]

        # N x 2 x F x T => N x 1 x F x T
        r_mask, i_mask = th.chunk(out, 2, 1)

        # N x F x T
        r_mask = r_mask.squeeze(1)
        i_mask = i_mask.squeeze(1)


        # compress
        r_mask = 10 * ((1 - th.exp(-0.1 * r_mask)) / (1 + th.exp(-0.1 * r_mask)))
        i_mask = 10 * ((1 - th.exp(-0.1 * i_mask)) / (1 + th.exp(-0.1 * i_mask)))
        
        # N x F x T
        r_out_spec = r_mask * r_spec - i_mask * i_spec
        i_out_spec = r_mask * i_spec + i_mask * r_spec
        r_out_spec = tf.pad(r_out_spec, [0, 0, 1, 0])
        i_out_spec = tf.pad(i_out_spec, [0, 0, 1, 0])
        r_mask = tf.pad(r_mask, [0, 0, 1, 0])
        i_mask = tf.pad(i_mask, [0, 0, 1, 0])

        masked = [ComplexTensor(torch.transpose(r_out_spec, 1, 2),torch.transpose(i_out_spec, 1, 2))]
        masks= [torch.transpose(r_mask,1,2),torch.transpose(i_mask,1,2)]
        
        if rt_mask:
            return masked, ilens, masks
        else:
            return masked, ilens
        
    @property
    def num_spk(self):
        return self._num_spk
