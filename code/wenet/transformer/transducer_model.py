from collections import defaultdict
from typing import List, Optional, Tuple
import pdb
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from torchaudio.functional import rnnt_loss
from torch.nn.functional import log_softmax

from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import (TransformerDecoder,
                                       BiTransformerDecoder)
from wenet.transformer.encoder import ConformerEncoder
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.cmvn import load_cmvn
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class JointNet(torch.nn.Module):
    def __init__(
        self, 
        input_size,
        inner_dim, 
        vocab_size,
        joint_func = "add",
        pre_linear = True,
        post_linear = False
    ):
        super().__init__()
        self.pre_linear = pre_linear
        self.post_linear = post_linear
        self.enc_ffn = nn.Linear(input_size,inner_dim)
        self.prd_ffn = nn.Linear(input_size,inner_dim)
        self.forward_layer = Linear(input_size, inner_dim, bias=True)
        self.tanh = nn.Tanh()
        self.project_layer = Linear(inner_dim, vocab_size, bias=True)
        self.joint_func = joint_func
    
    def forward(self, enc_state, dec_state):
        if self.pre_linear:
            enc_state = self.enc_ffn(enc_state)
            dec_state = self.prd_ffn(dec_state)
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)
            t = enc_state.size(1)
            u = dec_state.size(2)
            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()
        
        if self.joint_func == "cat":
            concat_state = torch.cat((enc_state, dec_state), dim=-1)
        elif self.joint_func == "add":
            concat_state = torch.add(enc_state, dec_state)
        else:
            print("wtf")
            concat_state = torch.add(enc_state, dec_state)
        if self.post_linear:
            outputs = self.forward_layer(concat_state)
        else:
            outputs = concat_state
        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)
        outputs = nn.functional.log_softmax(outputs,dim=-1)
        return outputs


class TransducerDecoder(torch.nn.Module):
    "Transducer decoder"
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        joint_inner_dim: int = 2048,
        num_blocks: int = 1,
        dropout_rate: float = 0.1,
        layer_drop_rate: float = 0.0,
        embed_pos_enc: bool = False,
        padding_idx: int = IGNORE_ID,
        prediction_layernorm: bool= True,
        predictor_embed_size: int = -1,
        predictor_hidden_size: int = -1,
        predictor_output_size: int = -1,
        embed_dropout: float = 0.1,
        joint_func: str = "add",
        pre_linear = True,
        post_linear = False
    ):
        super().__init__()
        embed_dim = encoder_output_size if predictor_embed_size == -1 else predictor_embed_size
        input_size = embed_dim if predictor_embed_size == -1 else predictor_embed_size
        hidden_size = embed_dim if predictor_hidden_size == -1 else predictor_hidden_size
        output_size = embed_dim if predictor_output_size == -1 else predictor_output_size
        self.text_embedding = self.build_embedding(vocab_size, embed_dim , padding_idx)
        self.padding_idx = padding_idx
        self.prediction = nn.LSTM(
            input_size = input_size, 
            hidden_size = hidden_size,
            num_layers = num_blocks,
            dropout = dropout_rate,
            bias = True,
        )
        self.dropout = nn.Dropout(embed_dropout)
        self.project = nn.Linear(hidden_size, output_size)
        

        if prediction_layernorm :
            self.prediction_norm = nn.LayerNorm(output_size, eps=1e-5)

        self.joint_model = JointNet(embed_dim, joint_inner_dim, vocab_size, joint_func, pre_linear, post_linear)
        self.pred_norm = prediction_layernorm
        self.layer_drop_rate = layer_drop_rate

    def forward(
        self,
        memory: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor
    ):
        tgt = ys_in_pad
        x = self.text_embedding(tgt)
        x = self.dropout(x)
        rand = torch.rand(1)
        if not self.training or rand > self.layer_drop_rate:
            x,(_,_) = self.prediction(x.transpose(0,1))
            x = x.transpose(0,1)
        else:
            x =x 
        x = self.project(x)
        if self.pred_norm :
            x = self.prediction_norm(x)
        else:
            x = x
        x = self.joint_model(memory, x)
        return x

    def forward_one_step(
        self,
        memory: torch.Tensor,
        pre_ys: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor
    ):
        
        x = self.text_embedding(pre_ys)
        x = self.dropout(x)
        x = x.unsqueeze(0)
        x,(h_1, c_1) = self.prediction(x,(h,c))
        x =x.transpose(0,1)
        x = self.project(x)
        if self.pred_norm:
            x = self.prediction_norm(x)
        x = self.joint_model(memory, x)
        return x, (h_1,c_1)

    def build_embedding(
        self, 
        vocab_size: int, 
        embed_dim: int,
        padding_idx: int
    ):
        emb = Embedding(vocab_size, embed_dim, padding_idx)
        return emb

class Sequence():
    def __init__(
        self,
        hyp:torch.Tensor,
        score,
        h_0:torch.Tensor,
        c_0: torch.Tensor
    ):
        self.hyp = hyp
        self.score = score
        self.h_0 = h_0
        self.c_0 = c_0
class TransducerModel(torch.nn.Module):
    "Transducer model"
    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransducerDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        transducer_weight: float = 0.7, 
        ignore_id: int = IGNORE_ID
    ):
        super().__init__()
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        
        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc

        self.blank = 0
        
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. Transducer-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_transducer_loss(encoder_out, encoder_mask,
                                                    text, text_lengths)
        else:
            loss_att = None

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
        return loss, loss_att, loss_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        ys_in_pad, ys_out_pad = self.add_sos(ys_pad, self.sos, self.ignore_id)
        ys_in_lens = ys_pad_lens 
        xs_in_lens = encoder_mask.sum(-1).squeeze().int()
        ys_out_pad = ys_out_pad.int()

        # 1. Forward decoder
        decoder_out = self.decoder(encoder_out, ys_in_pad, ys_in_lens)
        # 2. Compute attention loss
        loss_rnnt = self.criterion_transducer(decoder_out, ys_out_pad, xs_in_lens, ys_in_lens)
        
        # TO DO: validation do greedy search
        
        return loss_rnnt, loss_rnnt

    def criterion_transducer(
        self,
        x,
        y,
        x_len,
        y_len,
    ):
        return rnnt_loss(x, y, x_len, y_len, blank=self.blank, reduction="mean")


    def add_sos(self, ys_pad: torch.Tensor, sos: int,
                ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        eos = sos
        _sos = torch.tensor([sos],
                            dtype=torch.long,
                            requires_grad=False,
                            device=ys_pad.device)
        
        ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
        ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
        return self.pad_list(ys_in, eos), ys_pad
    
    def pad_list(self, xs: List[torch.Tensor], pad_value: int):
        """Perform padding for the list of tensors.
    
        Args:
            xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
            pad_value (float): Value for padding.
    
        Returns:
            Tensor: Padded tensor (B, Tmax, `*`).
    
        Examples:
            >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
            >>> x
            [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
            >>> pad_list(x, 0)
            tensor([[1., 1., 1., 1.],
                    [1., 1., 0., 0.],
                    [1., 0., 0., 0.]])
    
        """
        n_batch = len(xs)
        max_len = max([x.size(0) for x in xs])
        pad = torch.zeros(n_batch, max_len, dtype=xs[0].dtype, device=xs[0].device)
        pad = pad.fill_(pad_value)
        for i in range(n_batch):
            pad[i, :xs[i].size(0)] = xs[i]
    
        return pad
    
    
    
    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    def beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        beam_size: int = 5,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False
    ):
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]
        assert batch_size == 1
        # TO DO: batch parallel decoding

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  # (B*N, 1, max_len)

        beam_init = [Sequence(
            hyp = [self.sos],
            score = torch.tensor(0.0),
            h_0 = torch.zeros([1, 1, encoder_dim],dtype=torch.float,device=device),
            c_0 = torch.zeros([1, 1, encoder_dim],dtype=torch.float,device=device)
        ) for  i in range(beam_size)]

        for  i in range(maxlen):

            # hype build
            input_hyp = [ s.hyp[-1]  for s in beam_init]
            hyps_last = [ s.hyp for  s in beam_init]
            input_hyp_tensor = torch.tensor(input_hyp, dtype=torch.int, device=device)
            
            h_0 = torch.cat(
                [s.h_0 for s in beam_init], dim = 1
            ).to(device)
            c_0 = torch.concat(
                [s.c_0 for s in beam_init], dim = 1
            ).to(device)

            scores = torch.concat(
                [s.score.unsqueeze(0) for s in beam_init], dim = 0
            ).to(device)
            
            logp, (h_1, c_1) = self.decoder.forward_one_step(
                encoder_out[:,i,:].unsqueeze(1), input_hyp_tensor, h_0, c_0
            )# logp: (N, 1, 1, vocab_size)
            logp = logp.squeeze(1).squeeze(2) # logp: (N, vocab_size)
            # first beam prune
            top_k_logp, top_k_index = logp.topk(beam_size)  # (N, N)
            scores = torch.add(scores, top_k_logp)
            # prefix fusion
            beam_A = []
            for j in range(beam_size):
                # update seq
                base_seq = beam_init[j]
                for t in range(beam_size):
                    if top_k_index[j,t] != self.blank:
                        hyp_new = base_seq.hyp.copy()
                        hyp_new.append(top_k_index[j,t].item())
                        new_seq = Sequence(
                            hyp = hyp_new,
                            score = scores[j,t],
                            h_0 = h_1[:,j,:].unsqueeze(1),
                            c_0 = c_1[:,j,:].unsqueeze(1)
                        )
                        
                        beam_A.append(new_seq)
                    else:
                        new_seq = Sequence(
                            hyp = base_seq.hyp.copy(),
                            score = scores[j,t],
                            h_0 = h_0[:,j,:].unsqueeze(1),
                            c_0 = c_0[:,j,:].unsqueeze(1)
                        )

                        beam_A.append(new_seq)

            fusion_A = [beam_A[0]]
            for j in range(1,beam_size*beam_size):
                s1 = beam_A[j]
                if_add = True
                for t in range(len(fusion_A)):
                    if s1.hyp == fusion_A[t].hyp:
                        fusion_A[t].score = torch.add(fusion_A[t].score, s1.score)
                        if_add = False
                        break
                if if_add:   
                    fusion_A.append(s1)
            fusion_A.sort(key=lambda x:x.score, reverse=True)
            beam_init = fusion_A[:beam_size]

        # 3. Select best of best
        best_hyps = beam_init[0].hyp
        best_scores = beam_init[0].score
        return best_hyps, best_scores
