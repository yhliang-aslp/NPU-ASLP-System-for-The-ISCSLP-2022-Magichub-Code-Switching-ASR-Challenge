#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math

import numpy
import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.layers.conv_dropout import Conv_Dropout
import matplotlib.pyplot as plt


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate, dropout_type='dropout'):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout_rate = dropout_rate
        if dropout_type == 'conv_dropout':
            self.dropout = Conv_Dropout(dropout_rate=dropout_rate)
        else:
            self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        if self.dropout_rate != 0.0:
            p_attn = self.dropout(self.attn)
        else:
            p_attn = self.attn
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        #import pdb;pdb.set_trace()
        return self.forward_attention(v, scores, mask)



class LegacyRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, time2).

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False, dropout_type='dropout'):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, dropout_type)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class CosineDistanceAttention(nn.Module):
    """ Compute Cosine Distance between spk decoder output and speaker profile 
    Args:
        profile_path: speaker profile file path (.npy file)
    """

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, spk_decoder_out, profile, profile_lens=None):
        """
        Args:
            spk_decoder_out(torch.Tensor):(B, T, D)
            spk_profiles(torch.Tensor):(B, N, D)
        """
        x = spk_decoder_out.unsqueeze(2)  # (B, T, 1, D)
        if profile_lens is not None:
            mask = (~make_pad_mask(profile_lens)[:, None, :]).to(profile.device)
            weights = self.softmax(F.cosine_similarity(
                x, profile.unsqueeze(1), dim=-1) * mask)  # (B, T, N)
        else:
            x = x[:, -1:, :, :]
            weights = self.softmax(F.cosine_similarity(
                x, profile.unsqueeze(1).to(x.device), dim=-1))  # (B, 1, N)
        spk_embedding = torch.matmul(weights, profile.to(weights.device))  # (B, T, D)
        return spk_embedding

class CosineDistanceAttentionAllspk(nn.Module):
    """ Compute Cosine Distance between spk decoder output and speaker profile 
    Args:
        profile_path: speaker profile file path (.npy file)
    """

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.profile = torch.tensor(numpy.load("/home/work_nfs5_ssd/yhliang/workspace/espnet_meeting/dump/raw/Train_Ali_far/allspk_emb.npy"))

    def forward(self, spk_decoder_out):
        """
        Args:
            spk_decoder_out(torch.Tensor):(B, T, D)
            spk_profiles(torch.Tensor):(B, N, D)
        """
        x = spk_decoder_out.unsqueeze(2)  # (B, T, 1, D)
        x = x[:, -1:, :, :]
        profile = self.profile.unsqueeze(0).expand(x.shape[0], -1, -1)
        weights = self.softmax(F.cosine_similarity(
            x, profile.unsqueeze(1).to(x.device), dim=-1))  # (B, 1, N)
        spk_embedding = torch.matmul(weights, profile.to(weights.device))  # (B, T, D)
        return spk_embedding

class MultiLevelMultiHeadedAttention(nn.Module):
    """Multi-level Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, dropout_type: str = 'dropout'):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        if dropout_type == 'conv_dropout':
            self.dropout = Conv_Dropout(dropout_rate=dropout_rate)
        else:
            self.dropout = nn.Dropout(p=dropout_rate)
    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, 1, time1, size).
            key (torch.Tensor): Key tensor (#batch, n, time2, size).
            value (torch.Tensor): Value tensor (#batch, n, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        n_candidate = key.size(1)
        # import pdb;pdb.set_trace()
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, n_candidate, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, n_candidate, -1, self.h, self.d_k)
        q1 = q.transpose(1, 2).unsqueeze(1).expand(n_batch, n_candidate, self.h, -1,
                                    self.d_k)  # (batch, n_candidate, head, time1, d_k)
        k = k.transpose(2, 3)  # (batch, n_candidate, head, time2, d_k)
        v = v.transpose(2, 3)  # (batch, n_candidate, head, time2, d_k)

        return q1, k, v, q

    def forward_attention1(self, value: torch.Tensor, scores: torch.Tensor,
                           mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_candidate, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_candidate, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, n_candidate, 1, time2) 

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        if mask is not None:
            # mask = mask.unsqueeze(2).eq(0)  # (batch, n_candidate 1, 1, time2)
            mask = mask.unsqueeze(2).eq(0)  # (batch, n_candidate 1, 1, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, n_candidate, head, time1, time2)
        else:
            # (batch, n_candidate, head, time1, time2)
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, n_candidate, head, time1, d_k)

        return x  # (batch, n_candidate, head, time1, d_k)

    def forward_attention2(self, value: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor
                           ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (batch, time1, head, n_candidate, d_k).
            scores (torch.Tensor): Attention score, size
                (batch, time1, head, n_candidate, n_candidate).
            mask (torch.Tensor): Mask, size 
                (batch, n_candidate)

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).eq(0) # (batch, 1, 1, 1, n_candidate)
        scores = scores.masked_fill(mask, -float('inf'))
        attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, time1, head, 1, n_candidate)
        #import pdb;pdb.set_trace()
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value).squeeze()  # (batch, time1, head, d_k)
        x = (x.contiguous().view(n_batch, -1,
                                 self.h * self.d_k)
             )
        # fig0 = plt.figure()
        # plt.plot([x for x in range(0, attn.shape[1])], torch.mean(attn[0,:,:,0,0], dim=1).cpu().numpy())
        # plt.plot([x for x in range(0, attn.shape[1])], torch.mean(attn[0,:,:,0,1], dim=1).cpu().numpy())
        # plt.plot([x for x in range(0, attn.shape[1])], torch.mean(attn[0,:,:,0,2], dim=1).cpu().numpy())
        # plt.plot([x for x in range(0, attn.shape[1])], torch.mean(attn[0,:,:,0,3], dim=1).cpu().numpy())
        # fig0.savefig("Eval-far-R8003_M8001_MS801-N_SPK8001-0060645-0061659.png")
        # exit()
        # (batch, time1, d_model)
        #import matplotlib.pyplot as plt;fig = plt.figure();plt.colorbar(plt.imshow(sentence_scores[0,:,0,0,:].cpu().numpy()));fig.savefig("1.png")
        #import matplotlib.pyplot as plt;fig = plt.figure();plt.plot([x for x in range(0, attn.shape[1])], attn[0,:,0,0,2].cpu().numpy());fig.savefig("3.png")
        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask_audio: Optional[torch.Tensor],
                mask_spk: Optional[torch.Tensor],) -> torch.Tensor:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, n_candidate, time2, size).
            value (torch.Tensor): Value tensor (#batch, n_candidate, time2, size).
            mask_audio (torch.Tensor): Mask tensor (#batch, n_candidate, 1, time2)
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, n_candidate, 1, T) shape.
            mask_spk (torch.Tensor): speaker(candidate) mask (#batch, n_candidate)


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        # import pdb;pdb.set_trace()
        q1, k, v, q2 = self.forward_qkv(query, key, value)
        scores = torch.matmul(q1, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # (batch, n_candidate, head, time1, d_k)
        phone_context = self.forward_attention1(v, scores, mask_audio)
        sentence_scores = torch.matmul(q2.unsqueeze(3), phone_context.transpose(
            1, 3).transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, time1, head, 1, n_candidate)
        # import pdb;pdb.set_trace()
        # (batch, time1, d_model)
        return self.forward_attention2(phone_context.transpose(1, 3), sentence_scores, mask_spk)


class MultiLevelMultiHeadedAttentionRegister(nn.Module):
    """Multi-level Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, channel, time1, size).
            key (torch.Tensor): Key tensor (#batch, spk_num, size).
            value (torch.Tensor): Value tensor (#batch, spk_num, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        channel = query.size(1)
        query = query.transpose(1,2) # B, T1, C, size
        q = self.linear_q(query).view(n_batch, -1, channel, self.h, self.d_k) # B, T1, C, h, d_k
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k) # B, sn, h, d_k
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k) # B, sn, h, d_k
        v2 = q.transpose(2, 3) # B, T1, h, C, d_k
        q1 = q.mean(2, keepdim=True).transpose(1, 2).transpose(2, 3) # B, 1, h, T1, d_k
        k = k.transpose(1, 2).unsqueeze(1)  # (batch, 1, head, sn, d_k)
        v = v.transpose(1, 2).unsqueeze(1)  # (batch, 1, head, sn, d_k)

        return q1, k, v, v2

    def forward_attention1(self, value: torch.Tensor, scores: torch.Tensor,
                           mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, 1, n_head, sn, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, 1, n_head, time1, sn).
            mask (torch.Tensor): Mask, size (#batch, 1, time1, spk_num) 

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, 8, time1, sn)
            #import pdb;pdb.set_trace()
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # batch, 1, n_head, time1, sn
        else:
            # batch, 1, n_head, time1, sn
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)

        return x  # batch, 1, n_head, time1, dk

    def forward_attention2(self, value: torch.Tensor, scores: torch.Tensor,
                           ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (batch, time1, head, channel, d_k).
            scores (torch.Tensor): Attention score, size
                (batch, time1, head, 1, channel).
            mask (torch.Tensor): Mask, size 

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        attn = torch.softmax(scores, dim=-1)  # (batch, time1, head, 1, channel)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value).squeeze()  # (batch, time1, head, d_k)
        x = (x.contiguous().view(n_batch, -1,
                                 self.h * self.d_k)
             )  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor],) -> torch.Tensor:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, channel, time1, size).
            key (torch.Tensor): Key tensor (#batch, 1, spk_num, size).
            value (torch.Tensor): Value tensor (#batch, 1, spk_num, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time1, spk_num)
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, channel, 1, T) shape.


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q1, k, v, v2 = self.forward_qkv(query, key, value)
        scores = torch.matmul(q1, k.transpose(-2, -1)) / math.sqrt(self.d_k) # B, 1, h, T1, sn
        phone_context = self.forward_attention1(v, scores, mask)  # batch, 1, n_head, time1, dk 
        sentence_scores = torch.matmul(phone_context.transpose(1, -2), v2.transpose(-2, -1)) / math.sqrt(self.d_k) # B, T1, h, 1, C
        # (batch, time1, d_model)
        return self.forward_attention2(v2, sentence_scores)
