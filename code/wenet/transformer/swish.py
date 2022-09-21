#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Swish() activation function for Conformer."""

import torch
import torch.nn as nn

class Swish(torch.nn.Module):
    """Construct an Swish object."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swish activation function."""
        return x * torch.sigmoid(x)

# class BasicNorm(torch.nn.Module):
#     """
#     This is intended to be a simpler, and hopefully cheaper, replacement for
#     LayerNorm.  The observation this is based on, is that Transformer-type
#     networks, especially with pre-norm, sometimes seem to set one of the
#     feature dimensions to a large constant value (e.g. 50), which "defeats"
#     the LayerNorm because the output magnitude is then not strongly dependent
#     on the other (useful) features.  Presumably the weight and bias of the
#     LayerNorm are required to allow it to do this.
#     So the idea is to introduce this large constant value as an explicit
#     parameter, that takes the role of the "eps" in LayerNorm, so the network
#     doesn't have to do this trick.  We make the "eps" learnable.
#     Args:
#        num_channels: the number of channels, e.g. 512.
#       channel_dim: the axis/dimension corresponding to the channel,
#         interprted as an offset from the input's ndim if negative.
#         shis is NOT the num_channels; it should typically be one of
#         {-2, -1, 0, 1, 2, 3}.
#        eps: the initial "epsilon" that we add as ballast in:
#              scale = ((input_vec**2).mean() + epsilon)**-0.5
#           Note: our epsilon is actually large, but we keep the name
#           to indicate the connection with conventional LayerNorm.
#        learn_eps: if true, we learn epsilon; if false, we keep it
#          at the initial value.
#     """

#     def __init__(
#         self,
#         num_channels: int,
#         channel_dim: int = -1,
#         eps: float = 0.25,
#         learn_eps: bool = True,
#     ) -> None:
#         super(BasicNorm, self).__init__()
#         self.num_channels=num_channels
#         self.channel_dim=channel_dim
#         if learn_eps:
#             self.eps = nn.Parameter(torch.tensor(eps).log().detach())
#         else:
#             self.register_buffer("eps", torch.tensor(eps).log().detach())

#     def forward(self, x: Tensor) -> Tensor:
#         assert x.shape[self.channel_dim] == self.num_channels
#         scales = (
#             torch.mean(x ** 2, dim=self.channel_dim, keepdim=True)
#             + self.eps.exp()
#         ) ** -0.5
#         return x * scales

# class ActivationBalancer(torch.nn.Module):
#     """
#     Modifies the backpropped derivatives of a function to try to encourage, for
#     each channel, that it is positive at least a proportion `threshold` of the
#     time.  It does this by multiplying negative derivative values by up to
#     (1+max_factor), and positive derivative values by up to (1-max_factor),
#     interpolated from 1 at the threshold to those extremal values when none
#     of the inputs are positive.
#     Args:
#            channel_dim: the dimension/axis corresponding to the channel, e.g.
#                -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
#            min_positive: the minimum, per channel, of the proportion of the time
#                that (x > 0), below which we start to modify the derivatives.
#            max_positive: the maximum, per channel, of the proportion of the time
#                that (x > 0), above which we start to modify the derivatives.
#            max_factor: the maximum factor by which we modify the derivatives for
#               either the sign constraint or the magnitude constraint;
#               e.g. with max_factor=0.02, the the derivatives would be multiplied by
#               values in the range [0.98..1.02].
#            min_abs:  the minimum average-absolute-value per channel, which
#               we allow, before we start to modify the derivatives to prevent
#               this.
#            max_abs:  the maximum average-absolute-value per channel, which
#                we allow, before we start to modify the derivatives to prevent
#                this.
#     """

#     def __init__(
#         self,
#         channel_dim: int,
#         min_positive: float = 0.05,
#         max_positive: float = 0.95,
#         max_factor: float = 0.01,
#         min_abs: float = 0.2,
#         max_abs: float = 100.0,
#     ):
#         super(ActivationBalancer, self).__init__()
#         self.channel_dim = channel_dim
#         self.min_positive = min_positive
#         self.max_positive = max_positive
#         self.max_factor = max_factor
#         self.min_abs = min_abs
#         self.max_abs = max_abs

#     def forward(self, x: Tensor) -> Tensor:
#         if torch.jit.is_scripting():
#             return x
#         else:
#             return ActivationBalancerFunction.apply(
#                 x,
#                 self.channel_dim,
#                 self.min_positive,
#                 self.max_positive,
#                 self.max_factor,
#                 self.min_abs,
#                 self.max_abs,
#             )