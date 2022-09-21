from typing import List
from typing import Union

import numpy
import torch
import torch.nn as nn


class Frontend(nn.Module):
    def __init__(
        self,
        bf_weight,
    ):
        super().__init__()
        self.bf_weight = torch.load(bf_weight)[:, 2: -2, :, :]

    def forward(
        self, x: torch.Tensor, ilens: Union[torch.LongTensor, numpy.ndarray, List[int]], direction_mask: torch.Tensor = None,
    ):
        assert len(x) == len(ilens), (len(x), len(ilens))
        # (B, T, F) or (B, T, C, F)
        if x.dim() not in (3, 4):
            raise ValueError(f"Input dim must be 3 or 4: {x.dim()}")
        if not torch.is_tensor(ilens):
            ilens = torch.from_numpy(numpy.asarray(ilens)).to(x.device)
        mask = None
        x = x.permute(0, 2, 3, 1)  # (B, C, F, T)
        
        weight = torch.complex(self.bf_weight[0, ...], self.bf_weight[1, ...])  # (D, C, F)
        x = (weight[..., None].conj().unsqueeze(0).to(x.device)
            * x.unsqueeze(1)).sum(dim=2)  # (B, D, F, T)

        return x.transpose(1, 2).transpose(1, 3), ilens, mask
