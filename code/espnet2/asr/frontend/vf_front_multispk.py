import copy
from typing import Optional
from typing import Tuple
from typing import Union

import humanfriendly
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types
#import torchaudio

from espnet.nets.pytorch_backend.frontends.voice_filter_multispk import Frontend
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_kwargs import get_default_kwargs
import random
from torch.utils.checkpoint import checkpoint
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

class VFFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Mel-Fbank -> CMVN
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)

        if apply_stft:
            self.stft = Stft(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=center,
                window=window,
                normalized=normalized,
                onesided=onesided,
            )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        if frontend_conf is not None:
            self.frontend = Frontend(**frontend_conf)
        else:
            self.frontend = None
        #self.frontend = None

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels

    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor, e: torch.Tensor = None, dia_mask_perframe: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input.dim() == 3:
            # h: (B, N, C) -> h: (B, N)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input.size(2))
                input = input[:, :, ch]
            else:
                # Use the first channel
                input = input[:, :, 0]
        max_norm = torch.max(torch.abs(input), dim=1)[0].unsqueeze(1)
        input = (input + 1e-12) / (max_norm + 1e-6)
        # import pdb;pdb.set_trace()
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        if self.stft is not None:
            input_stft, feats_lens = self._compute_stft(input, input_lengths)
        else:
            input_stft = ComplexTensor(input[..., 0], input[..., 1])
            feats_lens = input_lengths
        # if input_stft.shape[0] == 1:
        if self.frontend is not None and e is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            # e.requires_grad_()
            # wav = checkpoint(self.frontend, input_stft.transpose(1, 2).real, input_stft.transpose(1, 2).imag, e, max_norm)
            wav = self.frontend(input_stft.transpose(1, 2).real, input_stft.transpose(1, 2).imag, e, max_norm)
            if dia_mask_perframe is not None:
                masks = (~make_pad_mask(input_lengths)[:, :wav.shape[1]]).to(wav.device)
                dia_mask_perframe = dia_mask_perframe[:, :wav.shape[1]]
                frontend_l2_loss = (torch.linalg.norm((1 - dia_mask_perframe) * wav * masks)) / torch.sum((1 - dia_mask_perframe) * masks) * 1e6
            else:
                frontend_l2_loss = None
            input_stft, feats_lens = self._compute_stft(wav, input_lengths)
        # else:
        #     index1 = random.sample([x for x in range(input_stft.shape[0])], input_stft.shape[0] // 2)
        #     index2 = [x for x in range(input_stft.shape[0]) if x not in index1]
        #     part1 = input_stft[index1, :, :]
        #     e = e[index1, :]
        #     part2 = input_stft[index2, :, :]
        #     # 2. [Option] Speech enhancement
        #     if self.frontend is not None:
        #         assert isinstance(input_stft, ComplexTensor), type(input_stft)
        #         # input_stft: (Batch, Length, [Channel], Freq)
        #         part1, _, mask = self.frontend(part1, feats_lens, e)
        #     part2 = part2.real ** 2 + part2.imag ** 2
        #     input_power = torch.cat([part1, part2], 0)
        # 3. [Multi channel case]: Select a channel

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real ** 2 + input_stft.imag ** 2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)
        if e is None:
            frontend_l2_loss = None

        return input_feats, feats_lens, frontend_l2_loss

    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        
        input_stft, feats_lens = self.stft(input, input_lengths)
        # import pdb;pdb.set_trace()
        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens
