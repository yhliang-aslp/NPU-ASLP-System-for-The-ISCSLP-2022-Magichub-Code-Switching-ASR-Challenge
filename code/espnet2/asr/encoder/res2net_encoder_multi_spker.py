import math
import torch
import torch.nn as nn
# from espnet2.asr.encoder.abs_encoder import AbsEncoder
# from espnet.nets.pytorch_backend.resnet import ResNetBasicBlock


# from component.pooling import MultiHeadAttentionPooling
from espnet2.asr.encoder.res2net import Res2NetBlock
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
# from egs.res2net.res2net_speaker import UtteranceLevelMeanNormalization


class UtteranceLevelMeanNormalization(nn.Module):
    def __init__(self):
        super(UtteranceLevelMeanNormalization, self).__init__()

    def forward(self, x):
        """
            x: batch_size, channel_num, frame_length, feat_dim
        """
        mean = torch.mean(x, dim=2, keepdim=True)
        return x - mean


class StandardRes2NetSpeakerVerificationMultispk(nn.Module):
    """
        From the article `RESNEXT AND RES2NET STRUCTURES FOR SPEAKER VERIFICATION`
    """
    def __init__(self,
                 layers=[2, 2, 2],
                 num_filters=[64, 128, 256],
                 embedding_size=256,
                 base_width=26,
                 scale=8,
                 use_se=False,
                 se_channels=128,
                 hidden_dim=256):
        super(StandardRes2NetSpeakerVerificationMultispk, self).__init__()

        assert len(layers) == len(num_filters) == 3

        self.base_width = base_width
        self.scale = scale
        self.use_se = use_se
        self.se_channels = se_channels
        #self.utt_cmvn = UtteranceLevelMeanNormalization()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, num_filters[0], kernel_size=3, stride=2),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True)
        )
        self.block1 = self._make_layer(num_filters[0], num_filters[0], layers[0], stride=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size=3, stride=2),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True)
        )
        self.block2 = self._make_layer(num_filters[1], num_filters[1], layers[1], stride=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_filters[1], num_filters[2], kernel_size=3, stride=(1, 2), padding=(1, 0)),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True)
        )
        self.block3 = self._make_layer(num_filters[2], num_filters[2], layers[2], stride=1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_filters[2], num_filters[2], kernel_size=3, stride=(1, 2), padding=(1, 0)),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(num_filters[2], num_filters[2] // 2, kernel_size=3, stride=1, padding=(1, 0)),
            nn.BatchNorm2d(num_filters[2] // 2),
        )
        self.fc = nn.Linear(hidden_dim, embedding_size)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, planes, block_num, stride=1):
        short_cut = None
        if stride != 1 or in_planes != planes * Res2NetBlock.expansion:
            short_cut = nn.Sequential(
                nn.Conv2d(in_planes, planes * Res2NetBlock.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * Res2NetBlock.expansion)
            )

        layers = [
            Res2NetBlock(
                in_channels=in_planes,
                out_channels=planes,
                stride=stride,
                shortcut=short_cut,
                base_width=self.base_width,
                scale=self.scale,
                use_se=self.use_se,
                se_channels=self.se_channels
            )
        ]

        for i in range(1, block_num):
            layers.append(
                Res2NetBlock(
                    in_channels=in_planes,
                    out_channels=planes,
                    base_width=self.base_width,
                    scale=self.scale,
                    use_se=self.use_se,
                    se_channels=self.se_channels
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, x_lens):
        x = x.unsqueeze(1)
        # x = self.utt_cmvn(x)

        x = self.conv1(x)
        x = self.block1(x)
        x = self.conv2(x)
        x = self.block2(x)
        x = self.conv3(x)
        x = self.block3(x)
        x = self.conv4(x)
        f = self.conv5(x)

        f = f.transpose(1, 2)
        f = f.reshape(f.shape[0], f.shape[1], f.shape[2] * f.shape[3])
        f = self.fc(f)
        f_lens = ((x_lens - 1) // 2 - 1) // 2  # (B * N, )
        masks = (~make_pad_mask(f_lens)[:, :, None]).to(f.device)
        f = f * masks
        return f, f_lens
        # x = self.mha(f)
        # x = x.view(x.shape[0], -1)
        # e = self.fc(x)
        # x = self.classifier(e)
        # return x, e, f


# if __name__ == '__main__':
#     model = StandardRes2NetSpeakerVerification(layers=[2, 2, 2], num_filters=[64, 128, 256])
#     print(model)
#     print(sum([p.data.nelement() for p in model.parameters()]))
#     import torch

#     x = torch.randn([32, 1, 400, 80])
#     x, e, f = model(x)
#     print(f"outputs.shape = {x.shape}, embedding.shape = {e.shape}, frames.shape = {f.shape}")
