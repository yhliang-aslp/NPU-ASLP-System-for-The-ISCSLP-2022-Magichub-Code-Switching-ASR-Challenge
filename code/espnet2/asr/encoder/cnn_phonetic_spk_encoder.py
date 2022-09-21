import torch
import torch.nn as nn
#from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.resnet import ResNetBasicBlock


class StandardCnnPhoneticSpeakerVerification(nn.Module):
    def __init__(self, embedding_size=256):
        super(StandardCnnPhoneticSpeakerVerification, self).__init__()
        # (B, 1, 400, 80)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=2),
            ResNetBasicBlock(in_channels=64, out_channels=64),
            ResNetBasicBlock(in_channels=64, out_channels=64),
        )
        # (B, 64, 199, 39)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2),
            ResNetBasicBlock(in_channels=64, out_channels=64),
            ResNetBasicBlock(in_channels=64, out_channels=64),
        )
        # (B, 64, 99, 19)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0)),
            ResNetBasicBlock(in_channels=128, out_channels=128),
            ResNetBasicBlock(in_channels=128, out_channels=128),
        )
        # (B, 128, 99, 9)
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # (B, 256, 99, 4)
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # (B, 128, 99, 2)

    def forward(self, x):
        """
            Args:
                x: torch.Tensor, (B, T, 80)
        """
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        batch_size, channel_num, time_length, feat_dim = x.shape
        x = x.view(batch_size, -1, time_length)
        x = x.transpose(1, 2)

        return x

