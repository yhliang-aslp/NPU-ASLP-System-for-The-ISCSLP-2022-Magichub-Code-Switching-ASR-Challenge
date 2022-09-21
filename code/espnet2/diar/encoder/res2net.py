import math
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self,
                 channels,
                 se_channels):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, se_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(se_channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        out = self.avg_pool(inputs)
        out = out.view(out.shape[0], out.shape[1])
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = out.view(out.shape[0], out.shape[1], 1, 1)
        return inputs * out


class Res2NetBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 shortcut=None,
                 base_width=26,
                 scale=4,
                 use_se=True,
                 se_channels=128):
        super(Res2NetBlock, self).__init__()
        self.scale = scale

        width = int(math.floor(out_channels * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(in_channels, width * scale, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=width,
                    out_channels=width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
                for _ in range(scale - 1)
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(width) for _ in range(scale - 1)]
        )

        self.conv3 = nn.Conv2d(width * scale, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        if use_se:
            self.se = SEBlock(channels=out_channels * self.expansion, se_channels=se_channels)
        else:
            self.se = nn.Identity()

        self.shortcut = shortcut

    def forward(self, inputs):
        residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        # print(f"in res2net block {out.shape}")

        xs = torch.chunk(out, self.scale, dim=1)
        out = []
        for i in range(self.scale):
            if i == 0:
                out.append(xs[i])
            elif i == 1:
                out.append(self.relu(self.bns[i - 1](self.convs[i - 1](xs[i]))))
            else:
                out.append(self.relu(self.bns[i - 1](self.convs[i - 1](xs[i] + out[-1]))))
        out = torch.cat(out, dim=1)
        # print(f"in res2net block after chunk {out.shape}")

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        # print(f"in res2net block after se {out.shape}")

        if self.shortcut is not None:
            residual = self.shortcut(inputs)
        # print(f"in res2net residual {residual.shape}")
        out += residual

        out = self.relu(out)
        return out
