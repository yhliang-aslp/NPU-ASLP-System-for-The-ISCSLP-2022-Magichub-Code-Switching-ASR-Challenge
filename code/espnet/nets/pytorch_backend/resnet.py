"""
    ResNet Block:
        BasicBlock: For ResNet18 and ResNet34
        Bottleneck: For ResNet
"""
import torch.nn as nn
import torch.nn.functional as functional


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBasicBlock, self).__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * ResNetBasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * ResNetBasicBlock.expansion)
        )

        self.short_cut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * ResNetBasicBlock.expansion:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResNetBasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ResNetBasicBlock.expansion)
            )

    def forward(self, inputs):
        return functional.relu(self.short_cut(inputs) + self.residual_function(inputs), inplace=True)


class ResNetBottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBottleNeck, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * ResNetBottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * ResNetBottleNeck.expansion),
        )

        self.short_cut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * ResNetBottleNeck.expansion:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResNetBottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * ResNetBottleNeck.expansion)
            )

    def forward(self, inputs):
        return functional.relu(self.short_cut(inputs) + self.residual_function(inputs), inplace=True)
