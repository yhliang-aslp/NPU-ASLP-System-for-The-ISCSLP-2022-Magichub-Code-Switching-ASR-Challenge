import math
import torch
import torch.nn as nn

from espnet2.diar.encoder.res2net import Res2NetBlock

import pdb

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self,
                 in_dim,
                 att_dim=128,
                 num_heads=4):
        super(MultiHeadAttentionPooling, self).__init__()
        self.num_heads = num_heads

        self.linear1 = nn.Linear(in_dim, att_dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(att_dim, num_heads, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        if num_heads != 1:
            self.final_linear = nn.Linear(in_dim * num_heads, in_dim)

    def forward(self, inputs):
        out = self.sigmoid(self.linear1(inputs))
        out = self.softmax(self.linear2(out))

        # inputs: (b, t, d)
        # out: (b, t, n)
        out = torch.einsum('btc,bth->bch', inputs, out)
        if self.num_heads == 1:
            return out.squeeze()
        else:
            return self.final_linear(out.view(out.shape[0], -1))



class StandardRes2NetSpeakerVerificationAttention(nn.Module):
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
                 hidden_dim=256,
                 num_heads=8):
        super(StandardRes2NetSpeakerVerificationAttention, self).__init__()

        assert len(layers) == len(num_filters) == 3

        self.base_width = base_width
        self.scale = scale
        self.use_se = use_se
        self.se_channels = se_channels
        self.embedding_size = embedding_size

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
        self.mha = MultiHeadAttentionPooling(
            in_dim=hidden_dim,
            att_dim=hidden_dim,
            num_heads=num_heads
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

    def output_size(self) -> int:
        return self.embedding_size

    def forward(self, x):
        # torch.Size([4, 376, 80])
        x = x.unsqueeze(1)
        # torch.Size([4, 1, 376, 80])
        #pdb.set_trace()
        x = self.conv1(x)
        #pdb.set_trace()
        # torch.Size([4, 64, 187, 39])
        x = self.block1(x)
        #pdb.set_trace()
        # torch.Size([4, 64, 187, 39])
        x = self.conv2(x)
        # torch.Size([4, 128, 93, 19])
        x = self.block2(x)
        # torch.Size([4, 128, 93, 19])
        x = self.conv3(x)
        # torch.Size([4, 256, 93, 9])
        x = self.block3(x)
        # torch.Size([4, 256, 93, 9])
        x = self.conv4(x)
        #pdb.set_trace()
        # torch.Size([4, 256, 93, 4])
        f = self.conv5(x)
        # torch.Size([4, 128, 93, 2])
        f = f.transpose(1, 2)
        # torch.Size([4, 93, 128, 2])
        #pdb.set_trace()
        f = f.reshape(f.shape[0], f.shape[1], f.shape[2] * f.shape[3])
        # torch.Size([4, 93, 256])
        #pdb.set_trace()
        x = self.mha(f)
        x = x.view(x.shape[0], -1)
        e = self.fc(x)
        # torch.Size([4, 93, 256])
        return e


# if __name__ == '__main__':
#     model = StandardRes2NetSpeakerVerification(layers=[2, 2, 2], num_filters=[64, 128, 256])
#     print(model)
#     print(sum([p.data.nelement() for p in model.parameters()]))
#     import torch

#     x = torch.randn([32, 1, 400, 80])
#     x, e, f = model(x)
#     print(f"outputs.shape = {x.shape}, embedding.shape = {e.shape}, frames.shape = {f.shape}")
