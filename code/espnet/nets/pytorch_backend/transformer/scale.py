import torch


class ScaleLayer(torch.nn.Module):
    def __init__(self, input_dim=256):
        super(ScaleLayer, self).__init__()
        self.scale = torch.nn.Parameter(torch.ones(1, 1, input_dim))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, input_dim))

    def forward(self, x):
        return x * self.scale + self.bias
