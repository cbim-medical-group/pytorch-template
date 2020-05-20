import torch
import torch.nn as nn

from model.vgg_3d import _vgg


class Vgg19Bn(nn.Module):
    def __init__(self, pretrained=False, progress=True, **kwargs):
        super().__init__()
        self.model = _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)

    def forward(self, x):
        return torch.clamp(self.model(x), -1, 1)
        # Tanh = nn.Tanh()
        # return Tanh(self.model(x))
        # sigmoid = nn.Sigmoid()
        # return (sigmoid(self.model(x))-0.5)*2
