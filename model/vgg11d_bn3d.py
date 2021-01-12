import torch
import torch.nn as nn

import torch.nn.functional as F

from model.backbones.vgg2deeper_3d import VGG


class Vgg11dBn3d(nn.Module):
    def __init__(self, pretrained=False, progress=True, **kwargs):
        super().__init__()
        # self.model = _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)
        self.model = VGG("VGG11D", **kwargs)

    def forward(self, x, pred_y):
        x = self.model(x, pred_y)
        return torch.clamp(x, -1, 1)
        # return torch.clamp(self.model(x, pred_y), -1, 1)
        # Tanh = nn.Tanh()
        # return Tanh(self.model(x))
        # sigmoid = nn.Sigmoid()
        # return (sigmoid(self.model(x))-0.5)*2
