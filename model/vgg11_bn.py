import torch
import torch.nn as nn

from model.backbones.vgg2 import VGG


class Vgg11Bn(nn.Module):
    def __init__(self, pretrained=False, progress=True, **kwargs):
        super().__init__()
        self.model = VGG("VGG11")

    def forward(self, x):
        return self.model(x)
        # return torch.clamp(self.model(x), -1, 1)
        # Tanh = nn.Tanh()
        # return Tanh(self.model(x))
        # sigmoid = nn.Sigmoid()
        # return (sigmoid(self.model(x))-0.5)*2
