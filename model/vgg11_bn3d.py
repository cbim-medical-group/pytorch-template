import torch
import torch.nn as nn


from model.backbones.vgg2_3d import VGG


class Vgg11Bn3d(nn.Module):
    def __init__(self, pretrained=False, progress=True, **kwargs):
        super().__init__()
        # self.model = _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)
        self.model = VGG("VGG11", **kwargs)

    def forward(self, x, pred_y):
        return self.model(x, pred_y)
        # return torch.clamp(self.model(x, pred_y), -1, 1)
        # Tanh = nn.Tanh()
        # return Tanh(self.model(x))
        # sigmoid = nn.Sigmoid()
        # return (sigmoid(self.model(x))-0.5)*2