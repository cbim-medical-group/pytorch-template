import torch
import torch.nn as nn
from model.backbones.vgg2deeper_3d import VGG


class Vgg19dBn3d(nn.Module):
    def __init__(self, pretrained=False, progress=True, **kwargs):
        super().__init__()
        # self.model = _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)
        self.model = VGG("VGG19D", **kwargs)

    def forward(self, x, pred_y):
        return self.model(x, pred_y)
