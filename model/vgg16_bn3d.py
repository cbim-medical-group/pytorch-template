import torch.nn as nn

from model.vgg_3d import _vgg


class Vgg16Bn(nn.Module):
    def __init__(self, features, num_classes=20, init_weights=True):
        self.model = _vgg('vgg16_bn', 'D', True, False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
