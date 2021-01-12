
from torch import nn

from model.backbones.resnet import ResNet101


class Resnet101(nn.Module):
    def __init__(self, num_classes=20, in_channel=1, **kwargs):
        super().__init__()
        self.resnet101 = ResNet101(num_classes, in_channel)

    def forward(self, x):

        return self.resnet101(x)

