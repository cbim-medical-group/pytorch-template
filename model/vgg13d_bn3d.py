import torch.nn as nn
from model.backbones.vgg2deeper_3d import VGG


class Vgg13dBn3d(nn.Module):
    def __init__(self, pretrained=False, progress=True, **kwargs):
        super().__init__()
        # self.model = _vgg('vgg16_bn', 'D', True, False, num_classes=num_classes)
        self.model = VGG("VGG13D", **kwargs)

    def forward(self, x, pred_y):
        return self.model(x, pred_y)
