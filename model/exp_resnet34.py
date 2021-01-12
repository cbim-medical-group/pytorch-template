from torch import nn

from model.backbones.exp_resnet3d import ResNet34
from model.resnet_3d import generate_model


class ExpResnet34(nn.Module):
    def __init__(self, pretrained=False, progress=True, num_classes=20, **kwargs):
        super().__init__()
        self.model = ResNet34(num_classes)

    def forward(self, x, pred_y):
        return self.model(x, pred_y)

