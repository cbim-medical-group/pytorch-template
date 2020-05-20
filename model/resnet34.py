
from torch import nn
from torchvision.models import resnet101, resnet34


class Resnet34(nn.Module):
    def __init__(self, num_classes=20, **kwargs):
        super().__init__()
        self.resnet34 = resnet34(pretrained=False, num_classes=num_classes)

    def forward(self, x):

        return self.resnet34(x)

