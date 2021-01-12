import torch
from torch import nn

from model.resnet_3d import generate_model


class Resnet34_3d(nn.Module):
    def __init__(self, pretrained=False, progress=True, num_classes=20, n_input_channels=3, **kwargs):
        super().__init__()
        self.model = generate_model(34, n_classes=num_classes, n_input_channels=n_input_channels).cuda()

    def forward(self, x, pred_y):

        x = self.model(x, pred_y)
        return torch.clamp(x, -1, 1)
        # return self.model(x)
        # return torch.clamp(self.model(x), -1, 1)
        # Tanh = nn.Tanh()
        # return Tanh(self.model(x))
        # sigmoid = nn.Sigmoid()
        # return (sigmoid(self.model(x))-0.5)*2
