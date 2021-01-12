from torch import nn

from model.backbones_deeplab.deeplab import DeepLab


class Deeplabv3(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super().__init__()
        self.model = DeepLab(num_classes=out_channel, output_stride=16, input_channel=in_channel)



    def forward(self, x, **kwargs):
        return self.model(x)



