import torch
import torch.nn as nn
import torch.nn.functional as F

# Transfer Learning ResNet as Encoder part of UNet
from model.resnet_multi_channel import resnet34
from model.unet import ConvUpBlock


class ResUnet34(nn.Module):
    def __init__(self, out_channel=2, pretrained=True, fixed_feature=False, in_channel=3):
        super().__init__()
        # load weight of pre-trained resnet
        self.resnet = resnet34(pretrained=pretrained, in_channel=in_channel)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # up conv
        l = [64, 64, 128, 256, 512]
        self.u5 = ConvUpBlock(l[4], l[3], dropout_rate=0.1)
        self.u6 = ConvUpBlock(l[3], l[2], dropout_rate=0.1)
        self.u7 = ConvUpBlock(l[2], l[1], dropout_rate=0.1)
        self.u8 = ConvUpBlock(l[1], l[0], dropout_rate=0.1)
        # final conv
        self.ce = nn.ConvTranspose2d(l[0], out_channel, 2, stride=2)

    def forward(self, x):
        # refer https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = c1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = c2 = self.resnet.layer1(x)
        x = c3 = self.resnet.layer2(x)
        x = c4 = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.ce(x)
        return x
