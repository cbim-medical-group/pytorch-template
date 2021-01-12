'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M2', 128, 'M2', 256, 256, 'M2', 512, 512, 'M', 512, 512, 'M'],
    'VGG11D': [64, 'M2', 128, 'M2', 256, 256, 'M2', 512, 512, 'M', 512, 512, 'M', 512, 512, "M"],
    'VGG13': [64, 64, 'M2', 128, 128, 'M2', 256, 256, 'M2', 512, 512, 'M', 512, 512, 'M'],
    'VGG13D': [64, 64, 'M2', 128, 128, 'M2', 256, 256, 'M2', 512, 512, 'M', 512, 512, 'M', 512, 512, "M"],
    'VGG16': [64, 64, 'M2', 128, 128, 'M2', 256, 256, 256, 'M2', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16D': [64, 64, 'M2', 128, 128, 'M2', 256, 256, 256, 'M2', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M2', 128, 128, 'M2', 256, 256, 256, 256, 'M2', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19D': [64, 64, 'M2', 128, 128, 'M2', 256, 256, 256, 256, 'M2', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512*3*3+num_classes, num_classes)

    def forward(self, x, pred_y):
        out = self.features(x)
        # print(f"after features:{out.shape}")
        out = out.view(out.size(0), -1)
        # print(f"after view:{out.shape}")

        out = torch.cat((out, pred_y),1)
        # print(f"after cat:{out.shape}")
        out = self.classifier(out)
        # print(f"after cls:{out.shape}")
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 4
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))]
            elif x == "M2":
                layers += [nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))]
            # if x == 'M':
            #     layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv3d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm3d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool3d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11D', num_classes=32)
    x = torch.randn(2,4,200,200,13)

    zero_y = torch.zeros((2, 32))

    y = net(x, zero_y)
    print(y.size())

# test()