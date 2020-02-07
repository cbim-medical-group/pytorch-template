import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class SvhnModelV2(BaseModel):
    def __init__(self, num_classes=10, in_channel=1):
        """
        Use the Mnist classification architecture from:
        https://github.com/cmasch/zalando-fashion-mnist/blob/master/Simple_Convolutional_Neural_Network_Fashion-MNIST.ipynb
        :param num_classes:
        :param in_channel:
        """
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel,64, (4,4))
        self.relu1 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d((2,2))
        self.conv1_drop = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(64, 64, (4,4))
        self.relu2 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d((2,2))
        self.conv2_drop = nn.Dropout2d(0.3)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1600, 256)
        self.relu3= nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc1_drop=nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1_drop(self.mp1(self.relu1(self.conv1(x))))
        # print(f"1:{x.shape}")

        x = self.conv2_drop(self.mp2(self.relu2(self.conv2(x))))
        # print(f"2:{x.shape}")

        x = self.flatten(x)
        # print(f"3:{x.shape}")

        x = self.fc1_drop(self.bn2(self.relu3(self.fc1(x))))
        # print(f"4:{x.shape}")

        x = self.bn3(self.relu4(self.fc2(x)))
        # print(f"5:{x.shape}")

        x = self.fc3(x)
        # print(f"6:{x.shape}")

        return F.log_softmax(x, dim=1)
