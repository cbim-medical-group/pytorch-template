import torch
import torch.nn as nn
import torch.nn.functional as F


class dilated_conv(nn.Module):
    """ same as original conv if dilation equals to 1 """

    def __init__(self, in_channel, out_channel, kernel_size=3, dropout_rate=0.0, activation=F.relu, dilation=1,
                 padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channel)
        self.activation = activation
        if dropout_rate > 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        else:
            self.drop = lambda x: x  # no-op

    def forward(self, x):
        # CAB: conv -> activation -> batch normal
        x = self.norm(self.activation(self.conv(x)))
        x = self.drop(x)
        return x


class ConvDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1, padding=1):
        super().__init__()
        self.conv1 = dilated_conv(in_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation,
                                  padding=padding)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation,
                                  padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=(1,2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x


class ConvUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, (1,2), stride=(1,2))
        self.conv1 = dilated_conv(in_channel // 2 + out_channel, out_channel, dropout_rate=dropout_rate,
                                  dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)

    def forward(self, x, x_skip):
        # print(f"up:x:{x.shape}, x_skip:{x_skip.shape}")
        x = self.up(x)
        H_diff = x.shape[2] - x_skip.shape[2]
        W_diff = x.shape[3] - x_skip.shape[3]
        # print(f"x:{x.shape}, x_skip:{x_skip.shape}, (0, W_diff, 0, H_diff): {(0, W_diff, 0, H_diff)}")
        x_skip = F.pad(x_skip, (0, W_diff, 0, H_diff), mode='reflect')
        x = torch.cat([x, x_skip], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetLax(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, padding=1):
        super().__init__()
        # down conv
        self.c1 = ConvDownBlock(in_channel, 64, padding=padding)
        self.c2 = ConvDownBlock(64, 128, padding=padding)
        self.c3 = ConvDownBlock(128, 256, padding=padding)
        self.c4 = ConvDownBlock(256, 512, padding=padding)
        self.cu = ConvDownBlock(512, 1024, padding=padding)
        # up conv
        self.u5 = ConvUpBlock(1024, 512)
        self.u6 = ConvUpBlock(512, 256)
        self.u7 = ConvUpBlock(256, 128)
        self.u8 = ConvUpBlock(128, 64)
        # final conv
        self.ce = nn.Conv2d(64, out_channel, kernel_size=1)

    def forward(self, x):
        x, c1 = self.c1(x)
        # print(f"x:{x.shape}, c1:{c1.shape}")
        x, c2 = self.c2(x)
        # print(f"x:{x.shape}, c2:{c2.shape}")
        x, c3 = self.c3(x)
        # print(f"x:{x.shape}, c3:{c3.shape}")
        x, c4 = self.c4(x)
        # print(f"x:{x.shape}, c4:{c4.shape}")
        _, x = self.cu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # print(f"x:{x.shape}")
        x = self.u5(x, c4)
        # print(f"x:{x.shape}, c4:{c4.shape}")
        x = self.u6(x, c3)
        # print(f"x:{x.shape}, c3:{c3.shape}")
        x = self.u7(x, c2)
        # print(f"x:{x.shape}, c2:{c2.shape}")
        x = self.u8(x, c1)
        # print(f"x:{x.shape}, c1:{c1.shape}")
        x = self.ce(x)
        # print(f"x:{x.shape}")
        return x
