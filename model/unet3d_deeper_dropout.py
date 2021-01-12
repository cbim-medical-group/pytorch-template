import torch
import torch.nn as nn
import torch.nn.functional as F


class dilated_conv(nn.Module):
    """ same as original conv if dilation equals to 1 """

    def __init__(self, in_channel, out_channel, kernel_size=3, dropout_rate=0.0, activation=F.relu, dilation=1,
                 padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm3d(out_channel)
        self.activation = activation
        if dropout_rate > 0:
            self.drop = nn.Dropout3d(p=dropout_rate)
        else:
            self.drop = lambda x: x  # no-op

    def forward(self, x):
        # CAB: conv -> activation -> batch normal
        x = self.norm(self.activation(self.conv(x)))
        x = self.drop(x)
        return x


class ConvDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1, padding=1, kernel_size=(2,2,2)):
        super().__init__()
        self.conv1 = dilated_conv(in_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation,
                                  padding=padding)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation,
                                  padding=padding)
        self.pool = nn.MaxPool3d(kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x


class ConvUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1, kernel_size=(2,2,2), stride=(2,2,2)):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channel, in_channel // 2, kernel_size, stride=stride)
        self.conv1 = dilated_conv(in_channel // 2 + out_channel, out_channel, dropout_rate=dropout_rate,
                                  dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)

    def forward(self, x, x_skip):
        # print(f"orig:{x.shape},skip:{x_skip.shape}")

        # H_diff = x.shape[2] - x_skip.shape[2]
        # W_diff = x.shape[3] - x_skip.shape[3]
        # D_diff = x.shape[4] - x_skip.shape[4]
        # # print(f"up: {x.shape}, skip{x_skip.shape}")
        # x_skip = F.pad(x_skip, (0, D_diff, 0, W_diff, 0, H_diff), mode='replicate')
        # print(f"pad: {x.shape}, skip{x_skip.shape}")
        x = self.up(x, output_size=x_skip.shape)
        x = torch.cat([x, x_skip], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Unet3dDeeperDropout(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, channel_start=64, padding=1, dropout_rate=0.0):
        super().__init__()
        # down conv
        self.c1 = ConvDownBlock(in_channel, channel_start, padding=padding)
        self.c2 = ConvDownBlock(channel_start, channel_start * 2, padding=padding)
        self.c3 = ConvDownBlock(channel_start * 2, channel_start * 4, padding=padding)
        self.c4 = ConvDownBlock(channel_start * 4, channel_start * 8, padding=padding,kernel_size=(2,2,1))
        self.c5 = ConvDownBlock(channel_start * 8, channel_start * 16, padding=padding,kernel_size=(2,2,1), dropout_rate=dropout_rate)
        self.cu = ConvDownBlock(channel_start * 16, channel_start * 32, padding=padding,kernel_size=(2,2,1), dropout_rate=dropout_rate)
        # up conv
        self.u4 = ConvUpBlock(channel_start * 32, channel_start * 16,kernel_size=(2,2,1), stride=(2,2,1))
        self.u5 = ConvUpBlock(channel_start * 16, channel_start * 8,kernel_size=(2,2,1), stride=(2,2,1))
        self.u6 = ConvUpBlock(channel_start * 8, channel_start * 4)
        self.u7 = ConvUpBlock(channel_start * 4, channel_start * 2)
        self.u8 = ConvUpBlock(channel_start * 2, channel_start)
        # final conv
        self.ce = nn.Conv3d(channel_start, out_channel, kernel_size=1)

    def forward(self, x):
        x, c1 = self.c1(x)
        # print(f"x:{x.shape}, c1:{c1.shape}")
        x, c2 = self.c2(x)
        # print(f"x:{x.shape}, c2:{c2.shape}")
        x, c3 = self.c3(x)
        # print(f"x:{x.shape}, c3:{c3.shape}")
        x, c4 = self.c4(x)
        # print(f"x:{x.shape}, c4:{c4.shape}")
        x, c5 = self.c5(x)
        # print(f"x:{x.shape}, c4:{c5.shape}")
        _, x = self.cu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # print(f"up...x:{x.shape}, c4:{c5.shape}")
        x = self.u4(x, c5)
        # print(f"x:{x.shape}, c4:{c4.shape}")
        x = self.u5(x, c4)
        # print(f"x:{x.shape}, c3:{c3.shape}")
        x = self.u6(x, c3)
        # print(f"x:{x.shape}, c2:{c2.shape}")
        x = self.u7(x, c2)
        # print(f"x:{x.shape}, c1:{c1.shape}")
        x = self.u8(x, c1)
        # print(f"x:{x.shape}")
        x = self.ce(x)
        return x

