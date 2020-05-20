import torch

from model.unet import Unet
from model.unet3D import Unet3D


class TestUnet3D:
    def test_unet3d(self):
        unet = Unet3D(in_channel=5, out_channel=4).cuda()

        input = torch.Tensor(6, 5, 64, 64, 64).cuda()

        result = unet(input)
        assert list(result.shape) == [6, 4, 64, 64, 64]
