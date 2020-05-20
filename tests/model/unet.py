import torch

from model.unet import Unet


class TestUnet:
    def test_unet(self):
        unet = Unet(in_channel=5, out_channel=4).cuda()

        input = torch.Tensor(10, 5, 128, 128).cuda()

        result = unet(input)
        assert result.shape == [10, 4, 128, 128]

input = torch.Tensor(5,3,20,20).cuda()
vae = VAE(512, 3, 3).cuda()
result = vae(input)