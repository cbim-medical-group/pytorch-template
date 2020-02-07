import torch

from model.mnist_model_v2 import MnistModelV2
from model.svhn_model_v2 import SvhnModelV2


class TestMnistt:
    def test_mnist(self):
        mnist = MnistModelV2(num_classes=10, in_channel=1).cuda()

        input = torch.Tensor(10, 1, 28, 28).cuda()

        result = mnist(input)
        assert list(result.shape) == [10, 10]
    def test_svhn(self):
        mnist = SvhnModelV2(num_classes=10, in_channel=3).cuda()

        input = torch.Tensor(20, 3, 32, 32).cuda()

        result = mnist(input)
        assert list(result.shape) == [20, 10]
