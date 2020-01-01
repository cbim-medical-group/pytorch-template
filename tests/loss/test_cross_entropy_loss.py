import math
import torch
from torch.nn import functional as F

from loss.cross_entropy_loss import cross_entropy_loss


class TestCrossEntropyLoss:

    def test_cross_entropy_loss_2d(self):
        batch_size = 5
        channel = 3
        x = 10
        y = 10
        # output = np.zeros((batch_size, x, y))
        # target = np.copy(output)
        # result = cross_entropy_loss(torch.Tensor(output), torch.Tensor(target))
        # assert result == 0
        #
        # target = torch.rand((batch_size, x, y)).fill_(1)
        # input = torch.rand((batch_size, x, y)).fill_(1)
        # result = cross_entropy_loss(input, target)
        # assert result == 0

        input = torch.rand((batch_size, channel, x, y))
        target = torch.round(torch.rand((batch_size, x, y)) * (channel - 1)).long()
        loss = cross_entropy_loss(input, target)

        calc_loss = F.nll_loss(F.log_softmax(input, 1), target)

        assert math.isclose(float(loss), float(calc_loss), rel_tol=0.01)

    def test_cross_entropy_loss_3d(self):
        batch_size = 5
        channel = 5
        x = 5
        y = 5
        z = 5

        input = torch.rand((batch_size, channel, x, y, z))
        target = torch.round(torch.rand((batch_size, x, y, z)) * (channel - 1)).long()

        loss = cross_entropy_loss(input, target)

        calc_loss = F.nll_loss(F.log_softmax(input, 1), target)

        assert math.isclose(float(loss), float(calc_loss), rel_tol=0.01)

    def test_cross_entropy_loss_2d_weighted(self):
        batch_size = 50
        channel = 3
        x = 100
        y = 100
        # output = np.zeros((batch_size, x, y))
        # target = np.copy(output)
        # result = cross_entropy_loss(torch.Tensor(output), torch.Tensor(target))
        # assert result == 0
        #
        # target = torch.rand((batch_size, x, y)).fill_(1)
        # input = torch.rand((batch_size, x, y)).fill_(1)
        # result = cross_entropy_loss(input, target)
        # assert result == 0

        input = torch.rand((batch_size, channel, x, y))
        target = torch.round(torch.rand((batch_size, x, y)) * (channel - 1)).long()
        loss = cross_entropy_loss(input, target, [1, 1, 1])

        calc_loss = F.nll_loss(F.log_softmax(input, 1), target)

        assert math.isclose(float(loss), float(calc_loss), rel_tol=0.00001)

        loss2 = cross_entropy_loss(input, target, [0.001, 1000, 0.001])
        assert not math.isclose(float(loss2), float(calc_loss), rel_tol=0.00001)

