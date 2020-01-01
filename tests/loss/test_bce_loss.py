import math
import numpy as np
import torch

from loss.bce_loss import bce_loss


class TestBceLoss:

    def test_bce_loss_2d(self):
        batch_size = 5
        x = 10
        y = 10
        output = np.zeros((batch_size, x, y))
        target = np.copy(output)
        result = bce_loss(torch.Tensor(output), torch.Tensor(target))
        assert result == 0

        target = torch.rand((batch_size, x, y)).fill_(1)
        input = torch.rand((batch_size, x, y)).fill_(1)
        result = bce_loss(input, target)
        assert result == 0

        target = torch.rand((batch_size, x, y))
        input = torch.rand((batch_size, x, y))
        loss = bce_loss(input, target)

        calc_loss = - target * torch.log(input) - (1 - target) * torch.log(1 - input)
        calc_loss = calc_loss.sum() / calc_loss.numel()
        assert math.isclose(float(loss), float(calc_loss), rel_tol=0.01)

    def test_bce_loss_3d(self):
        batch_size = 5
        x = 5
        y = 5
        z = 5

        target = torch.rand((batch_size, x, y, z))
        input = torch.rand((batch_size, x, y, z))
        loss = bce_loss(input, target)

        calc_loss = - target * torch.log(input) - (1 - target) * torch.log(1 - input)
        calc_loss = calc_loss.sum() / calc_loss.numel()
        assert math.isclose(float(loss), float(calc_loss), rel_tol=0.01)


