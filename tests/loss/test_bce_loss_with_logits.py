import math
import torch
import numpy as np
from loss.bce_loss_with_logits import bce_loss_with_logits
from torch.nn import functional as F

class TestBceLossWithLogits:

    def test_bce_loss_with_logits_2d(self):
        batch_size = 5
        x = 10
        y = 10

        output = torch.Tensor(np.zeros((batch_size, x, y))).fill_(-100)
        target = torch.Tensor(np.copy(output)).fill_(0)
        result = bce_loss_with_logits(output, target)
        assert result == 0
        #
        output = torch.rand((batch_size, x, y)).fill_(100)
        target = torch.rand((batch_size, x, y)).fill_(1)
        result = bce_loss_with_logits(output, target)
        assert result == 0


        target = torch.rand((batch_size, x, y))
        input = torch.rand((batch_size, x, y))
        loss = bce_loss_with_logits(input, target)

        calc_loss = - target * torch.log(F.sigmoid(input)) - (1-target) * torch.log(1-F.sigmoid(input))
        calc_loss = calc_loss.sum()/calc_loss.numel()
        assert math.isclose(float(loss), float(calc_loss), rel_tol=0.01)

    def test_bce_loss_with_logits_3d(self):
        batch_size = 5
        x = 5
        y = 5
        z = 5


        target = torch.rand((batch_size, x, y, z))
        input = torch.rand((batch_size, x, y, z))
        loss = bce_loss_with_logits(input, target)

        calc_loss = - target * torch.log(F.sigmoid(input)) - (1-target) * torch.log(1-F.sigmoid(input))
        calc_loss = calc_loss.sum()/calc_loss.numel()
        assert math.isclose(float(loss), float(calc_loss), rel_tol=0.01)




