import math
import torch
from torch.nn import functional as F

from loss.cross_entropy_loss import cross_entropy_loss


import numpy as np

class TestDiceLossV2:

    def test_diceloss(self):
        batch_size = 5
        channel = 3
        x = 10
        y = 10
        output = np.zeros((batch_size, channel, x, y))
        output[1,1, 3:6,4:7] = 1
        output[2, 2, 3:6, 4:7] = 2

        target = np.zeros((batch_size, x, y))
        target[1, 3:6,4:7] = 1
        target[2, 3:6,4:7] = 2


        # result = cross_entropy_loss(torch.Tensor(output), torch.Tensor(target))
        # assert result == 0
        #
        # target = torch.rand((batch_size, x, y)).fill_(1)
        # input = torch.rand((batch_size, x, y)).fill_(1)
        # result = cross_entropy_loss(input, target)
        # assert result == 0



