import numpy as np

from metric.accuracy import accuracy

import torch
from torch.autograd import Variable

class TestAccuracy:

    def test_accuracy(self):
        batch = 5
        class_num = 4  # class type: 0,1,2,3
        x = 10
        y = 10
        z = 10

        output = np.random.rand(batch, class_num, x, y, z)
        target = np.random.randint(0, 4, (batch, x, y, z))

        v_output = Variable(torch.Tensor(output))
        v_target = Variable(torch.Tensor(target))
        result1 = accuracy(v_output, v_target)

        target2 = np.argmax(output, 1)
        v_target2 = Variable(torch.Tensor(target2))
        result2 = accuracy(v_output, v_target2)

        assert result1 <= result2
        assert result2 == 1
