import math
import numpy as np
import torch
from torch.autograd import Variable

from metric.dice import one_cls_dice, dice


class TestDice:

    def test_one_cls_dice(self):
        batch = 5
        class_num = 4  # class type: 0,1,2,3
        x = 10
        y = 10
        z = 10
        output = np.random.rand(batch, class_num, x, y, z)
        target = np.random.randint(0, 4, (batch, x, y, z), 'int')
        v_output = Variable(torch.Tensor(output))
        v_target = Variable(torch.Tensor(target))
        dice1 = one_cls_dice(v_output, v_target, 1)
        dice2 = one_cls_dice(v_output, v_target, 2)
        dice3 = one_cls_dice(v_output, v_target, 3)
        dice0 = one_cls_dice(v_output, v_target, 0)
        assert dice1 < 1
        assert dice2 < 1
        assert dice3 < 1
        assert dice0 < 1

    def test_one_cls_dice_one_cls(self):
        batch = 5
        class_num = 2  # class type: 0,1,2,3
        x = 10
        y = 10
        z = 10
        output = np.random.rand(batch, class_num, x, y, z)
        target = np.random.randint(0, 2, (batch, x, y, z), 'int')
        v_output = Variable(torch.Tensor(output))
        v_target = Variable(torch.Tensor(target))
        dice1 = one_cls_dice(v_output, v_target, 1)
        dice0 = one_cls_dice(v_output, v_target, 0)
        assert dice1 < 1
        assert dice0 < 1

    def test_one_cls_dice_identical(self):
        batch = 5
        class_num = 2  # class type: 0,1,2,3
        x = 10
        y = 10
        z = 10
        output = np.random.rand(batch, class_num, x, y, z)
        target = np.argmax(output, 1)

        v_output = Variable(torch.Tensor(output))
        v_target = Variable(torch.Tensor(target))
        dice1 = one_cls_dice(v_output, v_target, 1)
        dice2 = one_cls_dice(v_output, v_target, 2)
        dice3 = one_cls_dice(v_output, v_target, 3)
        dice0 = one_cls_dice(v_output, v_target, 0)

        assert math.isclose(dice1, 1, rel_tol=0.001)
        assert math.isclose(dice2, 0, rel_tol=0.001)
        assert math.isclose(dice3, 0, rel_tol=0.001)
        assert math.isclose(dice0, 1, rel_tol=0.001)

        class_num = 4
        output = np.random.rand(batch, class_num, x, y, z)
        target = np.argmax(output, 1)

        v_output = Variable(torch.Tensor(output))
        v_target = Variable(torch.Tensor(target))
        dice1 = one_cls_dice(v_output, v_target, 1)
        dice2 = one_cls_dice(v_output, v_target, 2)
        dice3 = one_cls_dice(v_output, v_target, 3)
        dice0 = one_cls_dice(v_output, v_target, 0)

        assert math.isclose(dice1, 1, rel_tol=0.001)
        assert math.isclose(dice2, 1, rel_tol=0.001)
        assert math.isclose(dice3, 1, rel_tol=0.001)
        assert math.isclose(dice0, 1, rel_tol=0.001)

    def test_dice_1(self):
        batch = 5
        class_num = 2  # class type: 0,1,2,3
        x = 10
        y = 10
        z = 10
        output = np.random.rand(batch, class_num, x, y, z)
        target = np.argmax(output, 1)

        v_output = Variable(torch.Tensor(output))
        v_target = Variable(torch.Tensor(target))
        dice_avg = dice(v_output, v_target)

        assert math.isclose(dice_avg, 1, rel_tol=0.001)

    def test_dice_2(self):
        batch = 5
        class_num = 10  # class type: 0,1,2,3
        x = 10
        y = 10
        z = 10
        output = np.random.rand(batch, class_num, x, y, z)
        target = np.argmax(output, 1)

        v_output = Variable(torch.Tensor(output))
        v_target = Variable(torch.Tensor(target))
        dice_avg = dice(v_output, v_target)

        assert math.isclose(dice_avg, 1, rel_tol=0.001)

    def test_dice_3(self):
        batch = 5
        class_num = 4  # class type: 0,1,2,3
        x = 10
        y = 10
        z = 10
        output = np.random.rand(batch, class_num, x, y, z)
        target = np.random.randint(0, 4, (batch, x, y, z), 'int')
        v_output = Variable(torch.Tensor(output))
        v_target = Variable(torch.Tensor(target))
        dice_avg = dice(v_output, v_target)

        assert dice_avg <=1
