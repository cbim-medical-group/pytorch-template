import numpy as np
import torch


def one_cls_dice(output, target, label_idx):
    """
    Calculate Dice metrics for one channel
    :param output:Output dimension: Batch x X x Y (x Z) float
    :param target:Target dimension: Batch x X x Y (x Z) int:[0, Channel]
    :param label_idx:
    :return:
    """
    eps = 0.0001
    with torch.no_grad():
        pred_bool = (output > 0.5)
        target_bool = (target == 1)

        intersection = pred_bool * target_bool
        sum_range = tuple(range(1,len(intersection.shape)))
        return 2* intersection.sum(sum_range) / (pred_bool.sum(sum_range) + target_bool.sum(sum_range) + eps)



def exp_dt_dice_binary(output, target, misc=None):
    """
    Calculate dice for all channels.
    :param output:Output dimension: Batch x Channel x X x Y (x Z) float
    :param target:Target dimension: Batch x X x Y (x Z) int:[0, Channel]
    :return:
    """
    # channel_num = output.shape[1]
    # assert 1 < channel_num <= (target.max()+1)  # At least should have 1 foreground channel, and should have less than the target max.
    with torch.no_grad():
        output_dt = output[:,0]
        target_dt = target[:,0]

        bi_output = torch.zeros_like(output_dt)
        bi_target = torch.zeros_like(target_dt)
        bi_output[output_dt < 0.5] = 1
        bi_target[target_dt < 0.5] = 1
        dices = []
        # for i in range(1, channel_num):
        dice = one_cls_dice(bi_output, bi_target, label_idx=1)
        dices +=dice.tolist()

        return np.average(np.array(dices))
