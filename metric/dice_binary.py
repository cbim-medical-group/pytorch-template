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
        return (2 * int(intersection.sum())) / (int(pred_bool.sum()) + int(target_bool.sum()) + eps)


def dice_binary(output, target, misc=None):
    """
    Calculate dice for all channels.
    :param output:Output dimension: Batch x Channel x X x Y (x Z) float
    :param target:Target dimension: Batch x X x Y (x Z) int:[0, Channel]
    :return:
    """
    # channel_num = output.shape[1]
    # assert 1 < channel_num <= (target.max()+1)  # At least should have 1 foreground channel, and should have less than the target max.
    dices = []
    # for i in range(1, channel_num):
    dice = one_cls_dice(output, target, label_idx=1)
    dices.append(dice)

    return np.average(np.array(dices))