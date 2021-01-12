import numpy as np
import torch


def one_cls_jaccard(output, target, label_idx):
    """
    Calculate Jaccard metrics for one channel
    :param output:Output dimension: Batch x Channel x X x Y (x Z) float
    :param target:Target dimension: Batch x X x Y (x Z) int:[0, Channel]
    :param label_idx:
    :return:
    """
    eps = 0.0001

    pred = np.argmax(output, 1)
    pred_bool = (pred == label_idx)
    target_bool = (target == label_idx)
    intersection = pred_bool * target_bool


    return (intersection.sum((1,2,3))) / (pred_bool.sum((1,2,3)) + target_bool.sum((1,2,3)) - intersection.sum((1,2,3)) + eps)

    # with torch.no_grad():
    #     pred = torch.argmax(output, 1)
    #     pred_bool = (pred == label_idx)
    #     target_bool = (target == label_idx)
    #
    #     intersection = pred_bool * target_bool
    #     return (2 * int(intersection.sum())) / (int(pred_bool.sum()) + int(target_bool.sum()) + eps)


def jaccard(output, target, misc=None):
    """
    Calculate Jaccard index for all channels.
    :param output:Output dimension: Batch x Channel x X x Y (x Z) float
    :param target:Target dimension: Batch x X x Y (x Z) int:[0, Channel]
    :return:
    """
    channel_num = output.shape[1]
    # assert 1 < channel_num <= (target.max()+1)  # At least should have 1 foreground channel, and should have less than the target max.

    # We found numpy argmax is much faster than pytorch tensor
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    jaccard_idx = []
    for i in range(1, channel_num):
        jaccard = one_cls_jaccard(output, target, label_idx=i)
        jaccard_idx += jaccard.tolist()

    return np.average(np.array(jaccard_idx))
