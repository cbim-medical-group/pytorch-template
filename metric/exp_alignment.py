import torch


def exp_alignment(offset, mask, misc):
    # input.detach()
    # ratio_mask = mask / (119 / 2)

    cal = (offset - mask).abs()

    return cal.sum() / torch.nonzero(cal.data).size(0)
