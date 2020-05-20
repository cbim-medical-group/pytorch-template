import torch
import torch.nn.functional as f


def exp_alignment_loss(offset, input, misc=None):
    b, c, h, w, d = input.shape
    offset_reformat = offset.reshape(b * d, 2)
    offset_reformat = offset_reformat[:, (1, 0)]
    # offset_reformat = offset_reformat/100
    # offset_reformat = offset_reformat
    # Normalize input mean=1, std=100. Recover to original input channel 0 df as *100

    input = input.permute(0, 4, 1, 2, 3)
    input_reformat = input[:, :, 0].reshape(b * d, 1, h, w)
    input_reformat_lax1 = input[:, :, 1].reshape(b * d, 1, h, w)
    input_reformat_lax2 = input[:, :, 2].reshape(b * d, 1, h, w)

    shift_input = shift_slice(offset_reformat, input_reformat)

    cal = (shift_input * input_reformat_lax1) + (shift_input * input_reformat_lax2)
    # cal_elem = torch.where(cal > 0)[0].shape
    loss = (cal.abs().sum() / (torch.nonzero(cal.data).size(0)) + 0.0001)
    # loss = cal.abs().sum() / cal.numel()

    # loss = cal.pow(2).sum() / (torch.nonzero(cal.data).size(0)+0.0001)

    return loss

def shift_slice(offset_reformat, input_reformat):
    """

    :param offset_reformat: b*d x 2, should from -1,1 indicates ratio of whole length.
    :param input: b*d x 3 x h x w
    :return:
    """
    # input_reformat = input_reformat * 100

    grid_x = torch.linspace(-1, 1, 120).unsqueeze(-1)
    grid_x = grid_x.repeat(1, 120)
    grid_x = grid_x.unsqueeze(-1)
    grid_y = torch.linspace(-1, 1, 120).unsqueeze(0)
    grid_y = grid_y.repeat(120, 1)
    grid_y = grid_y.unsqueeze(-1)

    grid = torch.cat((grid_y, grid_x), -1)
    grid.unsqueeze_(0)
    grid = grid.repeat(input_reformat.shape[0], 1, 1, 1)

    offset_reformat = offset_reformat.unsqueeze(1)
    offset_reformat = offset_reformat.unsqueeze(1)

    grid = grid.cuda() + offset_reformat

    shift_input = f.grid_sample(input_reformat, grid, padding_mode="border")

    return shift_input