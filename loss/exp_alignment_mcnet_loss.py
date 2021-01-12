import torch
import torch.nn.functional as f


def shift_data_by_offset(offset, input):
    # Offset shape: B x (13+3)*2 = B x 32;
    # Input shape: B x 4 x H x W x D = B x 4 x 200 x 200 x 13
    b, c, h, w, d = input.shape

    sax_offset_reformat = offset[:, :26].reshape(b * 13, 2)
    lax1_offset_reformat = offset[:, 26:28].reshape(b, 2)
    lax1_offset_reformat = lax1_offset_reformat.repeat_interleave(13,0)

    lax2_offset_reformat = offset[:, 28:30].reshape(b, 2)
    lax2_offset_reformat = lax2_offset_reformat.repeat_interleave(13,0)

    lax3_offset_reformat = offset[:, 30:].reshape(b, 2)
    lax3_offset_reformat = lax3_offset_reformat.repeat_interleave(13,0)

    # sax_offset_reformat = sax_offset_reformat[:, (1, 0)]
    # lax1_offset_reformat = lax1_offset_reformat[:, (1, 0)]

    # offset_reformat = offset_reformat/100
    # offset_reformat = offset_reformat
    # Normalize input mean=1, std=100. Recover to original input channel 0 df as *100

    input = input.permute(0, 4, 1, 2, 3)  # B x 13 x 4 x 200 x 200
    input_reformat = input[:, :, 0].reshape(b * d, 1, h, w) # B*13 x 1 x 200 x 200
    input_reformat_lax1 = input[:, :, 1].reshape(b * d, 1, h, w)
    input_reformat_lax2 = input[:, :, 2].reshape(b * d, 1, h, w)
    input_reformat_lax3 = input[:, :, 3].reshape(b * d, 1, h, w)

    shift_input = shift_slice(sax_offset_reformat, input_reformat)
    shift_data_sax = shift_input.reshape(b, d, 1, h, w)
    shift_data_sax = shift_data_sax.permute(0, 2, 3, 4, 1)

    shift_input_lax1 = shift_slice(lax1_offset_reformat, input_reformat_lax1)
    shift_data_lax1 = shift_input_lax1.reshape(b, d, 1, h, w)
    shift_data_lax1 = shift_data_lax1.permute(0, 2, 3, 4, 1)

    shift_input_lax2 = shift_slice(lax2_offset_reformat, input_reformat_lax2)
    shift_data_lax2 = shift_input_lax2.reshape(b, d, 1, h, w)
    shift_data_lax2 = shift_data_lax2.permute(0, 2, 3, 4, 1)

    shift_input_lax3 = shift_slice(lax3_offset_reformat, input_reformat_lax3)
    shift_data_lax3 = shift_input_lax3.reshape(b, d, 1, h, w)
    shift_data_lax3 = shift_data_lax3.permute(0, 2, 3, 4, 1)

    shift_data = torch.cat((shift_data_sax, shift_data_lax1, shift_data_lax2, shift_data_lax3),1)


    return shift_data, shift_input, shift_input_lax1, shift_input_lax2, shift_input_lax3


def shift_slice(offset_reformat, input_reformat):
    """

    :param offset_reformat: b*d x 2, should from -1,1 indicates ratio of whole length.
    :param input: b*d x 3 x h x w
    :return:
    """
    # input_reformat = input_reformat * 100
    wh_size = 200

    grid_x = torch.linspace(-1, 1, wh_size).unsqueeze(-1)
    grid_x = grid_x.repeat(1, wh_size)
    grid_x = grid_x.unsqueeze(-1)
    grid_y = torch.linspace(-1, 1, wh_size).unsqueeze(0)
    grid_y = grid_y.repeat(wh_size, 1)
    grid_y = grid_y.unsqueeze(-1)

    grid = torch.cat((grid_y, grid_x), -1)
    grid.unsqueeze_(0)
    grid = grid.repeat(input_reformat.shape[0], 1, 1, 1)

    offset_reformat = offset_reformat.unsqueeze(1)
    offset_reformat = offset_reformat.unsqueeze(1)

    grid = grid.cuda() + offset_reformat

    shift_input = f.grid_sample(input_reformat, grid, padding_mode="border")

    return shift_input



def exp_alignment_mcnet_loss(offset, input, misc=None):
    _, shift_input, shift_input_lax1, shift_input_lax2, shift_input_lax3 = shift_data_by_offset(offset, input)

    cal = (shift_input * shift_input_lax1) + (shift_input * shift_input_lax2) + (shift_input * shift_input_lax3)
    # cal_elem = torch.where(cal > 0)[0].shape
    loss = (cal.abs().sum() / (torch.nonzero(cal.data).size(0)) + 0.0001)
    # loss = cal.abs().sum() / cal.numel()

    # loss = cal.pow(2).sum() / (torch.nonzero(cal.data).size(0)+0.0001)

    return loss


