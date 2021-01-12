import torch


def exp_alignment_mcnet(data, mask, misc):
    # input.detach()
    # ratio_mask = mask / (119 / 2)
    with torch.no_grad():
        b, c, h, w, d = data.shape
        input = data.permute(0, 4, 1, 2, 3)  # B x 13 x 4 x 200 x 200
        input_reformat_sax = input[:, :, 0].reshape(b * d, 1, h, w) # B*13 x 1 x 200 x 200
        input_reformat_lax1 = input[:, :, 1].reshape(b * d, 1, h, w)
        input_reformat_lax2 = input[:, :, 2].reshape(b * d, 1, h, w)
        input_reformat_lax3 = input[:, :, 3].reshape(b * d, 1, h, w)

        cal = (input_reformat_sax * input_reformat_lax1) + (input_reformat_sax * input_reformat_lax2) + (input_reformat_sax * input_reformat_lax3)

        return cal.abs().sum() / (torch.nonzero(cal.data).size(0)+ 1e-5)
