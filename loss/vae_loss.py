import torch.nn.functional as f


def vae_loss(input, target, misc):
    return f.mse_loss(input, target)+f.smooth_l1_loss(input, target)
    # return f.mse_loss(input, target)
