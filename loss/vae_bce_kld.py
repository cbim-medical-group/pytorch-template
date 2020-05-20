import torch
from torch.nn import functional as F

def vae_bce_kld(output, target, mu, logvar):
    target = target.type_as(output)
    BCE = F.binary_cross_entropy_with_logits(output, target)
    # BCE = torch.mean((output - target) ** 2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    # return BCE + KLD * 0.1
    return BCE
