from loss.lib.losses_pytorch.dice_loss import GDiceLoss, GDiceLossV2
from torch.nn.functional import softmax

def g_dice_loss(output, target, misc, weight=[1]):
    # diceloss = GDiceLoss()
    # return diceloss.forward(output, target)
    # def dice_loss(input, target):
    smooth = 0.001
    loss = 0.
    target = target.unsqueeze(1)
    for c in range(len(weight)):
        iflat = output[:, c].view(output.shape[0], -1)
        tflat = target[:, c].view(target.shape[0], -1)
        intersection = (iflat * tflat).sum(1)

        w = weight[c]
        oneloss = w * (1 - ((2. * intersection + smooth) /
                          (iflat.sum(1) + tflat.sum(1) + smooth)))
        loss +=oneloss.mean()
    return loss