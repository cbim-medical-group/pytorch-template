from loss.lib.losses_pytorch.dice_loss import GDiceLoss, GDiceLossV2, FocalTversky_loss


def g_focaltversky_loss(output, target, misc, weight=None):
    diceloss = FocalTversky_loss()
    return diceloss.forward(output, target)