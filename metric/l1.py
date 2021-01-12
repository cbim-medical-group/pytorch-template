import torch


def l1(output, target, misc):
    with torch.no_grad():
        assert output.shape == target.shape
        correct = torch.sum(torch.abs(output - target))
    return correct / target.numel()
