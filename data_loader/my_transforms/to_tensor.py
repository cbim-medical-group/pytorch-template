import numpy as np
import torch
from skimage.transform import resize


class ToTensor:
    def __init__(self, training=True):
        """
        Convert numpy array to Torch.Tensor
        """
        self.training = training

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        image = torch.tensor(image).float()
        mask = torch.tensor(mask).long()

        return {'image': image, 'mask': mask}
