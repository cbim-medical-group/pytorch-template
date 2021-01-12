import numpy as np


class Squeeze:
    def __init__(self, axis=None, squeeze_image=True, squeeze_label=False):
        """
        Squeeze any image if axis dimension is not 1. For instance, from (1, X, Y, Z) to (X, Y, Z).

        """
        self.axis = axis
        self.squeeze_image = squeeze_image
        self.squeeze_label = squeeze_label

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        if self.squeeze_image:
            image = np.squeeze(image, self.axis)
        if self.squeeze_label:
            mask = np.squeeze(mask, self.axis)

        return {'image': image, 'mask': mask, "misc": misc}
