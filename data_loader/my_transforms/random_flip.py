import random


class RandomFlip:
    def __init__(self, horizontal=True, vertical=True, training=True):
        self.horizontal = horizontal
        self.vertical = vertical
        self.training = training

    def __call__(self, sample):
        """
        Randomly flip the numpy image horizontal or/and vertical
        Args:
            sample: {'image':..., 'mask':...}
            image size: [c, h, w]
            mask size: [h, w]
        Returns:
            Randomly flipped image.
        """
        image, mask = sample['image'], sample['mask']
        if not self.training:
            # If testing, will not flip.
            return {'image': image, 'mask': mask}

        if self.horizontal and random.random() < 0.5:
            image, mask = image[:, :, ::-1], mask[:, ::-1]
        if self.vertical and random.random() < 0.5:
            image, mask = image[:, ::-1, :], mask[:, :-1, :]

        return {'image': image, 'mask': mask}
