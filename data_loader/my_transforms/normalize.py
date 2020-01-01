import torchvision.transforms.functional as F


class Normalize(object):
    """Normalize a tensor volume given the mean and standard deviation.
    :param mean: mean value.
    :param std: standard deviation value.
    """

    def __init__(self, mean, std, training=True):
        self.mean = mean
        self.std = std
        self.training = training

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if self.mean != 0 or self.std != 0:
            image = F.normalize(image, [self.mean for _ in range(0, image.shape[0])],
                                [self.std for _ in range(0, image.shape[0])])
        return {"image": image, "mask": mask}
