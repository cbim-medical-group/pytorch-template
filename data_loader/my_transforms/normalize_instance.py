import torchvision.transforms.functional as F


class NormalizeInstance(object):
    """Normalize a tensor volume with mean and standard deviation estimated
    from the sample itself.
    :param mean: mean value.
    :param std: standard deviation value.
    """
    def __init__(self, training=True, norm_label=False):
        self.training = training
        self.norm_label = norm_label

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        mean, std = image.mean(), image.std()

        if mean != 0 or std != 0:
            image = F.normalize(image, [mean for _ in range(0, image.shape[0])],
                                 [std for _ in range(0, image.shape[0])])
            if self.norm_label:
                mask = F.normalize(mask, [mean for _ in range(0, mask.shape[0])],
                                 [std for _ in range(0, mask.shape[0])])

        return {"image":image, "mask":mask, "misc": misc}
