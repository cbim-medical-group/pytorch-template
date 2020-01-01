import numpy as np
import torch

from data_loader.my_transforms.normalize import Normalize
from data_loader.my_transforms.normalize_instance import NormalizeInstance


class TestNormalizeInstance:
    def test_normalize(self, mocker):
        image = np.random.rand(3, 50, 50) * 255
        mask = np.random.rand(50, 50)
        mean = image.mean()
        std = image.std()
        normalize = Normalize(mean=mean, std=std)

        image = torch.tensor(image)
        mask = torch.tensor(mask)

        result = normalize({"image": image, "mask": mask})
        assert result['mask'].sum() == mask.sum()
        assert float(result['image'].sum()) < 0.000001

    def test_normalize2(self, mocker):

        image = np.random.rand(10, 50, 50) * 255
        mask = np.random.rand(50, 50)
        mean = image.mean()
        std = image.std()
        normalize = Normalize(mean=mean, std=std)

        image = torch.tensor(image)
        mask = torch.tensor(mask)

        result = normalize({"image": image, "mask": mask})
        assert result['mask'].sum() == mask.sum()
        assert float(result['image'].sum()) < 0.000001



