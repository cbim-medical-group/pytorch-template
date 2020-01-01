import numpy as np
import torch

from data_loader.my_transforms.normalize_instance import NormalizeInstance


class TestNormalizeInstance:
    def test_normalize_instance(self, mocker):
        normalize_instance = NormalizeInstance()

        image = np.random.rand(3, 50, 50) * 255
        mask = np.random.rand(50, 50)
        mean = image.mean()
        std = image.std()

        image = torch.tensor(image)
        mask = torch.tensor(mask)

        result = normalize_instance({"image": image, "mask": mask})
        assert result['mask'].sum() == mask.sum()
        assert float(result['image'].sum()) < 0.000001

    def test_normalize_instance2(self, mocker):
        normalize_instance = NormalizeInstance()

        image = np.random.rand(10, 50, 50) * 255
        mask = np.random.rand(50, 50)
        mean = image.mean()
        std = image.std()

        image = torch.tensor(image)
        mask = torch.tensor(mask)

        result = normalize_instance({"image": image, "mask": mask})
        assert result['mask'].sum() == mask.sum()
        assert float(result['image'].sum()) < 0.000001


