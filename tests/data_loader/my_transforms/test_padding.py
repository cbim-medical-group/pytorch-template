import numpy as np
from data_loader.my_transforms.padding import Padding

from data_loader.my_transforms.random_crop import RandomCrop
from data_loader.my_transforms.resize import Resize


class TestResize:
    def test_Padding(self, mocker):
        padding = Padding(100)
        image = np.random.rand(3, 50, 50)
        mask = np.random.rand(50, 50)

        result = padding({"image": image, "mask": mask})
        assert result['image'].shape == (3, 100, 100)
        assert result['mask'].shape == (100, 100)
        assert result['image'][:, 0, 0].any() == 0
        assert result['image'][:, 1, 1].any() == 0
        assert result['image'][:, 1, 0].any() == 0
        assert result['image'][:, 0, 1].any() == 0
        assert result['mask'][0, 0].any() == 0
        assert result['mask'][0, 1].any() == 0
        assert result['mask'][1, 0].any() == 0
        assert result['mask'][1, 1].any() == 0

    def test_Padding_1(self, mocker):
        padding = Padding(100)
        image = np.random.rand(3, 150, 50)
        mask = np.random.rand(150, 50)

        result = padding({"image": image, "mask": mask})
        assert result['image'].shape == (3, 150, 100)
        assert result['mask'].shape == (150, 100)
        assert result['image'][:, 0, 0].any() == 0
        assert result['image'][:, 75, 1].any() == 0
        assert result['image'][:, 75, 24].any() == 0
        assert result['image'][:, 75, 25].any() == 1
        assert result['mask'][0, 0].any() == 0
        assert result['mask'][75, 1].any() == 0
        assert result['mask'][75, 24].any() == 0
        assert result['mask'][75, 25].any() == 1


