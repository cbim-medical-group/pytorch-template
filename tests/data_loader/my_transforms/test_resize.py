import numpy as np

from data_loader.my_transforms.random_crop import RandomCrop
from data_loader.my_transforms.resize import Resize


class TestResize:
    def test_Resize_nearest(self, mocker):
        resize = Resize(100, interp="nearest")

        image = np.zeros((3, 50, 50))
        image[2, 0, 0] = 1
        mask = np.zeros((50, 50))
        mask[0, 0] = 1

        result = resize({"image": image, "mask": mask})
        assert result['image'].shape == (3, 100, 100)
        assert result['mask'].shape == (100, 100)
        assert result['image'][2, 0, 0] == 1
        assert result['image'][2, 1, 1] == 1
        assert result['image'][2, 1, 0] == 1
        assert result['image'][2, 0, 1] == 1
        assert result['mask'][0, 0] == 1
        assert result['mask'][0, 1] == 1
        assert result['mask'][1, 0] == 1
        assert result['mask'][1, 1] == 1

    def test_Resize_all_channels(self, mocker):
        resize = Resize(100, interp="nearest")

        image = np.zeros((3, 50, 50))
        image[:, 0, 0] = 1
        mask = np.zeros((50, 50))
        mask[0, 0] = 1

        result = resize({"image": image, "mask": mask})
        assert result['image'].shape == (3, 100, 100)
        assert result['mask'].shape == (100, 100)
        assert (result['image'][:, 0, 0]).any() == 1
        assert (result['image'][:, 1, 1]).any() == 1
        assert (result['image'][:, 1, 0]).any() == 1
        assert (result['image'][:, 0, 1]).any() == 1
        assert result['mask'][0, 0].any() == 1
        assert result['mask'][0, 1].any() == 1
        assert result['mask'][1, 0].any() == 1
        assert result['mask'][1, 1].any() == 1


