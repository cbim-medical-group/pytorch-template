import numpy as np

from data_loader.my_transforms.scale import Scale


class TestScale:
    def test_Scale_nearest(self, mocker):
        scale = Scale(100, interp="nearest")

        image = np.zeros((3, 50, 50))
        image[2, 0, 0] = 1
        mask = np.zeros((50, 50))
        mask[0, 0] = 1

        result = scale({"image": image, "mask": mask})
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

    def test_Scale_all_channels(self, mocker):
        scale = Scale(100, interp="nearest")

        image = np.zeros((3, 50, 50))
        image[:, 0, 0] = 1
        mask = np.zeros((50, 50))
        mask[0, 0] = 1

        result = scale({"image": image, "mask": mask})
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

    def test_Scale_differentHW(self, mocker):
        scale = Scale(100, interp="nearest")

        image = np.zeros((3, 150, 50))
        image[:, 0, 0] = 1
        mask = np.zeros((150, 50))
        mask[0, 0] = 1

        result = scale({"image": image, "mask": mask})
        assert result['image'].shape == (3, 300, 100)
        assert result['mask'].shape == (300, 100)
        assert (result['image'][:, 0, 0]).any() == 1
        assert (result['image'][:, 0, 1]).any() == 1
        assert (result['image'][:, 0, 2]).any() == 0
        assert (result['image'][:, 0, 3]).any() == 0
        assert (result['image'][:, 1, 0]).any() == 1
        assert (result['image'][:, 2, 0]).any() == 0
        assert (result['image'][:, 3, 0]).any() == 0
        assert result['mask'][0, 0].any() == 1
        assert result['mask'][0, 1].any() == 1
        assert result['mask'][1, 0].any() == 1
        assert result['mask'][1, 1].any() == 1

    def test_Scale_differentHW2(self, mocker):
        scale = Scale(100, interp="nearest")

        image = np.zeros((3, 50, 150))
        image[:, 0, 0] = 1
        mask = np.zeros((50, 150))
        mask[0, 0] = 1

        result = scale({"image": image, "mask": mask})
        assert result['image'].shape == (3, 100, 300)
        assert result['mask'].shape == (100, 300)
        assert (result['image'][:, 0, 0]).any() == 1
        assert (result['image'][:, 0, 1]).any() == 1
        assert (result['image'][:, 0, 2]).any() == 0
        assert (result['image'][:, 0, 3]).any() == 0
        assert (result['image'][:, 1, 0]).any() == 1
        assert (result['image'][:, 2, 0]).any() == 0
        assert (result['image'][:, 3, 0]).any() == 0
        assert result['mask'][0, 0].any() == 1
        assert result['mask'][0, 1].any() == 1
        assert result['mask'][1, 0].any() == 1
        assert result['mask'][1, 1].any() == 1
