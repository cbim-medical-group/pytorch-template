import numpy as np

from data_loader.my_transforms.padding3d import Padding3d


class TestResize:
    def test_Padding(self, mocker):
        padding = Padding3d(60)
        image = np.random.rand(3, 30, 30, 30)
        mask = np.random.rand(30, 30, 30)

        result = padding({"image": image, "mask": mask})
        assert result['image'].shape == (3, 60, 60, 60)
        assert result['mask'].shape == (60, 60, 60)
        assert result['image'][:, 0, 0, 0].any() == 0
        assert result['image'][:, 1, 1, 1].any() == 0
        assert result['image'][:, 1, 0, 0].any() == 0
        assert result['image'][:, 0, 1, 1].any() == 0
        assert result['image'][:, 14, 15, 15].any() == False
        assert result['image'][:, 15, 15, 15].any() == True
        assert result['image'][:, 16, 16, 16].any() == True
        assert result['mask'][0, 0, 0].any() == 0
        assert result['mask'][0, 1, 1].any() == 0
        assert result['mask'][1, 0, 0].any() == 0
        assert result['mask'][1, 1, 1].any() == 0
        assert result['mask'][15, 15, 14].any() == False
        assert result['mask'][15, 15, 15].any() == True
        assert result['mask'][16, 16, 16].any() == True

    def test_Padding_1(self, mocker):
        padding = Padding3d(60)
        image = np.random.rand(3, 100, 30, 30)
        mask = np.random.rand(100, 30, 30)

        result = padding({"image": image, "mask": mask})
        assert result['image'].shape == (3, 100, 60, 60)
        assert result['mask'].shape == (100, 60, 60)
        assert result['image'][:, 0, 0, 0].any() == 0
        assert result['image'][:, 50, 1, 1].any() == 0
        assert result['image'][:, 50, 14, 14].any() == False
        assert result['image'][:, 50, 15, 15].any() == True
        assert result['mask'][0, 0, 0].any() == 0
        assert result['mask'][75, 1, 1].any() == 0
        assert result['mask'][75, 14, 14].any() == False
        assert result['mask'][75, 15, 15].any() == True
        assert result['mask'][75, 46, 46].any() == False
        assert result['mask'][75, 45, 45].any() == False
        assert result['mask'][75, 44, 44].any() == True
