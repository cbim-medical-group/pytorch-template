import numpy as np

from data_loader.my_transforms.random_crop import RandomCrop
from data_loader.my_transforms.random_crop3d import RandomCrop3d


class TestRandomCrop:
    def test_random_crop(self, mocker):
        mocker.patch.object(np.random, "randint", return_value=0)
        random_crop = RandomCrop3d(100)

        image = np.zeros((3, 200, 100, 100))
        image[0, 0, 0, 0] = 1
        mask = np.zeros((200, 100, 100))
        mask[0, 0, 0] = 1

        result = random_crop({"image": image, "mask": mask})
        assert result['image'].shape == (3, 100, 100, 100)
        assert result['mask'].shape == (100, 100, 100)
        assert result['image'][0, 0, 0, 0] == 1
        assert result['mask'][0, 0, 0] == 1

    def test_random_crop_2(self, mocker):
        mocker.patch.object(np.random, "randint", return_value=100)
        random_crop = RandomCrop3d(100)

        image = np.zeros((3, 200, 400, 400))
        image[2, 100, 100, 100] = 1
        mask = np.zeros((200, 400, 400))
        mask[100, 100, 100] = 1

        result = random_crop({"image": image, "mask": mask})
        assert result['image'].shape == (3, 100, 100, 100)
        assert result['mask'].shape == (100, 100, 100)
        assert result['image'][2, 0, 0, 0] == 1
        assert result['mask'][0, 0, 0] == 1

    def test_random_crop_3(self, mocker):
        random_crop = RandomCrop3d(100)

        image = np.zeros((3, 100, 100, 100))
        image[2, 0, 0, 0] = 1
        mask = np.zeros((100, 100, 100))
        mask[0, 0, 0] = 1

        result = random_crop({"image": image, "mask": mask})
        assert result['image'].shape == (3, 100, 100, 100)
        assert result['mask'].shape == (100, 100, 100)
        assert result['image'][2, 0, 0, 0] == 1
        assert result['mask'][0, 0, 0] == 1

    def test_random_crop_4(self, mocker):
        random_crop = RandomCrop3d(30)

        image = np.zeros((3, 100, 100, 100))
        image[2, 0, 0, 0] = 1
        mask = np.zeros((100, 100, 100))
        mask[0, 0, 0] = 1

        result = random_crop({"image": image, "mask": mask})
        assert result['image'].shape == (3, 30, 30, 30)
        assert result['mask'].shape == (30, 30, 30)

    def test_random_crop_5(self, mocker):
        random_crop = RandomCrop3d([30, 60, 60])

        image = np.zeros((3, 100, 100, 100))
        image[2, 0, 0, 0] = 1
        mask = np.zeros((100, 100, 100))
        mask[0, 0, 0] = 1

        result = random_crop({"image": image, "mask": mask})
        assert result['image'].shape == (3, 30, 60, 60)
        assert result['mask'].shape == (30, 60, 60)

    def test_random_crop_6(self, mocker):
        random_crop = RandomCrop3d([30,100, 100])

        image = np.zeros((3, 100, 100, 100))
        image[2, 0, 0, 0] = 1
        mask = np.zeros((100, 100, 100))
        mask[0, 0, 0] = 1

        result = random_crop({"image": image, "mask": mask})
        assert result['image'].shape == (3, 30, 100, 100)
        assert result['mask'].shape == (30, 100, 100)

    def test_random_crop_7(self, mocker):
        random_crop = RandomCrop3d((30,100, 100))

        image = np.zeros((1, 100, 100, 100))
        mask = np.zeros((100, 100, 100))

        result = random_crop({"image": image, "mask": mask})
        assert result['image'].shape == (1, 30, 100, 100)
        assert result['mask'].shape == (30, 100, 100)


