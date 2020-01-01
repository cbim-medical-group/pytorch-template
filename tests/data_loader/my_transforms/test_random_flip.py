import numpy as np

from data_loader.my_transforms.random_flip import RandomFlip


class TestRandomFlip:
    def test_random_flip(self, mocker):
        random_flip = RandomFlip(True, True)

        image = np.zeros((3, 200, 100))
        image[0, 0, 0] = 1
        mask = np.zeros((200, 100))
        mask[0, 0] = 1

        result = random_flip({"image": image, "mask": mask})
        assert result['image'].shape == (3, 200, 100)
        assert result['mask'].shape == (200, 100)
        assert result['image'][0, 0, 0] + result['image'][0, 0, -1] + result['image'][0, -1, 0] + \
               result['image'][0, -1, -1] == 1
        assert result['mask'][0, 0] + result['mask'][0, -1] + result['mask'][-1, 0] + result['mask'][-1, -1] == 1

    def test_random_flip_false(self, mocker):
        random_flip = RandomFlip(False, False)

        image = np.zeros((3, 200, 100))
        image[0, 0, 0] = 1
        mask = np.zeros((200, 100))
        mask[0, 0] = 1

        result = random_flip({"image": image, "mask": mask})
        assert result['image'].shape == (3, 200, 100)
        assert result['mask'].shape == (200, 100)
        assert result['image'][0, 0, 0] == 1
        assert result['mask'][0, 0] == 1
