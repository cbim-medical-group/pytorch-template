import numpy as np

from data_loader.general_data_loader import GeneralDataset


class TestGeneralDataLoader:
    def test_split_volume(self, mocker):
        input = np.arange(8 * 8 * 8).reshape(8, 8, 8)
        result_input, result_label, orig_split_pos, ext_shape = GeneralDataset.split_volume(input, (4, 4, 4),
                                                                                            (4, 4, 4))

        assert result_input.shape == (2 * 2 * 2, 4, 4, 4)
        assert orig_split_pos == (2, 2, 2)
        assert ext_shape == (8, 8, 8)
        assert result_input[0, 0, 0, 0] == 0
        assert result_input[0, 0, 1, 1] == 9
        assert result_input[1, 0, 0, 0] == 4

    def test_split_volume2(self, mocker):
        input = np.arange(8 * 8 * 8).reshape(8, 8, 8)
        result_input, result_label, orig_split_pos, ext_shape = GeneralDataset.split_volume(input, (4, 4, 4),
                                                                                            (2, 2, 2))

        assert result_input.shape == (4 * 4 * 4, 4, 4, 4)
        assert orig_split_pos == (4, 4, 4)
        assert ext_shape == (10, 10, 10)
        assert result_input[0, 0, 0, 0] == 0
        assert result_input[0, 0, 1, 1] == 9
        assert result_input[1, 0, 0, 0] == 2

    def test_split_volume3(self, mocker):
        input = np.arange(8 * 8 * 8).reshape(8, 8, 8)
        result_input, result_label, orig_split_pos, ext_shape = GeneralDataset.split_volume(input, (4, 4, 4),
                                                                                            (3, 3, 3))

        assert result_input.shape == (3 * 3 * 3, 4, 4, 4)
        assert orig_split_pos == (3, 3, 3)
        assert ext_shape == (10, 10, 10)
        assert result_input[0, 0, 0, 0] == 0
        assert result_input[0, 0, 1, 1] == 9
        assert result_input[1, 0, 0, 0] == 3

    def test_combine_volume(self, mocker):
        input = np.arange(8 * 8 * 8).reshape(8, 8, 8)
        result_input, result_label, orig_split_pos, ext_shape = GeneralDataset.split_volume(input, (4, 4, 4),
                                                                                            (4, 4, 4))

        result = GeneralDataset.combine_volume(result_input, result_input, input.shape, ext_shape, orig_split_pos,
                                               (4, 4, 4))

        assert result.shape == (8, 8, 8)
        assert input.sum() == result.sum()

    def test_combine_volume2(self, mocker):
        input = np.arange(8 * 8 * 8).reshape(8, 8, 8)
        result_input, result_label, orig_split_pos, ext_shape = GeneralDataset.split_volume(input, (4, 4, 4),
                                                                                            (2, 2, 2))

        result = GeneralDataset.combine_volume(result_input, result_input, input.shape, ext_shape, orig_split_pos,
                                               (2, 2, 2))

        assert result.shape == (8, 8, 8)
        assert input.sum() < result.sum()


    def test_combine_volume3(self, mocker):
        input = np.arange(8 * 8 * 8).reshape(8, 8, 8)
        result_input, result_label, orig_split_pos, ext_shape = GeneralDataset.split_volume(input, (4, 4, 4),
                                                                                            (3, 3, 3))

        result = GeneralDataset.combine_volume(result_input, result_input, input.shape, ext_shape, orig_split_pos,
                                               (3, 3, 3))

        assert result.shape == (8, 8, 8)
        assert input.sum() < result.sum()
