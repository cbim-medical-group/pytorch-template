import numpy as np

from utils.clean_noise import CleanNoise


class TestCleanNoise:

    def test_clean_noise(self):
        cn = CleanNoise(top_num=1)

        test_case1 = np.array([[0, 0, 1, 1, 0],
                               [1, 0, 1, 0, 0],
                               [1, 0, 1, 0, 0]])
        result = cn.clean_small_obj(test_case1)
        assert (result == np.array([[0, 0, 1, 1, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 0, 1, 0, 0]])).all()

        test_case2 = np.array([[0, 0, 1, 1, 0],
                               [2, 0, 1, 0, 0],
                               [2, 0, 1, 0, 0]])
        result = cn.clean_small_obj(test_case2)
        assert (result == np.array([[0, 0, 1, 1, 0],
                                    [2, 0, 1, 0, 0],
                                    [2, 0, 1, 0, 0]])).all()

        test_case3 = np.array([[0, 0, 1, 1, 1],
                               [1, 0, 1, 1, 0],
                               [1, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0],
                               [1, 1, 1, 0, 0]])
        result = cn.clean_small_obj(test_case3)
        assert (result == np.array([[0, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 0],
                                    [1, 1, 1, 0, 0]])).all()

        test_case4 = np.array([[0, 0, 1, 1, 1],
                               [1, 0, 1, 1, 0],
                               [1, 0, 0, 0, 0],
                               [1, 1, 0, 0, 2],
                               [1, 1, 1, 0, 2]])
        result = cn.clean_small_obj(test_case4)
        assert (result == np.array([[0, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 2],
                                    [1, 1, 1, 0, 2]])).all()

        test_case5 = np.array([[0, 0, 10, 10, 10],
                               [1, 0, 1, 1, 0],
                               [1, 0, 0, 0, 0],
                               [1, 1, 0, 0, 10],
                               [1, 1, 1, 0, 10]])
        result = cn.clean_small_obj(test_case5)
        assert (result == np.array([[0, 0, 10, 10, 10],
                                    [1, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 0],
                                    [1, 1, 1, 0, 0]])).all()

        test_case5 = np.array([[0, 0, 10, 10, 10],
                               [10, 0, 10, 10, 0],
                               [10, 0, 0, 0, 0],
                               [10, 10, 0, 0, 10],
                               [10, 10, 10, 0, 10]])
        result = cn.clean_small_obj(test_case5)
        assert (result == np.array([[0, 0, 0, 0, 0],
                                    [10, 0, 0, 0, 0],
                                    [10, 0, 0, 0, 0],
                                    [10, 10, 0, 0, 0],
                                    [10, 10, 10, 0, 0]])).all()

    def test_clean_noise_top2(self):
        cn = CleanNoise(top_num=2)

        test_case1 = np.array([[0, 0, 1, 1, 0],
                               [1, 0, 1, 0, 0],
                               [1, 0, 1, 0, 0]])
        result = cn.clean_small_obj(test_case1)
        assert (result == np.array([[0, 0, 1, 1, 0],
                                    [1, 0, 1, 0, 0],
                                    [1, 0, 1, 0, 0]])).all()

        test_case2 = np.array([[0, 0, 1, 1, 0],
                               [2, 0, 1, 0, 0],
                               [2, 0, 1, 0, 0]])
        result = cn.clean_small_obj(test_case2)
        assert (result == np.array([[0, 0, 1, 1, 0],
                                    [2, 0, 1, 0, 0],
                                    [2, 0, 1, 0, 0]])).all()

        test_case3 = np.array([[0, 0, 1, 1, 1],
                               [1, 0, 1, 1, 0],
                               [1, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0],
                               [1, 1, 1, 0, 0]])
        result = cn.clean_small_obj(test_case3)
        assert (result == np.array([[0, 0, 1, 1, 1],
                                    [1, 0, 1, 1, 0],
                                    [1, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 0],
                                    [1, 1, 1, 0, 0]])).all()

        test_case4 = np.array([[0, 0, 1, 1, 1],
                               [1, 0, 1, 1, 0],
                               [1, 0, 0, 0, 0],
                               [1, 1, 0, 0, 2],
                               [1, 1, 1, 0, 2]])
        result = cn.clean_small_obj(test_case4)
        assert (result == np.array([[0, 0, 1, 1, 1],
                                    [1, 0, 1, 1, 0],
                                    [1, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 2],
                                    [1, 1, 1, 0, 2]])).all()

        test_case5 = np.array([[0, 0, 10, 10, 10],
                               [1, 0, 1, 1, 0],
                               [1, 0, 0, 0, 0],
                               [1, 1, 0, 0, 10],
                               [1, 1, 1, 0, 10]])
        result = cn.clean_small_obj(test_case5)
        assert (result == np.array([[0, 0, 10, 10, 10],
                                    [1, 0, 1, 1, 0],
                                    [1, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 10],
                                    [1, 1, 1, 0, 10]])).all()

        test_case5 = np.array([[0, 0, 10, 10, 10],
                               [10, 0, 10, 10, 0],
                               [10, 0, 0, 0, 0],
                               [10, 10, 0, 0, 10],
                               [10, 10, 10, 0, 10]])
        result = cn.clean_small_obj(test_case5)
        assert (result == np.array([[0, 0, 10, 10, 10],
                                    [10, 0, 10, 10, 0],
                                    [10, 0, 0, 0, 0],
                                    [10, 10, 0, 0, 0],
                                    [10, 10, 10, 0, 0]])).all()
