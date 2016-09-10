import unittest

import numpy as np

from utils.reversal_matrix import reversal_matrix


class ReversalMatrixCase(unittest.TestCase):
    def test_1(self):
        c1 = np.array([1, 2, 2, 4, 1, 2, 4, 2, 3], dtype=np.float).reshape((3, 3))
        answer1 = [[1, 2, -2],
                   [4, 5, -6],
                   [-4, -6, 7]]
        np.testing.assert_array_almost_equal(np.array(reversal_matrix(c1)), np.array(answer1))

    def test_2(self):
        c2 = np.array([0, 2, 1, 0, 1, 1, 1, 1, 1], dtype=np.float).reshape((3, 3))
        answer2 = [[0, -1, 1],
                   [1, -1, 0],
                   [-1, 2, 0]]
        np.testing.assert_array_almost_equal(np.array(reversal_matrix(c2)), np.array(answer2))

    def test_3(self):
        c3 = [[2, 1, 1], [-1, 0, -1], [1, 1, 2]]
        answer3 = [[0.5, - 0.5, - 0.5],
                   [0.5, 1.5, 0.5],
                   [-0.5, -0.5, 0.5]]
        np.testing.assert_array_almost_equal(np.array(reversal_matrix(c3)), np.array(answer3))

    def test_4(self):
        c4 = [[0, 1, 1], [-1, 1, -1], [1, 1, 2]]
        answer4 = [[-3, 1, 2],
                   [-1, 1, 1],
                   [2, -1, -1]]
        np.testing.assert_array_almost_equal(np.array(reversal_matrix(c4)), np.array(answer4))

    def test_5(self):
        c5 = [[0, 2, 3], [0, 1, 1], [1, -1, 1]]
        answer5 = [[-2, 5, 1],
                   [-1, 3, 0],
                   [1, -2, 0]]
        np.testing.assert_array_almost_equal(np.array(reversal_matrix(c5)), np.array(answer5))

    def test_6(self):
        c6 = [[0, 2, 3], [0, 1, 1], [1, -1, 2]]
        answer6 = [[-3, 7, 1],
                   [-1, 3, 0],
                   [1, -2, 0]]
        np.testing.assert_array_almost_equal(np.array(reversal_matrix(c6)), np.array(answer6))

    def test_7(self):
        c7 = [[0, 2, 3], [0, 1, 1], [0, 1, 1]]
        with self.assertRaises(ValueError):
            reversal_matrix(c7)


if __name__ == '__main__':
    unittest.main()
