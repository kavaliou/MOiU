import unittest

import numpy as np

from linear_programming import LinearProgrammingTask


class SimplexMethodCase(unittest.TestCase):
    def test_1(self):
        a = np.array([3, 1, 1, 0, 1, -2, 0, 1], dtype=np.float).reshape((2, 4))
        task = LinearProgrammingTask(
            a, np.array([1, 1]),
            np.array([1, 4, 1, -1]), np.array([0, 0, 1, 1], dtype=np.float)
        )
        self.assertTrue(task.solve_with_simplex_method())
        np.testing.assert_array_almost_equal(task.result, [0, 1, 0, 3])
        self.assertEqual(float(task.target_function_value), 1)

    def test_2(self):
        a = np.array(
            [0, 1, 4, 1, 0, -3, 1, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 7, -1, 0, -1, 3, -1, 0, 1, 1, 1, 1, 0, 3, -1, 1],
            dtype=np.float
        ).reshape((4, 8))
        task = LinearProgrammingTask(
            a, np.array([6, 10, -2, 15]), np.array([-5, -2, 3, -4, -6, 0, -1, -5]),
            np.array([10, 0, 1.5, 0, 0.5, 0, 0, 3.5], dtype=np.float)
        )
        self.assertTrue(task.solve_with_simplex_method())
        np.testing.assert_array_almost_equal(task.result, [10, 0, 2.2, 0, 2.6, 0.93333333, 0, 0])
        self.assertEqual(float(task.target_function_value), -59)

    def test_3(self):
        a = np.array(
            [0, 1, 4, 1, 0, -8, 1, 5, 0, -1, 0, -1, 0, 0, 0, 0, 0, 2, -1, 0, -1, 3, -1, 0, 1, 1, 1, 1, 0, 3, 1, 1],
            dtype=np.float
        ).reshape((4, 8))
        task = LinearProgrammingTask(
            a, np.array([36, -11, 10, 20]), np.array([-5, 2, 3, -4, -6, 0, 1, -5]),
            np.array([4, 5, 0, 6, 0, 0, 0, 5], dtype=np.float)
        )
        self.assertTrue(task.solve_with_simplex_method())
        np.testing.assert_array_almost_equal(task.result, [0, 9.5, 5.33333333, 1.5, 0, 0, 3.66666667, 0])
        self.assertAlmostEqual(float(task.target_function_value), 32.6666666667)

    def test_4(self):
        a = np.array([0, -1, 1, 7.5, 0, 0, 0, 2, 0, 2, 1, 0, -1, 3, -1.5, 0, 1, -1, 1, -1, 0, 3, 1, 1],
                     dtype=np.float
                     ).reshape((3, 8))
        task = LinearProgrammingTask(
            a, np.array([6, 1.5, 10]), np.array([-6, -9, -5, 2, -6, 0, 1, 3]),
            np.array([4, 0, 6, 0, 4.5, 0, 0, 0], dtype=np.float)
        )
        self.assertTrue(task.solve_with_simplex_method())
        np.testing.assert_array_almost_equal(task.result, [0, 0, 0, 0, 0, 1.6, 2.2, 3])
        self.assertAlmostEqual(float(task.target_function_value), 11.2)

    def test_5(self):
        a = np.array([-2, -1, 1, -7, 0, 0, 0, 2, 4, 2, -1, 0, 1, 5, -1, -5, 1, 11, 0, 1, 0, 3, 1, 1],
                     dtype=np.float
                     ).reshape((3, 8))
        task = LinearProgrammingTask(
            a, np.array([14, -31, 7]), np.array([6, -9, 5, -2, 6, 0, -1, 3]),
            np.array([4, 0, 6, 0, 4, 0, 0, 0], dtype=np.float)
        )
        self.assertTrue(task.solve_with_simplex_method())
        np.testing.assert_array_almost_equal(task.result, [0, 0, 26, 4, 40, 0, 0, 0])
        self.assertAlmostEqual(float(task.target_function_value), 362.0)

    def test_6(self):
        a = np.array([-2, 3, 1, 1, 0, 0, 1, 2, 2, 0, 1, 0, 2, -1, -3, 0, 0, 1],
                     dtype=np.float
                     ).reshape((3, 6))
        task = LinearProgrammingTask(
            a, np.array([36, 45, 30]), np.array([2, 4, 1, 0, 0, 0]),
            np.array([0, 0, 0, 36, 45, 30], dtype=np.float)
        )
        self.assertTrue(task.solve_with_simplex_method())
        np.testing.assert_array_almost_equal(task.result, [21, 12, 0, 42, 0, 0])
        self.assertAlmostEqual(float(task.target_function_value), 90)

    def test_7(self):
        a = np.array([1, 1, -2, 3, 2, -1, -1, 3],
                     dtype=np.float
                     ).reshape((2, 4))
        task = LinearProgrammingTask(
            a, np.array([1, 2]), np.array([1, 2, -1, 1]),
            np.array([0, 0, 1, 1], dtype=np.float)
        )
        self.assertFalse(task.solve_with_simplex_method())

    def test_8(self):
        a = np.array([-2, -1, 3, -7.5, 0, 0, 0, 2, 4, -2, 6, 0, 1, 5, -1, -4, 1, -1, 0, -1, 0, 3, 1, 1],
                     dtype=np.float).reshape((3, 8))
        task = LinearProgrammingTask(
            a, np.array([-23.5, -24, 2]), np.array([-6, 9, -5, 2, -6, 0, 1, 3]),
            np.array([0, 0, 0, 5, 4, 0, 0, 7], dtype=np.float)
        )
        self.assertFalse(task.solve_with_simplex_method())


if __name__ == '__main__':
    unittest.main()
