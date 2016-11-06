import unittest

import numpy as np

from linear_programming import LinearProgrammingTask


class MyTestCase(unittest.TestCase):
    def test_first(self):
        c = np.array([-3.5, 1, 0, 0, 0], dtype=np.float64)
        b = np.array([15, 6, 0], dtype=np.float64)
        A = np.array([
            [5, -1, 1, 0, 0],
            [-1, 2, 0, 1, 0],
            [-7, 2, 0, 0, 1],
        ], dtype=np.float64)
        task = LinearProgrammingTask(
            A, b, c, j_basis=[0, 1, 2]
        )

        self.assertTrue(task.solve_with_method_gomori())
        np.testing.assert_array_almost_equal(
            task.result_x,
            [0, 0, 15, 6, 0], 4)
        self.assertAlmostEqual(float(task.get_target_function_value()), 0, 4)

    def test_second(self):
        c = np.array([1, -1, 0, 0, 0], dtype=np.float64)
        b = np.array([4, 3, 7], dtype=np.float64)
        A = np.array([
            [5, 3, 1, 0, 0],
            [-1, 2, 0, 1, 0],
            [1, -2, 0, 0, 1],
        ], dtype=np.float64)
        task = LinearProgrammingTask(
            A, b, c, j_basis=[0, 3, 4]
        )

        self.assertTrue(task.solve_with_method_gomori())
        np.testing.assert_array_almost_equal(
            task.result_x,
            [0, 0, 4, 3, 7], 4)
        self.assertAlmostEqual(float(task.get_target_function_value()), 0, 4)

    def test_third(self):
        c = np.array([2, -5, 0, 0, 0], dtype=np.float64)
        b = np.array([-1, 10, 3], dtype=np.float64)
        A = np.array([
            [-2, -1, 1, 0, 0],
            [3, 1, 0, 1, 0],
            [-1, 1, 0, 0, 1],
        ], dtype=np.float64)
        task = LinearProgrammingTask(
            A, b, c, j_basis=[0, 2, 4]
        )

        self.assertTrue(task.solve_with_method_gomori())
        np.testing.assert_array_almost_equal(
            task.result_x,
            [3, 0, 5, 1, 6], 4)
        self.assertAlmostEqual(float(task.get_target_function_value()), 6, 4)

    def test_MAIN_TASK(self):
        c = np.array([7, -2, 6, 0, 5, 2], dtype=np.float64)
        b = np.array([-8, 22, 30], dtype=np.float64)
        A = np.array([
            [1, -5, 3, 1, 0, 0],
            [4, -1, 1, 0, 1, 0],
            [2, 4, 2, 0, 0, 1],
        ], dtype=np.float64)
        task = LinearProgrammingTask(
            A, b, c, j_basis=[3, 4, 5]
        )

        self.assertTrue(task.solve_with_method_gomori())
        np.testing.assert_array_almost_equal(
            task.result_x,
            [0, 2, 0, 2, 24, 22], 4)
        self.assertAlmostEqual(float(task.get_target_function_value()), 160, 4)


if __name__ == '__main__':
    unittest.main()
