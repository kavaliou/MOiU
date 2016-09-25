import unittest

import numpy as np

from linear_programming import LinearProgrammingTask


class SimplexMethodCase(unittest.TestCase):
    def test_task_1_from_example(self):
        c = np.array([7, -2, 6, 0, 5, 2], dtype=np.float64)
        b = np.array([-8, 22, 30], dtype=np.float64)
        A = np.array([
            [1, -5, 3, 1, 0, 0],
            [4, -1, 1, 0, 1, 0],
            [2, 4, 2, 0, 0, 1]
        ], dtype=np.float64)
        d_bottom = [2, 1, 0, 0, 1, 1]
        d_top = [6, 6, 5, 2, 4, 6]
        task = LinearProgrammingTask(
            A, b, c,
            d_bottom=d_bottom, d_top=d_top
        )
        self.assertTrue(task.solve_integral_linear_task())
        np.testing.assert_array_almost_equal(
            task.result_x,
            [6, 3, 0, 1, 1, 6], 4)
        self.assertAlmostEqual(float(task.get_target_function_value()), 53, 4)

    def test_task_2_from_example(self):
        c = np.array([7, -2, 6, 0, 5, 2], dtype=np.float64)
        b = np.array([10, 8, 10], dtype=np.float64)
        A = np.array([
            [1, 0, 3, 1, 0, 0],
            [0, -1, 1, 1, 1, 2],
            [-2, 4, 2, 0, 0, 1]
        ], dtype=np.float64)
        d_bottom = [0, 1, -1, 0, -2, 1]
        d_top = [3, 3, 6, 2, 4, 6]
        task = LinearProgrammingTask(
            A, b, c,
            d_bottom=d_bottom, d_top=d_top
        )
        self.assertTrue(task.solve_integral_linear_task())
        np.testing.assert_array_almost_equal(
            task.result_x,
            [1, 1, 3, 0, 2, 2], 4)
        self.assertAlmostEqual(float(task.get_target_function_value()), 37, 4)

    def test_task_3_from_example(self):
        c = np.array([-3, 2, 0, -2, -5, 2], dtype=np.float64)
        b = np.array([-3, 3, 13], dtype=np.float64)
        A = np.array([
            [1, 0, 1, 0, 0, 1],
            [1, 2, -1, 1, 1, 2],
            [-2, 4, 1, 0, 1, 0]
        ], dtype=np.float64)
        d_bottom = [-2, -1, -2, 0, 1, -4]
        d_top = [2, 3, 1, 5, 4, -1]
        task = LinearProgrammingTask(
            A, b, c,
            d_bottom=d_bottom, d_top=d_top
        )
        self.assertTrue(task.solve_integral_linear_task())
        np.testing.assert_array_almost_equal(
            task.result_x,
            [-2, 2, 0, 2, 1, -1], 4)
        self.assertAlmostEqual(float(task.get_target_function_value()), -1, 4)


if __name__ == '__main__':
    unittest.main()
