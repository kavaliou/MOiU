import unittest

import numpy as np

from linear_programming import LinearProgrammingTask


class SimplexMethodCase(unittest.TestCase):
    def test_1(self):
        c = np.array([0, -6, 6, 0, 0, 0, 0], dtype=np.float64)
        b = np.array([-1, 0, 4, 1], dtype=np.float64)
        A = np.array([
            [1, -2, 1, 1, 0, 0, 0],
            [-1, 2, -2, 0, 1, 0, 0],
            [2, 2, 1, 0, 0, 1, 0],
            [0, -2, 2, 0, 0, 0, 1]
        ], dtype=np.float64)
        jb = [3, 4, 5, 6]
        y = np.array([0, 0, 0, 0], dtype=np.float64)
        task = LinearProgrammingTask(
            A, b, c, j_basis=jb, y=y
        )
        self.assertTrue(task.solve_with_dual_simplex_method())
        self.assertAlmostEqual(task.get_target_function_value(task='simple'), task.get_target_function_value(task='dual'))
        np.testing.assert_array_almost_equal(task.result_y, [0, -3, 0, 0])
        self.assertEqual(float(task.get_target_function_value()), 0)

    def test_2(self):
        c = np.array([-3, -2, -1, 0, 0, 0], dtype=np.float64)
        b = np.array([-4, -6, -2], dtype=np.float64)
        A = np.array([
            [0, -1, -1, 1, 0, 0],
            [-2, -1, -2, 0, 1, 0],
            [-2, 1, -2, 0, 0, 1]
        ], dtype=np.float64)
        jb = [3, 4, 5]
        y = np.array([0, 0, 0], dtype=np.float64)
        task = LinearProgrammingTask(
            A, b, c, j_basis=jb, y=y
        )
        self.assertTrue(task.solve_with_dual_simplex_method())
        self.assertAlmostEqual(task.get_target_function_value(task='simple'), task.get_target_function_value(task='dual'))
        np.testing.assert_array_almost_equal(task.result_y, [1, 0, 0])
        self.assertEqual(float(task.get_target_function_value()), -4)

    def test_3(self):
        c = np.array([-4, 8, -18, -7, 0, 0, 0], dtype=np.float64)
        b = np.array([0, 5, -2], dtype=np.float64)
        A = np.array([
            [0, 0, -2, 1, 1, 0, 0],
            [2, 0, -1, 2, 0, 1, 0],
            [-2, 2, -2, -2, 0, 0, 1]
        ], dtype=np.float64)
        jb = [4, 5, 6]
        y = np.array([0, 0, 0], dtype=np.float64)
        task = LinearProgrammingTask(
            A, b, c, j_basis=jb, y=y
        )
        self.assertTrue(task.solve_with_dual_simplex_method())
        self.assertAlmostEqual(task.get_target_function_value(task='simple'), task.get_target_function_value(task='dual'))
        np.testing.assert_array_almost_equal(task.result_y, [0, 0, 2])
        self.assertEqual(float(task.get_target_function_value()), -4)

    def test_4(self):
        c = np.array([-4, 8, 8, -7], dtype=np.float64)
        b = np.array([-4, -6, 2], dtype=np.float64)
        A = np.array([
            [1, 1, 1, 1, 0, 0],
            [-1, 1, 0, 0, 1, 0],
            [-1, -2, 0, 0, 0, 1]
        ], dtype=np.float64)
        jb = [3, 4, 5]
        y = np.array([0, 0, 0], dtype=np.float64)
        task = LinearProgrammingTask(
            A, b, c, j_basis=jb, y=y
        )
        self.assertFalse(task.solve_with_dual_simplex_method())


if __name__ == '__main__':
    unittest.main()
