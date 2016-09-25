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
        task = LinearProgrammingTask(
            A, b, c, j_basis=jb
        )
        self.assertTrue(task.solve_with_dual_simplex_method())
        self.assertAlmostEqual(task.get_target_function_value(task='simple'),
                               task.get_target_function_value(task='dual'))
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
        task = LinearProgrammingTask(
            A, b, c, j_basis=jb
        )
        self.assertTrue(task.solve_with_dual_simplex_method())
        self.assertAlmostEqual(task.get_target_function_value(task='simple'),
                               task.get_target_function_value(task='dual'))
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
        task = LinearProgrammingTask(
            A, b, c, j_basis=jb
        )
        self.assertTrue(task.solve_with_dual_simplex_method())
        self.assertAlmostEqual(task.get_target_function_value(task='simple'),
                               task.get_target_function_value(task='dual'))
        np.testing.assert_array_almost_equal(task.result_y, [0, 0, 2])
        self.assertEqual(float(task.get_target_function_value()), -4)

    def test_4(self):
        c = np.array([-4, 8, 8, -7, 0, 0], dtype=np.float64)
        b = np.array([-4, -6, 2], dtype=np.float64)
        A = np.array([
            [1, 1, 1, 1, 0, 0],
            [-1, 1, 0, 0, 1, 0],
            [-1, -2, 0, 0, 0, 1]
        ], dtype=np.float64)
        jb = [3, 4, 5]
        task = LinearProgrammingTask(
            A, b, c, j_basis=jb
        )
        self.assertFalse(task.solve_with_dual_simplex_method())

    def test_5(self):
        c = np.array([3, 2, 0, 3, -2, -4], dtype=np.float64)
        b = np.array([2, 5, 0], dtype=np.float64)
        A = np.array([
            [2, 1, -1, 0, 0, 1],
            [1, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0]
        ], dtype=np.float64)
        d_bottom = [0, -1, 2, 1, -1, 0]
        d_top = [2, 4, 4, 3, 3, 5]
        task = LinearProgrammingTask(
            A, b, c,
            d_bottom=d_bottom, d_top=d_top
        )
        self.assertTrue(task.solve_with_dual_simplex_method_with_constraints())
        np.testing.assert_array_almost_equal(task.result_x, [1.5, 1, 2, 1.5, -1, 0])
        self.assertEqual(float(task.get_target_function_value()), 13)

    def test_task_1_from_method(self):
        c = np.array([7, -2, 6, 0, 5, 2], dtype=np.float64)
        b = np.array([-7, 22, 30], dtype=np.float64)
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
        self.assertTrue(task.solve_with_dual_simplex_method_with_constraints())
        np.testing.assert_array_almost_equal(task.result_x, [5, 3, 1, 0, 4, 6])
        self.assertEqual(float(task.get_target_function_value()), 67)

    def test_task_2_from_method(self):
        c = np.array([3, 0.5, 4, 4, 1, 5], dtype=np.float64)
        b = np.array([15, 0, 13], dtype=np.float64)
        A = np.array([[1, 0, 2, 2, -3, 3],
                      [0, 1, 0, -1, 0, 1],
                      [1, 0, 1, 3, 2, 1]], dtype=np.float64)
        d_bottom = [0, 0, 0, 0, 0, 0]
        d_top = [3, 5, 4, 3, 3, 4]
        task = LinearProgrammingTask(
            A, b, c,
            d_bottom=d_bottom, d_top=d_top
        )
        self.assertTrue(task.solve_with_dual_simplex_method_with_constraints())
        np.testing.assert_array_almost_equal(task.result_x, [3, 0, 4, 1.1818, 0.6364, 1.1818], 4)
        self.assertAlmostEqual(float(task.get_target_function_value()), 36.2727, 4)

    def test_task_3_from_method(self):
        c = np.array([2, 1, -2, -1, 4, -5, 5, 5], dtype=np.float64)
        b = np.array([40, 107, 61], dtype=np.float64)
        A = np.array([[1, 0, 0, 12, 1, -3, 4, -1],
                      [0, 1, 0, 11, 12, 3, 5, 3],
                      [0, 0, 1, 1, 0, 22, -2, 1]], dtype=np.float64)
        d_bottom = [0, 0, 0, 0, 0, 0, 0, 0]
        d_top = [3, 5, 5, 3, 4, 5, 6, 3]
        task = LinearProgrammingTask(
            A, b, c,
            d_bottom=d_bottom, d_top=d_top
        )
        self.assertTrue(task.solve_with_dual_simplex_method_with_constraints())
        np.testing.assert_array_almost_equal(task.result_x,
                                             [3, 5, 0, 1.8779, 2.7545, 3.0965, 6, 3], 4)
        self.assertAlmostEqual(float(task.get_target_function_value()), 49.6577, 4)

    def test_task_4_from_method(self):
        c = np.array([-1, 5, -2, 4, 3, 1, 2, 8, 3], dtype=np.float64)
        b = np.array([3, 9, 9, 5, 9], dtype=np.float64)
        A = np.array([[1, -3, 2, 0, 1, -1, 4, -1, 0],
                      [1, -1, 6, 1, 0, -2, 2, 2, 0],
                      [2, 2, -1, 1, 0, -3, 8, -1, 1],
                      [4, 1, 0, 0, 1, -1, 0, -1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float64)
        d_bottom = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        d_top = [5, 5, 5, 5, 5, 5, 5, 5, 5]
        task = LinearProgrammingTask(
            A, b, c,
            d_bottom=d_bottom, d_top=d_top
        )
        self.assertTrue(task.solve_with_dual_simplex_method_with_constraints())
        np.testing.assert_array_almost_equal(
            task.result_x,
            [1.1579, 0.6942, 0, 0, 2.8797, 0, 1.0627, 3.2055, 0], 4)
        self.assertAlmostEqual(float(task.get_target_function_value()), 38.7218, 4)

    def test_task_5_from_method(self):
        c = np.array([1, 2, 1, -3, 3, 1, 0], dtype=np.float64)
        b = np.array([1, 4, 7], dtype=np.float64)
        A = np.array([
            [1, 7, 2, 0, 1, -1, 4],
            [0, 5, 6, 1, 0, -3, -2],
            [3, 2, 2, 1, 1, 1, 5]
        ], dtype=np.float64)
        d_bottom = [-1, 1, -2, 0, 1, 2, 4]
        d_top = [3, 2, 2, 5, 3, 4, 5]
        task = LinearProgrammingTask(
            A, b, c,
            d_bottom=d_bottom, d_top=d_top
        )
        self.assertFalse(task.solve_with_dual_simplex_method_with_constraints())

    def test_task_6_from_method(self):
        c = np.array([0, 1, 2, 1, -3, 4, 7], dtype=np.float64)
        b = np.array([1.5, 9, 2], dtype=np.float64)
        A = np.array([[2, -1, 1, 0, 0, -1, 3],
                      [0, 4, -1, 2, 3, -2, 2],
                      [3, 1, 0, 1, 0, 1, 4]], dtype=np.float64)
        d_bottom = [0, 0, -3, 0, -1, 1, 0]
        d_top = [3, 3, 4, 7, 5, 3, 2]
        task = LinearProgrammingTask(
            A, b, c,
            d_bottom=d_bottom, d_top=d_top
        )
        self.assertTrue(task.solve_with_dual_simplex_method_with_constraints())
        np.testing.assert_array_almost_equal(
            task.result_x,
            [0, 1, 3.5, 0, 3.5, 1, 0], 4)
        self.assertAlmostEqual(float(task.get_target_function_value()), 1.5, 4)

    def test_task_7_from_method(self):
        c = np.array([0, -1, 1, 0, 4, 3], dtype=np.float64)
        b = np.array([2, 2, 5], dtype=np.float64)
        A = np.array([
            [2, 1, 0, 3, -1, -1],
            [0, 1, -2, 1, 0, 3],
            [3, 0, 1, 1, 1, 1]
        ], dtype=np.float64)
        d_bottom = [2, 0, -1, -3, 2, 1]
        d_top = [7, 3, 2, 3, 4, 5]
        task = LinearProgrammingTask(
            A, b, c,
            d_bottom=d_bottom, d_top=d_top
        )
        self.assertFalse(task.solve_with_dual_simplex_method_with_constraints())


if __name__ == '__main__':
    unittest.main()
