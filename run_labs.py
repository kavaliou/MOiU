import numpy as np

from linear_programming import LinearProgrammingTask


def test_task_for_first():
    c = np.array([-1, 5, -2, 4, 3, 1, 2, 8, 3], dtype=np.float64)
    b = np.array([3, 9, 9, 5, 9], dtype=np.float64)
    A = np.array([
        [1, -3, 2, 0, 1, -1, 4, -1, 0],
        [1, -1, 6, 1, 0, -2, 2, 2, 0],
        [2, 2, -1, 1, 0, -3, 8, -1, 1],
        [4, 1, 0, 0, 1, -1, 0, -1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.float64)
    d_bottom = [0] * 9
    d_top = [5] * 9
    task = LinearProgrammingTask(
        A, b, c,
        d_bottom=d_bottom, d_top=d_top
    )

    has_answer = task.solve_with_dual_simplex_method_with_constraints()

    print
    print 'Has answer: %s' % has_answer
    if has_answer:
        print 'x: %s' % map(lambda x: round(x, 3), task.result_x)
        print 'Target function: %s' % round(task.get_target_function_value(), 3)


def test_task_for_second():
    c = np.array([-2, 1, -2, -1, 8, -5, 3, 5, 1, 2], dtype=np.float64)
    b = np.array([27, 6, 18], dtype=np.float64)
    A = np.array([
        [1, 0, 0, 1, 1, -3, 4, -1, 3, 3],
        [0, 1, 0, -2, 1, 1, 7, 3, 4, 5],
        [0, 0, 1, 1, 0 ,2, -1, 1, -4, 7]
    ], dtype=np.float64)
    d_bottom = [0] * 10
    d_top = [8, 7, 6, 7, 8, 5, 6, 7, 8, 5]
    task = LinearProgrammingTask(
        A, b, c,
        d_bottom=d_bottom, d_top=d_top
    )

    has_answer = task.solve_integral_linear_task(True)

    print
    print 'Has answer: %s' % has_answer
    if has_answer:
        print 'x: %s' % map(lambda x: round(x, 3), task.result_x)
        print 'Target function: %s' % round(task.get_target_function_value(), 3)


def test_task_for_third():
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

    has_answer = task.solve_with_method_gomori()

    print
    print 'Has answer: %s' % has_answer
    if has_answer:
        print 'x: %s' % map(lambda x: round(x, 3), task.result_x)
        print 'Target function: %s' % round(task.get_target_function_value(), 3)


if __name__ == '__main__':
    test_task_for_third()
