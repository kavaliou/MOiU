import numpy as np
from numpy.linalg import det

from .exceptions import ReversalMatrixNotExistError
from .utils import get_matrix_d, get_unit_matrix


def reversal_matrix(matrix):
    if not isinstance(matrix, (np.ndarray, list, tuple)):
        raise TypeError("'matrix' must be np.ndarray, list or tuple")

    if isinstance(matrix, (list, tuple)):
        try:
            matrix = np.array(matrix, dtype=np.float)
        except ValueError:
            raise ValueError("Matrix must be square")

    if matrix.ndim != 2:
        raise ValueError("Matrix must be two-dimensional")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    if det(matrix) == 0:
        raise ValueError("Determinant of matrix must not be equal null")

    n = matrix.shape[0]
    unit_matrix = get_unit_matrix(n)
    S = np.zeros((n, ))
    B = [unit_matrix, ]
    C0 = [unit_matrix.copy(), ]
    J = [range(n), ]

    for i in xrange(n):
        k = -1
        for num, j in enumerate(J[i]):
            unit_vector = unit_matrix[:, i].transpose()
            a = np.dot(unit_vector, B[i])
            a = np.dot(a, matrix[:, j])
            if a != 0:
                k = j
                break

        if k == -1:
            raise ReversalMatrixNotExistError("Can't find the reversal matrix")

        temporary_array = J[i][:]
        temporary_array.remove(k)
        J.append(temporary_array)
        S[k] = i
        temporary_matrix_c = C0[i].copy()
        temporary_ci = matrix[:, k].copy()
        temporary_matrix_c[:, i] = temporary_ci
        C0.append(temporary_matrix_c)

        temporary_z = np.dot(B[i], temporary_ci)
        temporary_matrix_b = np.dot(get_matrix_d(i, temporary_z), B[i])
        B.append(temporary_matrix_b)

    returning_matrix_b = np.zeros((n, n))
    for num, s in enumerate(S):
        returning_matrix_b[num] = B[-1][s]

    return returning_matrix_b
