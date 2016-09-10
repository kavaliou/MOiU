import numpy as np


def get_unit_matrix(n):
    unit_matrix = np.zeros((n, n), dtype=np.float)
    for i in xrange(n):
        unit_matrix[i, i] = 1
    return unit_matrix


def get_matrix_d(k, z):
    zk = z[k]
    z[k] = -1
    z /= -1 * zk
    d = get_unit_matrix(z.shape[0])
    d[:, k] = z
    return d
