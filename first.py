import numpy as np
from numpy.linalg import det


def get_E(n):
    E = np.zeros((n, n), dtype=np.float)
    for i in xrange(n):
        E[i, i] = 1
    return E


def _D(k, z):
    zk = z[k]
    z[k] = -1
    z /= -1 * zk
    d = get_E(z.shape[0])
    d[:, k] = z
    return d


def reversal_matrix(C):
    assert isinstance(C, (np.ndarray, list, tuple))
    if isinstance(C, (list, tuple)):
        try:
            C = np.array(C, dtype=np.float)
        except ValueError:
            raise Exception("Matrix must be square")
    assert C.ndim == 2
    assert C.shape[0] == C.shape[1]
    assert det(C) != 0

    n = C.shape[0]
    E = get_E(n)
    S = np.zeros((n, ))
    B = [E, ]
    C0 = [E.copy(), ]
    J = [range(n), ]

    for i in xrange(n):
        k = -1
        for num, j in enumerate(J[i]):
            e = C0[i][:, i].transpose()
            a = np.dot(e, B[i])
            a = np.dot(a, C[:, j])
            if a != 0:
                k = j
                break

        assert k != -1, "Can't find the reversal matrix"

        temp_array = J[i][:]
        temp_array.remove(k)
        J.append(temp_array)
        S[k] = i
        temp_C = C0[i].copy()
        _ci = C[:, k].copy()
        temp_C[:, i] = _ci
        C0.append(temp_C)

        temp_z = np.dot(B[i], C[:, i])
        temp_B = np.dot(_D(k, temp_z), B[i])
        B.append(temp_B)

    new_B = np.zeros((n, n))
    for num, s in enumerate(S):
        new_B[s] = B[-1][num]

    return new_B
