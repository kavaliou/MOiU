import numpy as np


E = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float).reshape((3, 3))


def _D(k, z):
    zk = z[k]
    z[k] = -1
    z /= -1 * zk
    d = E.copy()
    d[:, k] = z
    return d


def reversal_matrix(C):
    assert isinstance(C, np.ndarray)
    assert C.ndim == 2
    assert C.shape[0] == C.shape[1]

    n = C.shape[0]
    S = np.zeros((n, ))
    B = [E, ]
    C0 = [E.copy(), ]
    J = [range(3), ]

    for i in range(n):
        k = -1
        for num, j in enumerate(J[i]):
            e = C0[i][:, i].transpose()
            a = np.dot(e, B[i])
            a = np.dot(a, C[:, j])
            if a != 0:
                k = j
                break

        assert k != -1

        temp_array = J[i][:]
        temp_array.remove(k)
        J.append(temp_array)
        S[k] = i
        temp_C = C0[i].copy()
        _ci = C[:, k].copy()
        temp_C[:, i] = _ci
        C0.append(temp_C)

        temp_z = np.dot(B[i], C[i])
        temp_B = np.dot(_D(k, temp_z), B[i])
        B.append(temp_B)

    new_B = np.zeros((3,3))
    for num, s in enumerate(S):
        new_B[s] = B[-1][num]

    return new_B.transpose()
