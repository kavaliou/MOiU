import numpy as np


E = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float).reshape((3, 3))


def D(k, z):
    zk = z[k]
    z[k] = -1
    z /= -1 * zk
    d = E.copy()
    d[:, k] = z
    return d


C = np.array([1, 2, 2, 4, 1, 2, 4, 2, 3], dtype=np.float).reshape((3, 3))
# C = np.array([0, 2, 1, 0, 1, 1, 1, 1, 1], dtype=np.float).reshape((3, 3))
n = 3

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

    if k == -1:
        raise Exception('Does not have reversal matrix')

    _ci = C[:, k].copy()
    temp_array = J[i][:]
    temp_array.remove(k)
    J.append(temp_array)
    S[k] = i
    temp_C = C0[i].copy()
    temp_C[:, i] = _ci
    C0.append(temp_C)

    temp_z = np.dot(B[i], C[i])
    temp_B = np.dot(D(k, temp_z), B[i])
    B.append(temp_B)

new_B = np.zeros((3,3))
for num, s in enumerate(S):
    new_B[s] = B[-1][num]

new_B = new_B.transpose()

print new_B
