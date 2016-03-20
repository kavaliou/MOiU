from first import reversal_matrix, get_E
import numpy as np


def simplex_method(A, b, C, x0=None, J_b=None):
    n = len(C)
    assert x0 is not None or J_b is not None
    if J_b is None:
        J_b = [num for num, i in enumerate(x0) if i]
    J_nb = [j for j in xrange(n) if j not in J_b]
    A_basis = np.array([A[:, j] for j in J_b]).transpose()
    B = reversal_matrix(A_basis)
    if x0 is None:
        x0 = [0] * n
        for num, i in enumerate(np.dot(B, b)):
            x0[J_b[num]] = i





    while True:

        c_basis = [C[j] for j in J_b]
        u__ = np.dot(c_basis, B)

        deltas = {j: np.dot(u__, A[:, j]) - C[j] for j, x in enumerate(x0) if not x}

        deltas = np.dot(u__, A) - C

        negative_deltas = filter(lambda delta: delta[1] < 0, filter(lambda x: x[0] in J_nb, enumerate(deltas)))

        if not negative_deltas:
            return x0  # WIN

        delta = min([(nd[1], nd[0]) for nd in negative_deltas])
        j0 = delta[1]
        z = np.dot(B, A[:, j0])

        assert any([zi > 0 for zi in z])

        psi, s = min((x0[J_b[i]] / z[i], i) for i in xrange(len(J_b)) if z[i] > 0)

        # x_s = [x for x in x0 if x]
        # psi = min([x_s[num]/zi for num, zi in enumerate(z) if zi > 0])
        # s = [num for num, zi in enumerate(z) if x_s[num]/zi == psi and zi > 0][0]

        js = J_b[s]

        for j in J_nb:
            x0[j] = 0

        x0[j0] = psi

        for num, j in enumerate(J_b):
            x0[j] -= psi * z[num]

        J_b[s] = j0
        for i in xrange(len(J_nb)):
            if J_nb[i] == j0:
                J_nb[i] = js
                break

        d = z.copy()
        z_s = d[s]
        d[s] = -1
        d /= -1 * z_s
        M = get_E(d.shape[0])
        M[:, s] = d
        B = np.dot(M, B)
