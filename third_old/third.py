# coding=utf-8
from utils.reversal_matrix import reversal_matrix
import numpy as np


def dual_simplex_method(A, b, C, y, J_b):
    n = len(C)
    J_nb = [j for j in xrange(n) if j not in J_b]

    while True:
        A_basis = np.array([A[:, j] for j in J_b]).transpose()
        B = reversal_matrix(A_basis)
        kappa = np.dot(B, b)

        negative_kappas = filter(lambda k: k[1] < 0, enumerate(kappa))

        if not negative_kappas:
            x = [0] * n
            for num, j in enumerate(J_b):
                x[j] = kappa[num]
            return x, y, J_b

        delta = min([(nd[1], nd[0]) for nd in negative_kappas])
        js = delta[1]

        mu = [np.dot(B[js], A[:, j]) for j in J_nb]
        sigma = min([[(C[j] - np.dot(A[:, j].transpose(), y)) / mu[num], j] for num, j in enumerate(J_nb) if mu[num] < 0]
                    or [None])
        assert sigma is not None, "LOSE Задача не имеет решения, т.к. пусто множество ее допустимых планов."

        y = y + sigma[0] * B[js]
        j0 = sigma[1]

        js = J_b[js]

        for i in xrange(len(J_nb)):
            if J_nb[i] == j0:
                J_nb[i] = js
                break

        for i in xrange(len(J_b)):
            if J_b[i] == js:
                J_b[i] = j0
                break
