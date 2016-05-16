from random import random

import numpy as np
from scipy.optimize import linprog


MULTIPLIER = 10
COUNT_CONSTANT = 50


def sixth(c, B, D, ci, Bi, Di, alpha, x_star, J_star):
    if D is None:
        D = np.dot(B.transpose(), B)
    if Di is None:
        Di = []
        for item in Bi:
            Di.append(np.dot(item.transpose(), item))

    n, m = D.shape[0], len(Di)

    while True:
        gi_x_star = []
        actives = []
        non_actives = []
        for i in range(m):
            gi_x_star.append(np.dot(ci[i], x_star) \
                             + 0.5 * np.dot(np.dot(x_star, Di[i]), x_star) \
                             + alpha[i])
            # if abs(gi_x_star[-1]) < 0.0001:
            #     gi_x_star[-1] = 0
            if gi_x_star[-1] == 0:
                actives.append(i)
            else:
                non_actives.append(i)

        Df__Dx = c + np.dot(D, x_star)

        Dg__Dx__i0 = []
        for i in actives:
            Dg__Dx__i0.append(ci[i] + np.dot(x_star, Di[i]))

        __c = Df__Dx.copy()
        A = np.array(Dg__Dx__i0, dtype=np.float64)
        b = [0] * len(actives)
        mn = [0 if x_star[i] == 0 else -1 for i in xrange(n)]
        mx = [1] * n
        ddd = zip(mn, mx)

        answer = linprog(__c, A_ub=A, b_ub=b, bounds=ddd).x

        F_x_star = np.dot(Df__Dx, answer)

        if F_x_star == 0:
            return 'WIN'  #

        x = np.array([0] * n, dtype=np.float64)
        dx = x - x_star
        b = np.dot(Df__Dx, dx)
        if b > 0:
            coefs_alpha = [- 2 * F_x_star / b]
            coefs_alpha.extend(- F_x_star / ((random() * MULTIPLIER + 1 + 1e-9) * b) for _ in xrange(COUNT_CONSTANT))
        else:
            coefs_alpha = [1]
            coefs_alpha.extend(random() * MULTIPLIER + 1e-9 for _ in xrange(COUNT_CONSTANT))

        ts = [0.5]
        ts.extend([random() for _ in xrange(COUNT_CONSTANT)])

        f_start = np.dot(c, x_star) + np.dot(np.dot(x_star, D), x_star) / 2
        for t in ts:
            for __alpha in coefs_alpha:
                x_new = x_star + (t * answer) + __alpha * t * dx
                if any(x_new[i] < 0 for i in xrange(n)):
                    continue
                f_new = np.dot(c, x_new) + np.dot(np.dot(x_new, D), x_new) / 2
                if f_new < f_start:
                    for i in xrange(m):
                        if (np.dot(ci[i], x_new) + np.dot(np.dot(x_new, Di[i]), x_new) / 2 + alpha[i]) > 0:
                            break
                    else:
                        return x_new

        raise Exception('There is no improving.')
