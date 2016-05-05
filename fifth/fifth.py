import numpy as np

from first import reversal_matrix


def fifth(A, B, b, d, x0, J_op, J_star=None, c=None, D=None):
    if D is None:
        D = np.dot(B.transpose(), B)
    if J_star is None:
        J_star = J_op[:]
    if c is None:
        c = np.dot(d, B) * -1
    n, m = A.shape

    print J_op, J_star
    pass_step_first = False
    while True:
        if not pass_step_first:
            # step 1
            c_x0 = np.dot(D, x0) + c
            c_op_x0 = np.array([_c for num, _c in enumerate(c_x0) if num in J_op], dtype=np.float64)
            print 'c_op', c_op_x0
            # A_op = np.array([_A for num, _A in enumerate(A.transpose()) if num in J_op], dtype=np.float64).transpose()
            A_op = np.array([A[:, j] for j in J_op], dtype=np.float64).transpose()
            u_ = -1 * np.dot(c_op_x0.transpose(), reversal_matrix(A_op))
            deltas = np.dot(u_, A) + c_x0
            print 'deltas', deltas

            # step 2
            min_delta, j0 = min([(d, num) for num, d in enumerate(deltas) if num not in J_star])
            print 'min_delta, j0', min_delta, j0
            if min_delta >= 0:
                print 'WIN'
                break

        # step 3
        l = [0] * m
        l[j0] = 1
        D_star = np.array([[D[i, j] for j in J_star]for i in J_star], dtype=np.float64)
        D_star_j0 = np.array([D[i, j0] for i in J_star], dtype=np.float64)
        A_star = np.array([A[:, j] for j in J_star], dtype=np.float64).transpose()
        print 'D_star', D_star
        print 'A_star', A_star
        H = np.array(
            [list(D_star[i]) + list(A_star.transpose()[i]) for i in range(D_star.shape[0])] +
            [list(A_star[i]) + [0]*A_star.shape[0] for i in range(A_star.shape[0])], dtype=np.float64)
        bb = np.array(list(D_star_j0) + list(A[:, j0]), dtype=np.float64)
        l_J_star__delta_y = -1 * np.dot(reversal_matrix(H), bb)
        l_J_star = l_J_star__delta_y[:D_star.shape[0]]
        delta_y = l_J_star__delta_y[D_star.shape[0]:]
        for num, j in enumerate(J_star):
            l[j] = l_J_star[num]
        l = np.array(l, dtype=np.float64)
        print 'l', l

        # step 4
        teta = np.array([0] * m, dtype=np.float64)
        for j in J_star:
            if l[j] >= 0:
                teta[j] = 'inf'
            else:
                teta[j] = -1 * x0[j] / l[j]
        delta__ = np.dot(np.dot(l, D), l)
        teta[j0] = (abs(min_delta) / delta__) if delta__ > 0 else 'inf'
        min_teta, j_star = min([(d, num) for num, d in enumerate(teta) if num in J_star or num == j0])

        print 'min_teta, j*', min_teta, j_star
        # step 5
        x0 = x0 + min_teta * l
        print 'x0', x0
        # step 6
        if j_star == j0:
            J_star.append(j0)
            pass_step_first = False
            print J_star
            print 'punkt a'
        elif j_star in [j for j in J_star if j not in J_op]:
            J_star.remove(j_star)
            min_delta += min_teta * delta__
            pass_step_first = True
            print J_star, min_delta
            print 'punkt b'
        else:
            has_not_null = None
            for s, js in enumerate(J_op):
                for j_plus in [j for j in J_star if j not in J_op]:
                    e = np.array([1 if i==s else 0 for i in range(n)], dtype=np.float64)
                    if np.dot(np.dot(e, reversal_matrix(A_op)), A[:, j_plus]) != 0:
                        has_not_null = True
                        break
                if has_not_null:
                    break
            if has_not_null is not None and has_not_null:  # c
                print 'punkt c'
                J_op[J_op.index(j_star)] = j_plus
                J_star.remove(j_star)
                min_delta += min_teta * delta__
                pass_step_first = True
                print J_op, J_star, min_delta
            else:  # d
                print 'punkt d'
                J_op[J_op.index(j_star)] = j0
                J_star[J_star.index(j_star)] = j0
                print J_op, J_star
                pass_step_first = False
        print '===================='
    return x0