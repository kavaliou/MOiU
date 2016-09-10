import numpy as np

from utils.reversal_matrix import reversal_matrix


def fifth(A, B, b, d, x0, J_op, J_star=None, c=None, D=None):
    if D is None:
        D = np.dot(B.transpose(), B)
    if J_star is None:
        J_star = J_op[:]
    if c is None:
        c = np.dot(d, B) * -1
    n, m = A.shape

    pass_step_first = False
    while True:
        if not pass_step_first:
            print '+++++++++++++++'
            # step 1
            c_x0 = np.dot(D, x0) + c
            c_op_x0 = np.array([_c for num, _c in enumerate(c_x0) if num in J_op], dtype=np.float64)
            A_op = np.array([_A for num, _A in enumerate(A.transpose()) if num in J_op], dtype=np.float64).transpose()
            # A_op = np.array([A[:, j] for j in J_op], dtype=np.float64).transpose()
            u_ = -1 * np.dot(c_op_x0.transpose(), reversal_matrix(A_op))
            deltas = np.dot(u_, A) + c_x0

            # step 2
            min_delta, j0 = min([(d, num) for num, d in enumerate(deltas) if num not in J_star])
            if min_delta >= 0:
                print 'WIN'
                break

        # step 3
        l = [0] * m
        l[j0] = 1
        D_star = np.array([[D[i, j] for j in J_star]for i in J_star], dtype=np.float64)
        D_star_j0 = np.array([D[i, j0] for i in J_star], dtype=np.float64)
        A_star = np.array([A[:, j] for j in J_star], dtype=np.float64).transpose()
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

        # step 5
        print min_teta, l
        x0 = x0 + min_teta * l
        # step 6
        print x0
        if j_star == j0:
            print 'a'
            J_star.append(j0)
            pass_step_first = False
        elif j_star in [j for j in J_star if j not in J_op]:
            print 'b'
            J_star.remove(j_star)
            min_delta += min_teta * delta__
            pass_step_first = True
        else:
            __index = 0
            __k = -1
            for val in J_op:
                if j_star == val:
                    __k = __index
            __index += 1
            assert __k != -1
            new_j = -1
            for j_plus in [j for j in J_star if j not in J_op]:
                e = np.array([1 if i==__k else 0 for i in range(n)], dtype=np.float64)
                if np.dot(np.dot(e, reversal_matrix(A_op)), A[:, j_plus]) > 0.0001:
                    new_j = j_plus

            # has_not_null = None
            # for s, js in enumerate(J_op):
            #     for j_plus in [j for j in J_star if j not in J_op]:
            #         e = np.array([1 if i==s else 0 for i in range(n)], dtype=np.float64)
            #         if np.dot(np.dot(e, reversal_matrix(A_op)), A[:, j_plus]) > 0.0001:
            #             has_not_null = True
            #             break
            #     if has_not_null:
            #         break
            # if has_not_null is not None and has_not_null:  # c

            if new_j != -1:
                print 'c'
                J_op.remove(j_star)
                J_op.append(new_j)
                J_star.remove(j_star)
                min_delta += min_teta * delta__
                pass_step_first = True
            else:  # d
                print 'd'
                J_op.remove(j_star)
                J_op.append(j0)
                J_star.remove(j_star)
                J_star.append(j0)
                pass_step_first = False
        print J_star, J_op
        print '--------'
    return x0
