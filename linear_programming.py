import math

import numpy as np

from utils.utils import get_unit_matrix
from utils.reversal_matrix import reversal_matrix


class LinearProgrammingTask(object):
    def __init__(self, matrix_a, vector_b, vector_c, x0=None, j_basis=None, y=None,
                 d_bottom=None, d_top=None):
        self.initial = {
            'matrix_a': matrix_a,
            'vector_b': vector_b,
            'vector_c': vector_c,
            'x0': x0,
            'j_basis': j_basis,
            'y': y,
            'd_bottom': d_bottom,
            'd_top': d_top
        }

        self.matrix_a = None
        self.matrix_a_basis = None
        self.vector_b = None
        self.vector_c = None
        self.x0 = None
        self.j_basis = None
        self.y = None
        self.matrix_b = None
        self._result_x = None
        self._result_y = None
        self._exception_message = None
        self.n = None
        self.d_bottom = None
        self.d_top = None

    def _set_variables(self):
        for key, value in self.initial.iteritems():
            setattr(self, key, value)

    def _prepare_task_for_simplex_method(self):
        self._set_variables()

        self.n = len(self.vector_c)

        assert self.x0 is not None or self.j_basis is not None

        if self.j_basis is None:
            self.j_basis = [num for num, i in enumerate(self.x0) if i]

        self.j_not_basis = [j for j in xrange(self.n) if j not in self.j_basis]

        self.matrix_a_basis = np.array([self.matrix_a[:, j] for j in self.j_basis]).transpose()

        self.matrix_b = reversal_matrix(self.matrix_a_basis)

        if self.x0 is None:
            self.x0 = [0] * self.n
            for num, i in enumerate(np.dot(self.matrix_b, self.vector_b)):
                self.x0[self.j_basis[num]] = i

    def solve_with_simplex_method(self):
        self._prepare_task_for_simplex_method()

        while True:
            c_basis = [self.vector_c[j] for j in self.j_basis]
            u__ = np.dot(c_basis, self.matrix_b)
            deltas = np.dot(u__, self.matrix_a) - self.vector_c
            negative_deltas = filter(
                lambda _d: _d[1] < 0,
                filter(lambda x: x[0] in self.j_not_basis, enumerate(deltas))
            )

            if not negative_deltas:
                self._result_x = self.x0
                return True

            delta = min([(nd[1], nd[0]) for nd in negative_deltas])
            j0 = delta[1]
            z = np.dot(self.matrix_b, self.matrix_a[:, j0])

            if not any([zi > 0 for zi in z]):
                self._exception_message = "Task is not bounded above"
                return False

            psi, s = min((self.x0[self.j_basis[i]] / z[i], i) for i in xrange(len(self.j_basis)) if z[i] > 0)

            js = self.j_basis[s]

            for j in self.j_not_basis:
                self.x0[j] = 0

            self.x0[j0] = psi

            for num, j in enumerate(self.j_basis):
                self.x0[j] -= psi * z[num]

            self.j_basis[s] = j0
            for i in xrange(len(self.j_not_basis)):
                if self.j_not_basis[i] == j0:
                    self.j_not_basis[i] = js
                    break

            d = z.copy()
            z_s = d[s]
            d[s] = -1
            d /= -1 * z_s
            matrix_m = get_unit_matrix(d.shape[0])
            matrix_m[:, s] = d
            self.matrix_b = np.dot(matrix_m, self.matrix_b)

    def _prepare_task_for_dual_simplex_method(self):
        self._set_variables()
        self.n = len(self.vector_c)
        self.m = len(self.vector_b)

        answer = []

        def gen(n, m, mas=None):
            if mas is None:
                mas = []

            for i in range(n):
                if i not in mas:
                    mas.append(i)
                    if m == len(mas):
                        answer.append(mas[:])
                    else:
                        gen(n, m, mas)
                    mas.remove(i)

        if self.j_basis is None:
            gen(self.n, self.m)
            for i in answer:
                matrix_a_basis = np.array([self.matrix_a[:, j] for j in i]).transpose()
                if np.linalg.det(matrix_a_basis):
                    self.j_basis = i
                    self.matrix_a_basis = matrix_a_basis
                    break

        self.j_not_basis = [j for j in xrange(self.n) if j not in self.j_basis]

        if self.matrix_a_basis is None:
            self.matrix_a_basis = np.array([self.matrix_a[:, j] for j in self.j_basis]).transpose()
        self.matrix_b = reversal_matrix(self.matrix_a_basis)
        c_basis = [self.vector_c[j] for j in xrange(self.n) if j in self.j_basis]
        if self.y is None:
            self.y = np.dot(c_basis, self.matrix_b)

    def solve_with_dual_simplex_method(self):
        self._prepare_task_for_dual_simplex_method()

        while True:
            A_basis = np.array([self.matrix_a[:, j] for j in self.j_basis]).transpose()
            B = reversal_matrix(A_basis)
            kappa = np.dot(B, self.vector_b)

            negative_kappas = filter(lambda k: k[1] < 0, enumerate(kappa))

            if not negative_kappas:
                x = [0] * self.n
                for num, j in enumerate(self.j_basis):
                    x[j] = kappa[num]
                self._result_x = x
                self._result_y = self.y
                return True

            delta = min([(nd[1], nd[0]) for nd in negative_kappas])
            js = delta[1]

            mu = [np.dot(B[js], self.matrix_a[:, j]) for j in self.j_not_basis]
            sigma = min(
                [[(self.vector_c[j] - np.dot(self.matrix_a[:, j].transpose(), self.y)) / mu[num], j] for num, j in
                 enumerate(self.j_not_basis) if mu[num] < 0]
                or [None])
            if sigma is None:
                self._exception_message = "Set of permissible plans is empty"
                return False

            self.y = self.y + sigma[0] * B[js]
            j0 = sigma[1]

            js = self.j_basis[js]

            for i in xrange(len(self.j_not_basis)):
                if self.j_not_basis[i] == j0:
                    self.j_not_basis[i] = js
                    break

            for i in xrange(len(self.j_basis)):
                if self.j_basis[i] == js:
                    self.j_basis[i] = j0
                    break

    def solve_with_dual_simplex_method_with_constraints(self):
        self._prepare_task_for_dual_simplex_method()
        J = sorted(self.j_basis + self.j_not_basis)
        # step 1
        deltas = [np.dot(self.y, self.matrix_a[:, j]) - self.vector_c[j] for j in J]
        j_not_basis_plus = [num for num, elem in enumerate(deltas)
                            if elem >= 0 and num in self.j_not_basis]
        j_not_basis_minus = [elem for elem in self.j_not_basis
                             if elem not in j_not_basis_plus]

        while True:
            # step 2
            kappas = [0] * self.n
            for j in J:
                if j in j_not_basis_plus:
                    kappas[j] = self.d_bottom[j]
                elif j in j_not_basis_minus:
                    kappas[j] = self.d_top[j]
            s = sum([self.matrix_a[:, j_] * kappas[j_]
                     for j_ in sorted(j_not_basis_minus + j_not_basis_plus)])
            kappas_a = np.dot(self.matrix_b, self.vector_b - s)
            for num, j in enumerate(self.j_basis):
                kappas[j] = kappas_a[num]

            # step 3
            if all(map(lambda (_d_bottom, _kappa, _d_top): _d_bottom <= _kappa <= _d_top,
                       zip(self.d_bottom, kappas, self.d_top))):
                self._result_x = kappas
                return True

            # step 4
            k, j_k = min([(num, j) for num, j in enumerate(self.j_basis)
                          if not (self.d_bottom[j] <= kappas[j] <= self.d_top[j])])

            # step 5
            mu_j_k = 1 if kappas[j_k] < self.d_bottom[j_k] else -1
            delta_y = mu_j_k * np.dot(np.array([1 if i == k else 0 for i in range(len(self.j_basis))]),
                                      self.matrix_b)
            mu = [np.dot(delta_y, self.matrix_a[:, j]) for j in J]

            # step 6
            sigmas = [-float(deltas[j]) / mu[j]
                      if (j in j_not_basis_plus and mu[j] < 0)
                         or (j in j_not_basis_minus and mu[j] > 0) else 'inf'
                      for j in sorted(j_not_basis_plus + j_not_basis_minus)]
            sigmas = zip(sigmas, sorted(j_not_basis_plus + j_not_basis_minus))
            sigma_0, j_star = min(sigmas)
            sigma_0 = float(sigma_0)

            # step 7
            if sigma_0 == float('inf'):
                self._exception_message = "Set of permissible plans is empty"
                return False

            # step 8
            deltas = [deltas[j] + sigma_0 * mu[j] for j in J]
            self.j_basis[k] = j_star
            self.matrix_a_basis = np.array([self.matrix_a[:, j] for j in self.j_basis]).transpose()
            self.matrix_b = reversal_matrix(self.matrix_a_basis)

            # step 9
            self.j_not_basis = [j for j in xrange(self.n) if j not in self.j_basis]
            if j_star in j_not_basis_plus:
                if mu_j_k == 1:
                    j_not_basis_plus[j_not_basis_plus.index(j_star)] = j_k
                else:
                    j_not_basis_plus.remove(j_star)
            else:
                if mu_j_k == 1:
                    j_not_basis_plus.append(j_k)

            j_not_basis_minus = [elem for elem in self.j_not_basis
                                 if elem not in j_not_basis_plus]

    def solve_integral_linear_task(self, log=False):
        self._set_variables()

        def round_function(number, ndigits_for_round=6):
            if round(number, ndigits_for_round) == round(number):
                return int(round(number))
            else:
                return round(number, ndigits_for_round)

        linear_programming_tasks_array = [LinearProgrammingTask(
            self.matrix_a, self.vector_b, self.vector_c,
            d_bottom=self.d_bottom, d_top=self.d_top
        )]
        integral_task_result = float('-inf')
        integral_x = None

        tasks_count = 0
        while linear_programming_tasks_array:
            tasks_count += 1
            if log:
                print '-------------------'

            task = linear_programming_tasks_array.pop(0)
            task_result = task.solve_with_dual_simplex_method_with_constraints()
            if not task_result:
                if log:
                    print 'not solve'
                continue

            x0 = task.result_x
            x0 = map(round_function, x0)
            if log:
                print 'x =', x0
            not_integral_result_x0 = filter(lambda x: not isinstance(x, int), x0)

            if task.get_target_function_value() <= integral_task_result:
                if log:
                    print 'target function value %s lesser than current record' % task.get_target_function_value()
                continue
            if not not_integral_result_x0:
                integral_task_result = task.get_target_function_value()
                if log:
                    print 'save task result =', integral_task_result
                integral_x = x0[:]
                continue

            x, j = not_integral_result_x0[0], x0.index(not_integral_result_x0[0])
            x_integral_part = math.floor(x)
            x_bottom, x_top = int(x_integral_part), int(x_integral_part + 1)
            d_bottom_new = task.d_bottom[:]
            d_bottom_new[j] = x_top
            d_top_new = task.d_top[:]
            d_top_new[j] = x_bottom

            # Added j_basis
            linear_programming_tasks_array.insert(0, LinearProgrammingTask(
                self.matrix_a, self.vector_b, self.vector_c,
                d_bottom=task.d_bottom, d_top=d_top_new, j_basis=task.j_basis
            ))
            linear_programming_tasks_array.insert(0, LinearProgrammingTask(
                self.matrix_a, self.vector_b, self.vector_c,
                d_bottom=d_bottom_new, d_top=task.d_top, j_basis=task.j_basis
            ))

        if log:
            print 'count =', tasks_count

        self._result_x = integral_x
        return integral_x is not None

    def solve_with_method_gomori(self, log=False):
        self._set_variables()

        def round_function(number, ndigits_for_round=6):
            if round(number, ndigits_for_round) == round(number):
                return int(round(number))
            else:
                return round(number, ndigits_for_round)

        def not_integral_part(number):
            return number - math.floor(number)

        n = len(self.vector_c)
        m = len(self.vector_b)
        j_basis = self.j_basis
        j_all = range(n)
        j_synthetic = []
        m_synthetic = []
        matrix_a = self.matrix_a
        vector_b = self.vector_b
        vector_c = self.vector_c

        while True:
            linear_programming_task = LinearProgrammingTask(
                matrix_a, vector_b, vector_c, j_basis=j_basis[:]
            )
            has_answer = linear_programming_task.solve_with_dual_simplex_method()

            fresh_j_basis = linear_programming_task.j_basis
            x0 = map(round_function, linear_programming_task.result_x)

            print 'x0 =', x0
            print 'j_basis =', fresh_j_basis

            # reduce task size
            synthetic_indexes_for_delete = [j for j in j_synthetic if j in fresh_j_basis]
            while synthetic_indexes_for_delete:
                idx = synthetic_indexes_for_delete[0]
                synthetic_map = dict(zip(j_synthetic, m_synthetic))

                row = matrix_a[synthetic_map[idx], :]
                _b = vector_b[synthetic_map[idx]]
                for i in m_synthetic:
                    if i == synthetic_map[idx]:
                        continue
                    row_for_change = matrix_a[i, :]
                    mul = row_for_change[idx]
                    sub_row = mul * row
                    vector_b[i] = vector_b[i] - mul * _b
                    matrix_a[i, :] = row_for_change - sub_row

                matrix_a = np.delete(matrix_a, idx, 1)
                matrix_a = np.delete(matrix_a, synthetic_map[idx], 0)
                vector_b = np.delete(vector_b, synthetic_map[idx], 0)
                vector_c = np.delete(vector_c, idx, 0)

                j_synthetic.pop()
                m_synthetic.pop()
                n -= 1
                m -= 1
                fresh_j_basis.remove(idx)
                x0.pop((j_all+j_synthetic).index(idx))

                # recalculate
                synthetic_indexes_for_delete = [j for j in j_synthetic if j in fresh_j_basis]

            x_opt = [x for idx, x in enumerate(x0) if idx in fresh_j_basis]
            not_integral_result_x0 = filter(lambda x: not isinstance(x, int), x_opt)

            print 'not integral x0 = ', not_integral_result_x0

            if not not_integral_result_x0:
                self._result_x = x0[:len(j_all)]
                return True

            matrix_a_basis = np.array([matrix_a[:, j] for j in fresh_j_basis]).transpose()
            reverse_a_matrix = reversal_matrix(matrix_a_basis)

            i0 = x0.index(not_integral_result_x0[0])
            s = fresh_j_basis.index(i0)
            y = reverse_a_matrix[s, :]

            print 'i0 =', i0
            print 's =', s

            a = [np.dot(y, matrix_a[:, j]) for j in j_all + j_synthetic]
            b = np.dot(y, vector_b)

            j_array = set(j_all + j_synthetic) - set(fresh_j_basis)

            a_new = [0] * (len(j_all) + len(j_synthetic))
            for j in j_array:
                a_new[j] = -not_integral_part(a[j])
            a_new.append(1)
            print 'new bound in matrix a =', map(round_function, a_new)
            print 'new bound in vector b =', round_function(b)
            m_a = [list(i) + [0] for i in matrix_a]
            m_a.append(a_new)
            matrix_a = np.array(m_a, dtype=np.float64)

            v_b = list(vector_b)
            v_b.append(-not_integral_part(b))
            vector_b = np.array(v_b, dtype=np.float64)

            v_c = list(vector_c)
            v_c.append(0)
            vector_c = np.array(v_c, dtype=np.float64)

            j_synthetic.append(n)
            m_synthetic.append(m)
            j_basis = fresh_j_basis[:]
            j_basis.append(n)
            n += 1
            m += 1

            print '\n----------\n'

    @property
    def result_x(self):
        self._raise_if_was_not_solved()
        return self._result_x

    @property
    def result_y(self):
        self._raise_if_was_not_solved()
        return self._result_y

    def get_target_function_value(self, task='simple'):
        self._raise_if_was_not_solved()
        if task == 'simple':
            return sum(map(lambda q, w: q * w, self.vector_c, self._result_x))
        if task == 'dual':
            return sum(map(lambda q, w: q * w, self.vector_b, self._result_y))

    @property
    def exception_message(self):
        self._raise_if_was_not_solved()
        return self._exception_message

    def _raise_if_was_not_solved(self):
        if self._result_x is None and self._exception_message is None:
            raise Exception("Task was not solved yet")
