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
                [[(self.vector_c[j] - np.dot(self.matrix_a[:, j].transpose(), self.y)) / mu[num], j] for num, j in enumerate(self.j_not_basis) if mu[num] < 0]
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
        deltas = [np.dot(self.y, self.matrix_a[:,j]) - self.vector_c[j] for j in J]
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
            deltas = [deltas[j] + sigma_0*mu[j] for j in J]
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
            return sum(map(lambda q, w: q*w, self.vector_c, self._result_x))
        if task == 'dual':
            return sum(map(lambda q, w: q*w, self.vector_b, self._result_y))

    @property
    def exception_message(self):
        self._raise_if_was_not_solved()
        return self._exception_message

    def _raise_if_was_not_solved(self):
        if self._result_x is None and self._exception_message is None:
            raise Exception("Task was not solved yet")
