import numpy as np

from utils.utils import get_unit_matrix
from utils.reversal_matrix import reversal_matrix


class LinearProgrammingTask(object):
    def __init__(self, matrix_a, vector_b, vector_c, x0=None, j_basis=None, y=None):
        self.initial = {
            'matrix_a': matrix_a,
            'vector_b': vector_b,
            'vector_c': vector_c,
            'x0': x0,
            'j_basis': j_basis
        }
        if y is not None:
            self.initial.update({'y': y})

        self.matrix_a = None
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
        self.j_not_basis = [j for j in xrange(self.n) if j not in self.j_basis]

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
