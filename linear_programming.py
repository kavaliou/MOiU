import numpy as np

from utils.utils import get_unit_matrix
from utils.reversal_matrix import reversal_matrix


class LinearProgrammingTask(object):
    def __init__(self, matrix_a, vector_b, vector_c, x0=None, j_basis=None):
        self.initial = {
            'matrix_a': matrix_a,
            'vector_b': vector_b,
            'vector_c': vector_c,
            'x0': x0,
            'j_basis': j_basis
        }
        self.matrix_a = None
        self.vector_b = None
        self.vector_c = None
        self.x0 = None
        self.j_basis = None
        self.matrix_b = None
        self._result = None
        self._exception_message = None

    def _set_variables(self):
        self.matrix_a = self.initial.get('matrix_a')
        self.vector_b = self.initial.get('vector_b')
        self.vector_c = self.initial.get('vector_c')
        self.x0 = self.initial.get('x0')
        self.j_basis = self.initial.get('j_basis')

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
                self._result = self.x0
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

    @property
    def result(self):
        if self._result is None:
            raise Exception("Task was not solved yet")
        return self._result

    @property
    def target_function_value(self):
        if self._result is None:
            raise Exception("Task was not solved yet")
        return sum(map(lambda q, w: q*w, self.vector_c, self._result))

    @property
    def exception_message(self):
        if self._exception_message is None:
            raise Exception("Task was not solved yet")
        return self._exception_message
