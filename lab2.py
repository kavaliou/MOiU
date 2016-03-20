import numpy as np
from second import simplex_method

# 4
A = np.array([-2, 3, 1, 1, 0, 0, 1, 2, 2, 0, 1, 0, 2, -1, -3, 0, 0, 1],
             dtype=np.float).reshape((3, 6))
s = simplex_method(A, np.array([36, 45, 30]), np.array([2, 4, 1, 0, 0, 0]),
                     np.array([0, 0, 0, 36, 45, 30], dtype=np.float))
print s, sum(map(lambda q, w: q*w, [2, 4, 1, 0, 0, 0], s))
print

# 7
A = np.array([2, 7, 1, 0, 0, 0,
              1, 4, 2, 8, 1, 0,
              -1, 0, 2, 5, 0, 1],
             dtype=np.float).reshape((3, 6))
s = simplex_method(A, np.array([5, 6, 9]), np.array([-1, -5, -4, 3, 0, 0]),
                     np.array([0, 0, 1, 0, 6, 9], dtype=np.float))
print s, sum(map(lambda q, w: q*w, [-1, -5, -4, 3, 0, 0], s))
