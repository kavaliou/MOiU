import numpy as np
from .second import simplex_method

# 4
A = np.array([-2, 3, 1, 1, 0, 0, 1, 2, 2, 0, 1, 0, 2, -1, -3, 0, 0, 1],
             dtype=np.float).reshape((3, 6))
s = simplex_method(A, np.array([36, 45, 30]), np.array([2, 4, 1, 0, 0, 0]),
                     np.array([0, 0, 0, 36, 45, 30], dtype=np.float))
print s, sum(map(lambda q, w: q*w, [2, 4, 1, 0, 0, 0], s))
print


A = np.array([1, -1, -1, -1, 3, 3, 1, 1],
             dtype=np.float).reshape((2, 4))
s = simplex_method(A, np.array([0, 12]), np.array([2, 0, 1, -1]),
                     np.array([2, 2, 0, 0], dtype=np.float))
print s, sum(map(lambda q, w: q*w, [2, 2, 0, 0], s))
print


# A = np.array([2, 7, 1, 0, 1, 0, 0,
#               1, 4, 2, 8, 0, 1, 0,
#               -1, 0, 2, 5, 0, 0, 1],
#              dtype=np.float).reshape((3, 7))
# s = simplex_method(A, np.array([5, 6, 9]), np.array([1, 5, 4, -3, 0, 0, 0]), J_b=[4, 5, 6])
# print s, sum(map(lambda q, w: q*w, [-1, -5, -4, 3, 0, 0, 0], s))
#

# A = np.array([2, 7, 1, 0, 1, 0, 0,
#               1, 4, 2, 8, 0, 1, 0,
#               -1, 0, 2, 5, 0, 0, 1],
#              dtype=np.float).reshape((3, 7))
# s = simplex_method(A, np.array([5, 6, 9]), np.array([-1, -5, -4, 3, 0, 0, 0]), J_b=[4, 5, 6])
# print s, sum(map(lambda q, w: q*w, [-1, -5, -4, 3, 0, 0, 0], s))

A = np.array([-3, 1, 1, 1, 0, 0,
              -1, 2, 2, 0, 1, 0,
              1, -3, 1, 0, 0, 1],
             dtype=np.float).reshape((3, 6))
s = simplex_method(A, np.array([10, 70, 10]), np.array([4, 6, 4, 0, 0, 0]), J_b=[3, 4, 5])
print s, sum(map(lambda q, w: q*w, [4, 6, 4, 0, 0, 0], s))
