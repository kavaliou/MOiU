import numpy as np

from first import reversal_matrix
from second import simplex_method


# C1 = np.array([1, 2, 2, 4, 1, 2, 4, 2, 3], dtype=np.float).reshape((3, 3))
# C2 = np.array([0, 2, 1, 0, 1, 1, 1, 1, 1], dtype=np.float).reshape((3, 3))

# print reversal_matrix(C1)
# print reversal_matrix(C2)


# A = np.array([3, 1, 1, 0, 1, -2, 0, 1], dtype=np.float).reshape((2, 4))
# print simplex_method(A, np.array([1, 1]), np.array([1, 4, 1, -1]), np.array([0, 0, 1, 1], dtype=np.float))
#
# !
# A = np.array([0, 1, 4, 1, 0, -3, 5, 0, 1, -1, 0, 1, 0, 0, 1, 0, 0, 7, -1, 0, -1, 3, 8, 0, 1, 1, 1, 1, 0, 3, -3, 1],
#              dtype=np.float).reshape((4, 8))
# print simplex_method(A, np.array([6, 10, -2, 15]), np.array([-5, -2, 3, -4, -6, 0, -1, -5]),
#                      np.array([4, 0, 0, 6, 2, 0, 0, 5], dtype=np.float))
#
# !
# A = np.array([0, 1, 4, 1, 0, -3, 1, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 7, -1, 0, -1, 3, -1, 0, 1, 1, 1, 1, 0, 3, -1, 1],
#              dtype=np.float).reshape((4, 8))
# s = simplex_method(A, np.array([6, 10, -2, 15]), np.array([-5, -2, 3, -4, -6, 0, -1, -5]),
#                      np.array([10, 0, 1.5, 0, 0.5, 0, 0, 3.5], dtype=np.float))
# print s, sum(map(lambda q, w: q*w, [-5, -2, 3, -4, -6, 0, -1, -5], s))


# !
# A = np.array([0, 1, 4, 1, 0, -8, 1, 5, 0, -1, 0, -1, 0, 0, 0, 0, 0, 2, -1, 0, -1, 3, -1, 0, 1, 1, 1, 1, 0, 3, 1, 1],
#              dtype=np.float).reshape((4, 8))
# s = simplex_method(A, np.array([36, -11, 10, 20]), np.array([-5, 2, 3, -4, -6, 0, 1, -5]),
#                      np.array([4, 5, 0, 6, 0, 0, 0, 5], dtype=np.float))
#
# print s, sum(map(lambda q, w: q*w, [-5, 2, 3, -4, -6, 0, 1, -5], s))

# ?
A = np.array([-2, 3, 1, 1, 0, 0, 1, 2, 2, 0, 1, 0, 2, -1, -3, 0, 0, 1],
             dtype=np.float).reshape((3, 6))
s = simplex_method(A, np.array([36, 45, 30]), np.array([2, 4, 1, 0, 0, 0]),
                     np.array([0, 0, 0, 1, 1, 1], dtype=np.float))

print s, sum(map(lambda q, w: q*w, [2, 4, 1, 0, 0, 0], s))

# !
# A = np.array([1, 1, -2, 3, 2, -1, -1, 3],
#              dtype=np.float).reshape((2, 4))
# s = simplex_method(A, np.array([1, 2]), np.array([1, 2, -1, 1]),
#                      np.array([0, 0, 1, 1], dtype=np.float))
#
# print s, sum(map(lambda q, w: q*w, [1, 2, -1, 1], s))
#
# ?
# A = np.array([0,-1,1,7.5,0,0,0,2,0,2,1,0,-1,3,-1.5,0,1,-1,1,-1,0,3,1,1],
#              dtype=np.float).reshape((3, 8))
# s = simplex_method(A, np.array([6, 1.5, 10]), np.array([-6, -9, -5, 2, -6, 0, 1, 3]),
#                      np.array([4, 0, 6, 0, 4.5, 0, 0, 0], dtype=np.float))
#
# print s, sum(map(lambda q, w: q*w, [-6, -9, -5, 2, -6, 0, 1, 3], s))

# !
# A = np.array([-2,-1,3,-7.5,0,0,0,2,4,-2,6,0,1,5,-1,-4,1,-1,0,-1,0,3,1,1],
#              dtype=np.float).reshape((3, 8))
# s = simplex_method(A, np.array([-23.5, -24, 2]), np.array([-6, 9, -5, 2, -6, 0, 1, 3]),
#                      np.array([0, 0, 0, 5, 4, 0, 0, 7], dtype=np.float))
#
# print s, sum(map(lambda q, w: q*w, [-6, 9, -5, 2, -6, 0, 1, 3], s))

# ?
# A = np.array([-2,-1,1,-7,0,0,0,2,4,2,-1,0,1,5,-1,-5,1,11,0,1,0,3,1,1],
#              dtype=np.float).reshape((3, 8))
# s = simplex_method(A, np.array([14, -31, 7]), np.array([6, -9, 5, -2, 6, 0, -1, 3]),
#                      np.array([4, 0, 6, 0, 4, 0, 0, 0], dtype=np.float))
#
# print s, sum(map(lambda q, w: q*w, [6, -9, 5, -2, 6, 0, -1, 3], s))

# A = np.array([-2, 3, 1, 1, 1, 2, 2, -1, -3], dtype=np.float).reshape((3, 3))
# print simplex_method(A, np.array([36, 45, 30]), np.array([2, 4, 1]), [1, 1, 1])

# print reversal_matrix([[2,1,1],[-1,0,-1],[1,1,2]])
# print reversal_matrix([[0,1,1],[-1,1,-1],[1,1,2]])
# print reversal_matrix([[0, 2, 1], [0, 1, 1], [1, 1, 1]])
