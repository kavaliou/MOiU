import numpy as np

from second import simplex_method
from third import dual_simplex_method


# c = np.array([0, -6, 6, 0, 0, 0, 0], dtype=np.float64)
# b = np.array([-1, 0, 4, 1], dtype=np.float64)
# A = np.array([
#         [1, -2, 1, 1, 0, 0, 0],
#         [-1, 2, -2, 0, 1, 0, 0],
#         [2, 2, 1, 0, 0, 1, 0],
#         [0, -2, 2, 0, 0, 0, 1]
# ], dtype=np.float64)
# jb = [3, 4, 5, 6]
# y = np.array([0, 0, 0, 0], dtype=np.float64)
# try:
#     s = dual_simplex_method(A, b, c, y, jb)
#     print s[0], sum(map(lambda q, w: q*w, c, s[0]))
#     print s[1], sum(map(lambda q, w: q*w, b, s[1]))
# except AssertionError as e:
#     print e.message
# print

# c = np.array([-3, -2, -1, 0, 0, 0], dtype=np.float64)
# b = np.array([-4, -6, -2], dtype=np.float64)
# A = np.array([
#         [0, -1, -1, 1, 0, 0],
#         [-2, -1, -2, 0, 1, 0],
#         [-2, 1, -2, 0, 0, 1]
# ], dtype=np.float64)
# jb = [3, 4, 5]
# y = np.array([0, 0, 0], dtype=np.float64)
# try:
#     s = dual_simplex_method(A, b, c, y, jb)
#     print s[0], sum(map(lambda q, w: q*w, c, s[0]))
#     print s[1], sum(map(lambda q, w: q*w, b, s[1]))
# except AssertionError as e:
#     print e.message
# print

c = np.array([-4, 8, -18, -7, 0, 0, 0], dtype=np.float64)
b = np.array([0, 5, -2], dtype=np.float64)
A = np.array([
        [0, 0, -2, 1, 1, 0, 0],
        [2, 0, -1, 2, 0, 1, 0],
        [-2, 2, -2, -2, 0, 0, 1]
], dtype=np.float64)
jb = [4, 5, 6]
y = np.array([0, 0, 0], dtype=np.float64)
try:
    s = dual_simplex_method(A, b, c, y, jb)
    print s[0], sum(map(lambda q, w: q*w, c, s[0]))
    print s[1], sum(map(lambda q, w: q*w, b, s[1]))
    s = simplex_method(A, b, c, s[0], s[2])
    print s, sum(map(lambda q, w: q*w, c, s))
except AssertionError as e:
    print e.message
print

c = np.array([-4, 8, 8, -7], dtype=np.float64)
b = np.array([-4, -6, 2], dtype=np.float64)
A = np.array([
        [1, 1, 1, 1, 0, 0],
        [-1, 1, 0, 0, 1, 0],
        [-1, -2, 0, 0, 0, 1]
], dtype=np.float64)
# jb = [3, 4, 5]
# y = np.array([0, 0, 0], dtype=np.float64)
# try:
#     s = dual_simplex_method(A, b, c, y, jb)
#     print s[0], sum(map(lambda q, w: q*w, c, s[0]))
#     print s[1], sum(map(lambda q, w: q*w, b, s[1]))
# except AssertionError as e:
#     print e.message
# print
