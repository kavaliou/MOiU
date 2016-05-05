import numpy as np

from fifth import fifth


# A = np.array([
#     [2, -2, 14, 1],
#     [1, -2, 10, 2]], dtype=np.float64)
# D = np.array([
#     [4, 0, 0, 0],
#     [0, 3, 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]], dtype=np.float64)
# b = np.array([2, 0], dtype=np.float64)
# # d = np.array([7, 3, 3], dtype=np.float64)
# x0 = np.array([2, 1, 0, 0], dtype=np.float64)
# J_op = [0, 1]
# J_star = [0, 1]
# c = np.array([0, 0, 0, 0], dtype=np.float64)
#
# answer = fifth(A, None, b, None, x0, J_op, c=c, J_star=J_star, D=D)
# print answer, np.dot(c, answer) + 0.5 * np.dot(np.dot(answer, D), answer)
# print
#
#
# A = np.array([
#     [0, 1, 1, 1, -2, 1],
#     [1, 0, 1, 1, 1, 2],
#     [1, 1, 0, 1, -1, 1]], dtype=np.float64)
# D = np.array([
#     [1, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 1, 2, 1],
#     [0, 0, 0, 2, 4, 2],
#     [0, 0, 0, 1, 2, 1]], dtype=np.float64)
# b = np.array([4, 3, 3], dtype=np.float64)
# # d = np.array([7, 3, 3], dtype=np.float64)
# x0 = np.array([1, 2, 2, 0, 0, 0], dtype=np.float64)
# J_op = [0, 1, 2]
# J_star = [0, 1, 2]
# c = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
#
# answer = fifth(A, None, b, None, x0, J_op, c=c, J_star=J_star, D=D)
# print answer, np.dot(c, answer) + 0.5 * np.dot(np.dot(answer, D), answer)
# print
#
#
# A = np.array([
#     [1,7, -1, 7, -8],
#     [0, 1, 8, 9, 7],
#     [0, 0, 1, 1, 1]], dtype=np.float64)
# D = np.array([
#     [2, 2, 2, 2, 2],
#     [2, 2, 2, 2, 2],
#     [2, 2, 2, 2, 2],
#     [2, 2, 2, 2, 2],
#     [2, 2, 2, 2, 2]], dtype=np.float64)
# b = np.array([14, 18, 2], dtype=np.float64)
# # d = np.array([7, 3, 3], dtype=np.float64)
# x0 = np.array([1, 1, 1, 1, 0], dtype=np.float64)
# J_op = [0, 1, 2]
# J_star = [0, 1, 2, 3]
# c = np.array([7, 0, 0, 9, -9], dtype=np.float64)  # np.dot(d, B) * -1
#
# answer = fifth(A, None, b, None, x0, J_op, c=c, J_star=J_star, D=D)
# print c
# print answer, np.dot(c, answer) + 0.5 * np.dot(np.dot(answer, D), answer)
# print

#
# A = np.array([
#     [1, 2, 0, 1, 0, 4, -1, -3],
#     [1, 3, 0, 0, 1, -1, -1, 2],
#     [1, 4, 1, 0, 0, 2, -2, 0]], dtype=np.float64)
# B = np.array([
#     [1, 1, -1, 0, 3, 4, -2, 1],
#     [2, 6, 0, 0, 1, -5, 0, -1],
#     [-1, 2, 0, 0, -1, 1, 1, 1]], dtype=np.float64)
# b = np.array([4, 5, 6], dtype=np.float64)
# d = np.array([7, 3, 3], dtype=np.float64)
# x0 = np.array([0, 0, 6, 4, 5, 0, 0, 0], dtype=np.float64)
# J_op = [2, 3, 4]
#
# c = np.dot(d, B) * -1
# D = np.dot(B.transpose(), B)
# answer = fifth(A, B, b, d, x0, J_op)
# print answer, np.dot(c.transpose(), answer) + 0.5 * np.dot(np.dot(answer.transpose(), D), answer)
# print
#
#
# # # 1 done
# A = np.array([
#     [11, 0, 0, 1, 0, -4, -1, 1],
#     [1, 1, 0, 0, 1, -1, -1, 1],
#     [1, 1, 1, 0, 1, 2, -2, 1]], dtype=np.float64)
# B = np.array([
#     [1, -1, 0, 3, -1, 5, -2, 1],
#     [2, 5, 0, 0, -1, 4, 0, 0],
#     [-1, 3, 0, 5, 4, -1, -2, 1]], dtype=np.float64)
# b = np.array([8, 5, 2], dtype=np.float64)
# d = np.array([6, 10, 9], dtype=np.float64)
# x0 = np.array([0.7273, 1.2727, 3.0000, 0, 0, 0, 0, 0], dtype=np.float64)
# J_op = [0, 1, 2]
#
# c = np.dot(d, B) * -1
# D = np.dot(B.transpose(), B)
# answer = fifth(A, B, b, d, x0, J_op)
# print answer, np.dot(c.transpose(), answer) + 0.5 * np.dot(np.dot(answer.transpose(), D), answer)
# print
# #
# # 2 near
# A = np.array([
#     [2, -3, 1, 1, 3, 0, 1, 2],
#     [-1, 3, 1, 0, 1, 4, 5, -6],
#     [1, 1, -1, 0, 1, -2, 4, 8]], dtype=np.float64)
# B = np.array([
#     [1, 0, 0, 3, -1, 5, 0, 1],
#     [2, 5, 0, 0, 0, 4, 0, 0],
#     [-1, 9, 0, 5, 2, -1, -1, 5]], dtype=np.float64)
# b = np.array([8, 4, 14], dtype=np.float64)
# d = np.array([6, 10, 9], dtype=np.float64)
# x0 = np.array([0, 2, 0, 0, 4, 0, 0, 1], dtype=np.float64)
# J_op = [1, 4, 7]
#
# c = np.array([-13, -217, 0, -117, -27, -71, 18, -99], dtype=np.float64)
# D = np.dot(B.transpose(), B)
# answer = fifth(A, B, b, d, x0, J_op, c=c)
# print answer, np.dot(c.transpose(), answer) + 0.5 * np.dot(np.dot(answer.transpose(), D), answer)
# print
#
# # 3 no
# A = np.array([
#     [2, -3, 1, 1, 3, 0, 1, 2],
#     [-1, 3, 1, 0, 1, 4, 5, -6],
#     [1, 1, -1, 0, 1, -2, 4, 8]], dtype=np.float64)
# D = np.array([
#     [1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1],], dtype=np.float64)
# b = np.array([6, 4, 14], dtype=np.float64)
# d = np.array([6, 10, 9], dtype=np.float64)
# x0 = np.array([0, 2, 0, 0, 4, 0, 0, 1], dtype=np.float64)
# J_op = [1, 4, 7]
#
# c = np.array([1, 3, -1, 3, 5, 2, -2, 0], dtype=np.float64)
# answer = fifth(A, None, b, d, x0, J_op, c=c, D=D)
# print answer, np.dot(c.transpose(), answer) + 0.5 * np.dot(np.dot(answer.transpose(), D), answer)
# print
#
# 4 DONE
A = np.array([
    [0, 2, 1, 4, 3, 0, -5, -10],
    [-1, 1, 1, 0, 1, 1, -1, -1],
    [1, 1, 1, 0, 1, -2, -5, 8]], dtype=np.float64)
D = np.array([
    [25, 10, 0, 3, -1, 13, 0, 1],
    [10, 45, 0, 0, 0, 20, 0, 0],
    [0, 0, 20, 0, 0, 0, 0, 0],
    [3, 0, 0, 29, -3, 15, 0, 3],
    [-1, 0, 0, -3, 21, -5, 0, -1],
    [13, 20, 0, 15, -5, 61, 0, 5],
    [0, 0, 0, 0, 0, 0, 20, 0],
    [1, 0, 0, 3, -1, 5, 0, 21]], dtype=np.float64)
b = np.array([20, 1, 7], dtype=np.float64)
d = np.array([6, 10, 9], dtype=np.float64)
x0 = np.array([3, 0, 0, 2, 4, 0, 0, 0], dtype=np.float64)
J_op = [0, 3, 4]

c = np.array([1, -3, 4, 3, 5, 6, -2, 0], dtype=np.float64)
answer = fifth(A, None, b, d, x0, J_op, c=c, D=D)
print answer, np.dot(c.transpose(), answer) + 0.5 * np.dot(np.dot(answer.transpose(), D), answer)
print
#
# # 5
# A = np.array([
#     [0, 0, 1, 5, 2, 0, -5, -4],
#     [1, 1, -1, 0, 1, -1, -1, -1],
#     [1, 1, 1, 0, 1, 2, 5, 8]], dtype=np.float64)
# D = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)
# b = np.array([15, -1, 9], dtype=np.float64)
# d = np.array([6, 10, 9], dtype=np.float64)
# x0 = np.array([4, 0, 5, 2, 0, 0, 0, 0], dtype=np.float64)
# J_op = [0, 2, 3]
#
# c = np.array([1, -3, 4, 3, 5, 6, -2, 0], dtype=np.float64)
# answer = fifth(A, None, b, d, x0, J_op, c=c, D=D)
# print answer, np.dot(c.transpose(), answer) + 0.5 * np.dot(np.dot(answer.transpose(), D), answer)
# print
