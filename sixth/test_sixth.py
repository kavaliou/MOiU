import numpy as np

from sixth import sixth


c = np.array([-1, -1, -1, -1, -2, 0, -2, -3], dtype=np.float64)
B = np.array([
    [2, 1, 0, 4, 0, 3, 0, 0],
    [0, 4, 0, 3, 1, 1, 3, 2],
    [1, 3, 0, 5, 0, 4, 0, 4]
], dtype=np.float64)
D = None

Bi = [
    np.array([
        [0, 0, 0.5, 2.5, 1, 0, -2.5, -2],
        [0.5, 0.5, -0.5, 0, 0.5, -0.5, -0.5, -0.5],
        [0.5, 0.5, 0.5, 0, 0.5, 1, 2.5, 4]
    ], dtype=np.float64),
    np.array([
        [1.0, 2.0, -1.5, 3.0, -2.5, 0, -1, -0.5],
        [-1.5, -0.5, -1, 2.50, 3.5, 3, -1.5, -0.5],
        [1.5, 2.5, 1, 1.0, 2.5, 1.5, 3, 0],
    ], dtype=np.float64),
    np.array([
        [0.75, 0.5, -1.0, 0.25, 0.25, 0, 0.25, 0.75],
        [-1.0, 1.0, 1.0, 0.75, 0.75, 0.5, 1, -0.75],
        [0.5, -0.25, 0.5, 0.75, 0.5, 1.250, -0.75, -0.25]
    ], dtype=np.float64),
    np.array([
        [1.50, -1.50, -1.50, 2.0, 1.50, 0, 0.5, -1.50],
        [-0.5, -2.50, -0.5, -1.0, -2.50, 2.50, 1.0, 2.0],
        [-2.50, 1.0, -2.0, -1.50, -2.50, 0.5, 2.50, -2.50]
    ], dtype=np.float64),
    np.array([
        [1.0, 0.25, -0.5, 1.250, 1.250, -0.5, 0.25, -0.75],
        [-1.0, -0.75, -0.75, 0.5, -0.25, 1.250, 0.25, -0.5],
        [0, 0.75, 0.5, -0.5, -1.0, 1.0, -1.0, 1.0]
    ], dtype=np.float64)
]
ci = [
    np.array([0, 60, 80, 0, 0, 0, 40, 0], dtype=np.float64),
    np.array([2, 0, 3, 0, 2, 0, 3, 0], dtype=np.float64),
    np.array([0, 0, 80, 0, 0, 0, 0, 0], dtype=np.float64),
    np.array([0, -2, 1, 2, 0, 0, -2, 1], dtype=np.float64),
    np.array([-4, -2, 6, 0, 4, -2, 60, 2], dtype=np.float64)
]
alpha = np.array([-51.75, -436.75, -33.7813, -303.375, -41.75], np.float64)
x_star = [1, 0, 0, 2, 4, 2, 0, 0]
J_star = [1, 2, 6, 7]

if D is None:
    D = np.dot(B.transpose(), B)
print 'start', np.dot(c, x_star) + np.dot(np.dot(x_star, D), x_star) / 2
answer = sixth(c, B, None, ci, Bi, None, alpha, x_star, J_star)
print 'end', np.dot(c, answer) + np.dot(np.dot(answer, D), answer) / 2
print


c = np.array([-1, -1, -1, -1, -2, 0, -2, -3], dtype=np.float64)
B = np.array([
    [2, 1, 0, 4, 0, 3, 0, 0],
    [0, 4, 0, 3, 1, 1, 3, 2],
    [1, 3, 0, 5, 0, 4, 0, 4]
], dtype=np.float64)
D = None

Bi = [
    np.array([
        [0, 0, 0.5, 2.5, 1, 0, -2.5, -2],
        [0.5, 0.5, -0.5, 0, 0.5, -0.5, -0.5, -0.5],
        [0.5, 0.5, 0.5, 0, 0.5, 1, 2.5, 4]
    ], dtype=np.float64),
    np.array([
        [1.0, 2.0, -1.5, 3.0, -2.5, 0, -1, -0.5],
        [-1.5, -0.5, -1, -2.50, 3.5, -3, -1.5, -0.5],
        [1.5, 2.5, -1, 1.0, 2.5, 1.5, 3, 0],
    ], dtype=np.float64),
    np.array([
        [0.75, 0.5, -1.0, 0.25, 0.25, 0, 0.25, 0.75],
        [-1.0, 1.0, 4.0, 0.75, 0.75, 0.5, 7, -0.75],
        [0.5, -0.25, 0.5, 0.75, 0.5, 1.250, -0.75, -0.25]
    ], dtype=np.float64),
    np.array([
        [1.50, -1.50, -1.50, 2.0, 1.50, 0, 0.5, -1.50],
        [-0.5, -2.50, -0.5, -5.0, -2.50, 3.50, 1.0, 2.0],
        [-2.50, 1.0, -2.0, -1.50, -2.50, 0.5, 8.50, -2.50]
    ], dtype=np.float64),
    np.array([
        [1.0, 0.25, -0.5, 1.250, 1.250, -0.5, 0.25, -0.75],
        [-1.0, -0.75, -0.75, 0.5, -0.25, 1.250, 0.25, -0.5],
        [0, 0.75, 0.5, -0.5, -1.0, 1.0, -1.0, 1.0]
    ], dtype=np.float64)
]
ci = [
    np.array([0, 60, 80, 0, 0, 0, 40, 0], dtype=np.float64),
    np.array([2, 0, 3, 0, 2, 0, 3, 0], dtype=np.float64),
    np.array([0, 0, 80, 0, 0, 0, 0, 0], dtype=np.float64),
    np.array([0, -2, 1, 2, 0, 0, -2, 1], dtype=np.float64),
    np.array([-4, -2, 6, 0, 4, -2, 60, 2], dtype=np.float64)
]
alpha = np.array([-687.125, -666.625, -349.5938, -254.625, -45.1563], np.float64)
x_star = [0, 8, 2, 1, 0, 4, 0, 0]
J_star = [1, 2, 6, 7]

if D is None:
    D = np.dot(B.transpose(), B)
print 'start', np.dot(c, x_star) + np.dot(np.dot(x_star, D), x_star) / 2
answer = sixth(c, B, None, ci, Bi, None, alpha, x_star, J_star)
print 'end', np.dot(c, answer) + np.dot(np.dot(answer, D), answer) / 2
print
