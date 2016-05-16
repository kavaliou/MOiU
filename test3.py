import numpy as np

from second import simplex_method
from third import dual_simplex_method

# from lesson
A = np.array([1, -1, 3, -2, 1, -5, 11, -6], dtype=np.float).reshape((2, 4))
s = dual_simplex_method(A, np.array([1, 9]), np.array([1, 1, -2, -3]),
                        np.array([1.5, -0.5], dtype=np.float), [0, 1])
print s[0], sum(map(lambda q, w: q*w, [1, 1, -2, -3], s[0]))
print

# Max Tishka #1
A = np.array([-3, -1, 1, 0, 0,
              -4, -3, 0, 1, 0,
              1, 2, 0, 0, 1], dtype=np.float).reshape((3, 5))
try:
    s = dual_simplex_method(A, np.array([-3, -6, 3]), np.array([2, 4, 0, 0, 0]),
                        np.array([-1, -2, 1], dtype=np.float), [2, 3, 4])
    print s
    s = simplex_method(A, np.array([-3, -6, 3]), np.array([2, 4, 0, 0, 0]), s[0], s[1])
    print s, sum(map(lambda q, w: q*w, [2, 4, 0, 0, 0], s))
except AssertionError as e:
    print e.message
print

def shevch():
    # Anton Shevchenya #1 ++++++++++++++++++++++++++
    c = np.array([-1, 2, 0, 0, 0], dtype=np.float64)
    b = np.array([-2, 4, -4], dtype=np.float64)
    A = np.array([
            [2, -1, 1, 0, 0],
            [1, 2, 0, 1, 0],
            [-1, -4, 0, 0, 1]], dtype=np.float64)
    jb = [2, 3, 4]
    try:
        s, _, __ = dual_simplex_method(A, b, c, b, jb)
        # s = simplex_method(A, b, c, s[0], s[1])
        print s, sum(map(lambda q, w: q*w, c, s))
    except AssertionError as e:
        print e.message
    print

    # #2 ++++++++++++++++++++
    c = np.array([-6, -3, 0, 0], dtype=np.float64)
    b = np.array([-1, -2], dtype=np.float64)
    A = np.array([
            [3, -1, 1, 0],
            [-2, 3, 0, 1]], dtype=np.float64)
    jb = [2, 3]
    try:
        s, _, __ = dual_simplex_method(A, b, c, b, jb)
        # s = simplex_method(A, b, c, s[0], s[1])
        print s, sum(map(lambda q, w: q*w, c, s))
    except AssertionError as e:
        print e.message
    print

    # #3  ????????????
    c = np.array([-4, 8, -16, -7, 0, 0, 0], dtype=np.float64)
    b = np.array([0, 5, -2], dtype=np.float64)
    A = np.array([
            [0, 0, -2, 1, 1, 0, 0],
            [2, 0, -1, 2, 0, 1, 0],
            [-2, 2, -2, -2, 0, 0, 1]], dtype=np.float64)
    jb = [4, 5, 6]
    y = np.array([0, 5, -2], dtype=np.float64)
    try:
        s, _, __ = dual_simplex_method(A, b, c, y, jb)
        # s = simplex_method(A, b, c, s[0], s[1])
        print s, sum(map(lambda q, w: q*w, c, s))
    except AssertionError as e:
        print e.message
    print

    # #4 ++-++-++-++-++-
    c = np.array([1, 1, 2, 0, 0], dtype=np.float64)
    b = np.array([8, -4, -6], dtype=np.float64)
    A = np.array([
            [1, 1, 1, 0, 0],
            [-1, 1, 0, 1, 0],
            [-1, -2, 0, 0, 1]], dtype=np.float64)
    jb = [1, 3, 4]
    try:
        s, _, __ = dual_simplex_method(A, b, c, b, jb)
        # s = simplex_method(A, b, c, s[0], s[1])
        print s, sum(map(lambda q, w: q*w, c, s))
    except AssertionError as e:
        print e.message
    print

    # #5 ??????????????????????
    c = np.array([-3, -2, -1, 0, 0, 0], dtype=np.float64)
    b = np.array([-4, -6, -2], dtype=np.float64)
    A = np.array([
            [0, -1, -1, 1, 0, 0],
            [-2, -1, -2, 0, 1, 0],
            [-2, 1, -2, 0, 0, 1]], dtype=np.float64)
    jb = [3, 4, 5]
    try:
        s, _, __ = dual_simplex_method(A, b, c, b, jb)
        # s = simplex_method(A, b, c, s[0], s[1])
        print s, sum(map(lambda q, w: q*w, c, s))
    except AssertionError as e:
        print e.message
    print

    # #6 +-+-+-+-+-+-+-+-
    c = np.array([0, -6, 6, 0, 0, 0, 0], dtype=np.float64)
    b = np.array([-1, 0, 4, 1], dtype=np.float64)
    A = np.array([
            [1, -2, 1, 1, 0, 0, 0],
            [-1, 2, -2, 0, 1, 0, 0],
            [2, 2, 1, 0, 0, 1, 0],
            [0, -1, 2, 0, 0, 0, 1]], dtype=np.float64)
    jb = [3, 4, 5, 6]
    try:
        s, _, __ = dual_simplex_method(A, b, c, b, jb)
        # s = simplex_method(A, b, c, s[0], s[1])
        print s, sum(map(lambda q, w: q*w, c, s))
    except AssertionError as e:
        print e.message
    print

shevch()

def from_metoda():
    # metoda #1 +++++++++++++++++
    c = np.array([5, 2, 3, -16, 1, 3, -3, -12], dtype=np.float64)
    b = np.array([-2, -4, -2], dtype=np.float64)
    A = np.array([
            [-2, -1, 1, -7, 0, 0, 0, 2],
            [4, 2, 1, 0, 1, 5, -1, -5],
            [1, 1, 0, -1, 0, 3, -1, 1]], dtype=np.float64)
    jb = [0, 1, 2]
    y = np.array([1, 2, -1], dtype=np.float64)
    try:
        s, _, __ = dual_simplex_method(A, b, c, y, jb)
        # s = simplex_method(A, b, c, s[0], s[1])
        print s, sum(map(lambda q, w: q*w, c, s))
    except AssertionError as e:
        print e.message
    print

    # metoda #2 +++++++++++++++++
    c = np.array([-12, 2, 2, -6, 10, -1, -9, 8], dtype=np.float64)
    b = np.array([-2, 4, -2], dtype=np.float64)
    A = np.array([
            [-2, -1, 1, -7, 1, 0, 0, 2],
            [-4, 2, 1, 0, 5, 1, -1, 5],
            [1, 1, 0, -1, 0, 3, -1, 1]], dtype=np.float64)
    jb = [1, 3, 5]
    y = np.array([1, 2, -1], dtype=np.float64)
    try:
        s, _, __ = dual_simplex_method(A, b, c, y, jb)
        # s = simplex_method(A, b, c, s[0], s[1])
        print s, sum(map(lambda q, w: q*w, c, s))
    except AssertionError as e:
        print e.message
    print

    # metoda #3 +++++++++++++++++
    c = np.array([12, -2, -6, 20, -18, -5, -7, -20], dtype=np.float64)
    b = np.array([-2, 8, -2], dtype=np.float64)
    A = np.array([
            [-2, -1, 1, -7, 1, 0, 0, 2],
            [-4, 2, 1, 0, 5, 1, -1, 5],
            [1, 1, 0, 1, 4, 3, 1, 1]], dtype=np.float64)
    jb = [1, 3, 5]
    y = np.array([-3, -2, -1], dtype=np.float64)
    try:
        s, _, __ = dual_simplex_method(A, b, c, y, jb)
        # s = simplex_method(A, b, c, s[0], s[1])
        print s, sum(map(lambda q, w: q*w, c, s))
    except AssertionError as e:
        print e.message
    print

    # metoda #4 ++++++++++++++
    c = np.array([10, -2, -38, 16, -9, -9, -5, -7], dtype=np.float64)
    b = np.array([-2, -5, 2], dtype=np.float64)
    A = np.array([
            [-2, -1, 10, -7, 1, 0, 0, 2],
            [-4, 2, 3, 0, 5, 1, -1, 0],
            [1, 1, 0, 1, -4, 3, -1, 1]], dtype=np.float64)
    jb = [1, 7, 4]
    y = np.array([-3, -2, -1], dtype=np.float64)
    try:
        s, _, __ = dual_simplex_method(A, b, c, y, jb)
        # s = simplex_method(A, b, c, s[0], s[1])
        print s, sum(map(lambda q, w: q*w, c, s))
    except AssertionError as e:
        print e.message
    print

    # metoda #5 +++++++++++++++++++
    c = np.array([36, -12, 66, 76, -5, 77, -76, -7], dtype=np.float64)
    b = np.array([2, 5, -2], dtype=np.float64)
    A = np.array([
            [3, -1, 10, -7, 1, 0, 0, 2],
            [7, -2, 14, 8, 0, 12, -11, 0],
            [1, 1, 0, 1, -4, 3, -1, 1]], dtype=np.float64)
    jb = [6, 7, 3]
    y = np.array([-3, 7, -1], dtype=np.float64)
    try:
        s, _, __ = dual_simplex_method(A, b, c, y, jb)
        # s = simplex_method(A, b, c, s[0], s[1])
        print s, sum(map(lambda q, w: q*w, c, s))
    except AssertionError as e:
        print e.message
    print

from_metoda()