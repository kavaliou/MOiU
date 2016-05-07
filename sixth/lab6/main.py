from random import random

import numpy as np
from scipy.optimize import linprog


OYEEEE_CONSTANT = 199
OYEEEE_MULTIPLIER = 10


# with open('test.in', 'r') as f:
#     n = int(f.readline())
#     c = map(float, f.readline().split())
#     d = [map(float, f.readline().split()) for _ in xrange(len(c))]
#     m = int(f.readline())
#     cg = []
#     dg = []
#     alpha = []
#     for i in xrange(m):
#         cg.append(map(float, f.readline().split()))
#         dg.append([map(float, f.readline().split()) for _ in xrange(n)])
#         alpha.append(float(f.readline()))
#     x_line = map(float, f.readline().split())
#     x_wave = map(float, f.readline().split())

with open('testk.in', 'r') as f:
    n = int(f.readline())
    B = np.matrix([map(float, f.readline().split()) for _ in xrange(3)])
    d = B.T * B
    c = map(float, f.readline().split())

    m = int(f.readline())
    cg = []
    dg = []
    alpha = []
    for i in xrange(m):
        B = np.matrix([map(float, f.readline().split()) for _ in xrange(3)])
        dg.append(B.T * B)
        cg.append(map(float, f.readline().split()))
        alpha.append(float(f.readline()))
    x_line = map(float, f.readline().split())
    x_wave = map(float, f.readline().split())

c = np.matrix(c).T
# d = np.matrix(d)
cg = [np.matrix(x).T for x in cg]
# dg = [np.matrix(x) for x in dg]
x_line = np.matrix(x_line).T
x_wave = np.matrix(x_wave).T

dfdx = c + d * x_wave
dgdx = [cc + dd * x_wave for cc, dd in zip(cg, dg)]

active = []
nonactive = []
for i in xrange(m):
    # if abs((cg[i].T * x_wave + x_wave.T * dg[i] * x_wave / 2 + alpha[i])[0, 0]) < 0.0001:
    if abs((cg[i].T * x_wave + x_wave.T * dg[i] * x_wave / 2 + alpha[i])[0, 0]) == 0.0:
        active.append(i)
    else:
        nonactive.append(i)

c_fun = [dfdx[i, 0] for i in xrange(n)]
A = [[dgdx[index][i, 0] for i in xrange(n)] for index in active]
b = [0] * len(active)

mn = [0 if x_wave[i, 0] == 0 else -1 for i in xrange(n)]
mx = [1] * 8
e_asterisk = np.matrix(linprog(c=c_fun, A_ub=A, b_ub=b, bounds=zip(mn, mx)).x)

a = (dfdx.T * e_asterisk.T)[0, 0]
if a == 0.0:
    print 'optimal'
    exit()
b = ((x_line - x_wave).T * dfdx)[0, 0]
print a
if b > 0:
    coefs = [- 2*a / ( b)]
    coefs.extend(- a / ((random() * OYEEEE_MULTIPLIER + 1 + 1e-9) * b) for _ in xrange(OYEEEE_CONSTANT))
else:
    coefs = [1]
    coefs.extend(random() * OYEEEE_MULTIPLIER + 1e-9 for _ in xrange(OYEEEE_CONSTANT))

ts = [0.5]
ts.extend([random() for _ in xrange(OYEEEE_CONSTANT)])

f_start = c.T * x_wave + x_wave.T * d * x_wave / 2
for t in ts:
    for coef in coefs:
        x_new = x_wave + (t * e_asterisk).T + coef * t * (x_line - x_wave)
        if any(x_new[i, 0] < 0 for i in xrange(n)):
            continue
        f_new = c.T * x_new + x_new.T * d * x_new / 2
        if f_new < f_start:
            for i in xrange(m):
                AAAtemp = cg[i].T * x_new + x_new.T * dg[i] * x_new / 2 + alpha[i]
                if (cg[i].T * x_new + x_new.T * dg[i] * x_new / 2 + alpha[i])[0, 0] > 0:
                    # print cg[i].T * x_new + x_new.T * dg[i] * x_new / 2 + alpha[i]
                    break
            else:
                print 'f start:', f_start[0, 0]
                print 'f new:', f_new[0, 0]
                print x_new
                exit()

print ':('
