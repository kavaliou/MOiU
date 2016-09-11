import numpy as np

with open('test4.in') as f:
        n, m = map(int, f.readline().split())
        A = [map(float, f.readline().split()) for _ in xrange(m)]
        b = map(float, f.readline().split())
        c = map(float, f.readline().split())
        x = map(float, f.readline().split())

Jb = [i for i in xrange(n) if x[i] != 0]
Jn = [i for i in xrange(n) if x[i] == 0]

Ab = np.matrix([[A[i][j] for j in Jb] for i in xrange(m)])
A = np.matrix(A)
B = Ab.I


ok = False
while True:
    # step 1
    c__ = np.matrix([c[j] for j in Jb])
    u = c__ * B
    delta = [0] * n
    for j in Jn:
        delta[j] = (u * A.T[j].T).item(0, 0) - c[j]

    # step 2
    if all(delta[j] >= 0 for j in Jn):
        ok = True
        break

    # step 3
    j0 = delta.index(min(delta))
    z = B * A.T[j0].T
    if all(z.item(i, 0) <= 0 for i in xrange(m)):
        break

    # step 4
    theta0, s = min((x[Jb[i]] / z.item(i, 0), i) for i in xrange(len(Jb)) if z.item(i, 0) > 0)
    js = Jb[s]

    # step 5
    for j in Jn:
        if j != j0:
            x[j] = 0
    x[j0] = theta0
    for i, ji in enumerate(Jb):
        x[ji] -= theta0 * z.item(i, 0)

    Jb[s] = j0
    for i in xrange(len(Jn)):
        if Jn[i] == j0:
            Jn[i] = js
            break

    # step 6
    zs = z.item(s, 0)
    z.itemset((s, 0), -1)
    z /= -zs

    M = np.matrix(np.identity(m))
    for i in xrange(m):
        M.itemset((i, s), z.item(i, 0))
    B = M * B
if ok:
    print x
    print sum(map(lambda q, w: q*w, c, x))
else:
    print 'No solution'
