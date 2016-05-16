import sympy
import numpy as np


with open('test0.in', 'r') as file_:
    n = int(file_.readline())
    x = [sympy.Symbol('x{}'.format(i), real=True) for i in xrange(n)]

    f = eval(file_.readline())
    m = int(file_.readline())

    g = [eval(file_.readline()) for _ in xrange(m)]
    lam = [sympy.Symbol('lam{}'.format(i)) for i in xrange(m)]
    x_val = map(float, file_.readline().split())

# indexes = [i for i in xrange(m) if g[i].subs(dict(zip(x, x_val))) == 0.0]
# indexes = []
# if not indexes:
#     indexes = range(m)

dfdx = [sympy.diff(f, x[i]).subs(dict(zip(x, x_val))) for i in xrange(n)]
dgdx = [
    [sympy.diff(g[i], x[j]).subs(dict(zip(x, x_val))) for i in xrange(m)]
    for j in xrange(n)
]

# system = []
# for i in xrange(n):
#     func = dfdx[i]
#     for j in indexes:
#         func += lam[j] * dgdx[i][j]
#     system.append(func)

system = [dfdx[i] + sum(map(lambda a, b: a * b, lam, dgdx[i])) for i in xrange(n)]
lams = sympy.solve(system)

ok = True
if isinstance(lams, dict):
    try:
        ok = all(val >= 0 for (var, val) in lams.iteritems() if str(var).startswith('lam'))
    except:
        ok = True
    lams = [lams]

if not lams or not ok:
    print 'no solution for system'
    print 'not optimal'
    exit()


L = f + sum(map(lambda a, b: a*b, lam, g))
for solution in lams:
    if x_val:
        xx = zip(x, x_val)
    else:
        xx = filter(lambda (var, val): str(var).startswith('x'), solution.iteritems())
    if any(g[i].subs(dict(xx)) > 0 for i in xrange(m)):
        print 'not solution', xx
        continue
    H = [
        [float(sympy.diff(L, i, j).subs(dict(zip(x, x_val) + solution.items()))) for j in x]
        for i in x
    ]
    H = np.matrix(H)
    if all(np.linalg.det(H[0:i, 0:i]) >= 0.0 for i in xrange(1, n+1)):
        print 'local minimum',
    elif all(np.linalg.det(H[0:i, 0:i]) <= 0.0 for i in xrange(1, n+1)):
        print 'local maximum',
    else:
        print 'not optimal',
    print xx