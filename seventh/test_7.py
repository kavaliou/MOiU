import sympy

from seventh import sev

tests = [
    'prim1.in',
    'prim2.in',
    'task1.in',
    'task2.in',
    'task3.in',
    'task4.in',
    'task5.in'
]

for test in tests:
    with open('tests/%s' % test, 'r') as file_:
        n = int(file_.readline())
        x = [sympy.Symbol('x{}'.format(i), real=True) for i in xrange(n)]

        f = eval(file_.readline())
        m = int(file_.readline())

        g = [eval(file_.readline()) for _ in xrange(m)]
        lam = [sympy.Symbol('lam{}'.format(i)) for i in xrange(m)]
        x_val = map(float, file_.readline().split())

    sev(n, m, x, x_val, f, g, lam)
