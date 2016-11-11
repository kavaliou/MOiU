def solve(data, n):
    m = len(data[0])
    b_matrix = [[0] * m for _ in range(n)]
    b_matrix[0] = data[0][:]

    x0 = [[0] * m for _ in range(n)]
    x0[0] = range(m)

    for i in range(1, n):
        for y in range(0, m):
            values = [data[i][z] + b_matrix[i-1][y-z] for z in range(0, y+1)]
            max_value = max(values)
            pos = values.index(max_value)
            b_matrix[i][y] = max_value
            x0[i][y] = pos

    return b_matrix, x0


if __name__ == '__main__':
    start_data_from_task = [
        [0, 3, 3, 6, 7, 8, 9, 14],
        [0, 2, 4, 4, 5, 6, 8, 13],
        [0, 1, 1, 2, 3, 3, 10, 11]
    ]

    # start_data_from_task = [
    #     [0, 1, 2, 2, 4, 5, 6],
    #     [0, 2, 3, 5, 7, 7, 8],
    #     [0, 2, 4, 5, 6, 7, 7]
    # ]
    #
    # start_data_from_task = [
    #     [0, 1, 1, 3, 6, 10, 11],
    #     [0, 2, 3, 5, 6, 7, 13],
    #     [0, 1, 4, 4, 7, 8, 9]
    # ]
    # start_data_from_task = [
    #     [0, 1, 2, 3, 4, 5],
    #     [0, 0, 1, 2, 4, 7],
    #     [0, 2, 2, 3, 3, 5]
    # ]

    n, c = 3, 7

    b_matrix, x0 = solve(start_data_from_task, n)

    for i in range(n):
        print ' | '.join(['{:5s}'.format('%s(%s)' % (value, product_count)) for value, product_count in zip(b_matrix[i], x0[i])])

    c_temp = c
    x = [0] * n
    for i in range(n-1, 0, -1):
        x[i] = x0[i][c_temp]
        c_temp -= x0[i][c_temp]
    x[0] = c_temp

    print 'Answer:', b_matrix[n - 1][c]
    print 'x     :', x
