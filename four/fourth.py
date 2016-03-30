def transport_task(a, b, c):
    def matrix_min(arr, check_matrix=None):
        _n, _m = len(arr), len(arr[0])
        if check_matrix is None:
            check_matrix = [[1] * _m for _ in range(_n)]
        min_value, _i, _j = None, 0, 0
        for i_ in xrange(_n):
            for j_ in xrange(_m):
                if (arr[i_][j_] < min_value or min_value is None) and check_matrix[i_][j_]:
                    min_value = arr[i_][j_]
                    _i = i_
                    _j = j_
        return min_value, (_i, _j)

    def has_cycles(matrix):
        def count():
            c = []
            [[c.append(j) for j in l if j] for l in matrix]
            return len(c)

        previous_count, current_count = None, count()
        while previous_count is None or previous_count != current_count:
            for i in range(len(matrix)):
                if sum(matrix[i]) == 1:
                    matrix[i] = [0] * len(matrix[i])
            for j in range(len(matrix[0])):
                if sum([matrix[i][j] for i in range(len(matrix))]) == 1:
                    for i in range(len(matrix)):
                        matrix[i][j] = 0
            previous_count, current_count = current_count, count()
        return current_count

    def calculate_potentials(u_dict):
        u_i, v_j = [[0, 0] for _ in range(n)], [[0, 0] for _ in range(m)]
        u_i[0][0] = 1
        previous_step_calculated = [('u', 0)]
        while not all([flag for flag, _ in u_i + v_j]):
            new_previous_step_calculated = []
            for flag, num in previous_step_calculated:
                if flag == 'u':
                    value = u_i[num][1]
                    items = {item[1]: c[item[0]][item[1]] for item in u_dict if item[0] == num}
                    new_flag = 'v'
                    array = v_j
                else:
                    value = v_j[num][1]
                    items = {item[0]: c[item[0]][item[1]] for item in u_dict if item[1] == num}
                    new_flag = 'u'
                    array = u_i

                for index, item_value in items.iteritems():
                    new_previous_step_calculated.append((new_flag, index))
                    array[index][0], array[index][1] = 1, item_value - value

            previous_step_calculated = new_previous_step_calculated
        return [i[1] for i in u_i], [j[1] for j in v_j]

    def find_cycle(point, u_dict):
        final_array = []

        def recursive_find(start_point, direction):
            def cond(direct):
                return 0 if direct else 1

            array = [_p for _p in u_dict.keys() + [point] if _p[cond(direction)] == start_point[cond(direction)]
                     and _p[cond(not direction)] != start_point[cond(not direction)]]
            for current_item in array:
                final_array.append(current_item)
                if current_item == point:
                    return True
                if recursive_find(current_item, not direction):
                    return True
                final_array.pop(-1)
            return False

        for it in [p for p in u_dict if p[0] == point[0] and p[1] != point[1]]:
            final_array.append(it)
            if recursive_find(it, False):
                return final_array
            final_array.pop(-1)

    diff = sum(a) - sum(b)
    if diff > 0:
        b.append(diff)
        for line in c:
            line.append(0)
    elif diff < 0:
        a.append(-diff)
        c.append([0] * len(c[0]))

    n, m = len(a), len(b)
    matrix_of_used_places = [[1] * m for _ in range(n)]
    u = {}

    while max([max(x) for x in matrix_of_used_places]):
        _, (i, j) = matrix_min(c, matrix_of_used_places)
        diff = a[i] - b[j]
        u[(i, j)] = b[j] if diff >= 0 else a[i]
        if diff >= 0:
            a[i] = diff
            b[j] = 0
            for line in matrix_of_used_places:
                line[j] = 0
        else:
            a[i] = 0
            b[j] = -diff
            matrix_of_used_places[i] = [0] * m

    while len(u) < n + m - 1:
        _u_ = [(i, j) for i in range(n) for j in range(m) if (i, j) not in u]
        for u_item in _u_:
            u[u_item] = 0
            if not has_cycles([[1 if (i, j) in u else 0 for j in range(m)] for i in range(n)]):
                break
            del u[u_item]

    while True:
        ui, vj = calculate_potentials(u)
        delta_ij = {}
        indexes = []
        [[indexes.append((i, j)) for j in range(m) if (i, j) not in u] for i in range(n)]
        for i, j in indexes:
            delta_ij[(i, j)] = c[i][j] - ui[i] - vj[j]
        value, item = min([(value, item) for item, value in delta_ij.iteritems()])
        if value >= 0:  # WIN
            break

        cycle = find_cycle(item, u)

        min_value, item_min_value = min([(u[it], it) for num, it in enumerate(cycle) if num % 2 == 0])
        u[item] = 0
        for num, point in enumerate(cycle):
            if num % 2 == 0:
                u[point] -= min_value
            else:
                u[point] += min_value
        del u[item_min_value]

    return u
