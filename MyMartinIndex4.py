def delta(x):
    if x == 0:
        return 0
    else:
        return 1


def get_pixel_map(seg):
    m = len(seg)
    n = len(seg[0])
    pixelMap = {}
    vm = 1

    for i in xrange(m):
        for j in xrange(n):
            if seg[i][j] == 0:
                continue
            elif pixelMap.get(seg[i][j]) is None:
                pixelMap.update({seg[i][j]: set([])})
                pixelMap[seg[i][j]].add((i, j))
                vm = max(vm, seg[i][j])
            else:
                pixelMap[seg[i][j]].add((i, j))
    for i in range(1, int(vm + 1)):
        if i in pixelMap:
            continue
        else:
            pixelMap.update(({i: set([])}))
    return pixelMap


def w_vector(m):
    res = [0]
    tmp = 0

    for i in xrange(1, len(m) + 1):
        if i in m:
            res.append(len(m[i]))
            tmp += len(m[i])
        else:
            res.append(0)

    if tmp == 0:
        res = [k * 0 for k in res]
    else:
        res = [k * 1.0 / tmp for k in res]

    return res


def w_matrix(m1, m2):
    mat = [[0 for x in range(len(m2) + 1)] for y in range(len(m1) + 1)]

    for j in xrange(1, len(m1) + 1):
        sum = 0
        temp = [0]
        if j in m1:
            Aj = m1[j]
        else:
            Aj = set([])
        for i in xrange(1, len(m2) + 1):
            if i in m2:
                Bi = m2[i]
            else:
                Bi = set([])
            temp.append(delta(len(Aj.intersection(Bi))) * len(Bi))
            sum += temp[-1]
        if sum == 0:
            mat[j] = [k * 0 for k in temp]
        else:
            mat[j] = [k * 1.0 / sum for k in temp]
    return mat


def p_eval(Ig, Is):
    m1 = get_pixel_map(Ig)
    m2 = get_pixel_map(Is)
    Wj = w_vector(m1)
    Wji = w_matrix(m1, m2)
    res = 0
    for j in xrange(1, len(m1) + 1):
        tmp = 0
        for i in xrange(1, len(m2) + 1):
            s1 = set([])
            s2 = set([])
            if j in m1:
                s1 = m1[j]
            if i in m2:
                s2 = m2[i]
            l1 = len(s1.intersection(s2))
            l2 = len(s1.union(s2))
            if l2 != 0:
                tmp += (l1 * 1.0 / l2 * Wji[j][i])
        res += (1 - tmp) * Wj[j]
    return res




def oce(seg, gt):
    e1 = p_eval(seg, gt)
    e2 = p_eval(gt, seg)
    return min(e1, e2)
