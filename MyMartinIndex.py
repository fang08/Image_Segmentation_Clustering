def delta(x):
    if x == 0:
        return 0
    else:
        return 1


def get_pixel_map(seg):
    m = len(seg)
    n = len(seg[0])
    pixelMap = {}

    for i in xrange(m):
        for j in xrange(n):
            if pixelMap.get(seg[i][j]) is None:
                pixelMap.update({seg[i][j]: set([])})
                pixelMap[seg[i][j]].add((i, j))
            else:
                pixelMap[seg[i][j]].add((i, j))
    return pixelMap


def w_vector(m):
    res = [0]
    tmp = 0

    for i in xrange(1, len(m) + 1):
        res.append(len(m[i]))
        tmp += len(m[i])

    res = [k * 1.0 / tmp for k in res]

    return res


def w_matrix(m1, m2):
    mat = [[0 for x in range(len(m2) + 1)] for y in range(len(m1) + 1)]

    for j in xrange(1, len(m1) + 1):
        sum = 0
        temp = [0]
        Aj = m1[j]
        for i in xrange(1, len(m2) + 1):
            Bi = m2[i]
            temp.append(delta(len(Aj.intersection(Bi))) * len(Bi))
            sum += temp[-1]
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
            l1 = len(m1[j].intersection(m2[i]))
            l2 = len(m1[j].union(m2[i]))
            tmp = l1 * 1.0 / l2 * Wji[j][i]
        res += (1 - tmp) * Wji[j][i]
    return res


def relabel(seg):
    m = len(seg)
    n = len(seg[0])
    lmap = {}
    label_count = 1
    for i in xrange(m):
        for j in xrange(n):
            if lmap.get(seg[i][j]) is None:
                lmap.update({seg[i][j]: label_count})
                label_count += 1
    for i in xrange(m):
        for j in xrange(n):
            seg[i][j] = lmap.get(seg[i][j])

    return seg


def oce(seg, gt):
    seg = relabel(seg)
    gt = relabel(gt)
    e1 = p_eval(seg, gt)
    e2 = p_eval(gt, seg)
    return min(e1, e2)
