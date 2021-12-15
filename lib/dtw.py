import numpy as np
from lib.distance import distance


def dtw(vec_existing: np.ndarray, vec_new, max_dist):
    rows, N = vec_existing.shape
    _, M = vec_new.shape
    d = distance(vec_existing, vec_new) / rows
    D = np.ones(d.shape) * np.infty
    D[0, 0] = d[0, 0]
    for n in range(1, N):
        D[n, 0] = d[n, 0] + D[n - 1, 0]
    for m in range(1, N):
        D[0, m] = d[0, m] + D[0, m - 1]
    encode_cost = 1
    mcost = encode_cost * np.std(np.hstack(vec_existing), ddof=1) * np.log2(M)
    ncost = encode_cost * np.std(np.hstack(vec_new), ddof=1) * np.log2(N)
    for n in range(2, N + 1):
        pass
        m_min = max(2, n - max_dist)
        m_max = min(M, n + max_dist)
        for m in range(m_min, m_max):
            D[n, m] = d[n, m] + min(
                min(D[n - 1, m] + mcost, D[n - 1, m - 1]), D[n, m - 1] + ncost
            )

    Dist = D[N, M]
    n = N
    m = M
    k = 1
    w = []
    w[0, :] = [N, M]
    while n + m != 2:
        if n - 1 == 0:
            m = m - 1
        elif m - 1 == 0:
            n = n - 1
        else:
            number = np.argmin([D[n - 1, m], D[n, m - 1], D[n - 1, m - 1]])
            if number == 1:
                n = n - 1
            elif number == 2:
                m = m - 1
            elif number == 3:
                n = n - 1
                m = m - 1
    k = k + 1
    w = np.block(w, [n, m])