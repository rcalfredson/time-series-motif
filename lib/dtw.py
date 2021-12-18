import numpy as np
from lib.distance import distance


def dtw(vec_existing: np.ndarray, vec_new: np.ndarray, max_dist):
    vec_existing, vec_new = [
        np.expand_dims(arr, 0) if arr.ndim == 1 else arr
        for arr in (vec_existing, vec_new)
    ]
    rows, N = vec_existing.shape
    _, M = vec_new.shape
    d = distance(vec_existing, vec_new) / rows
    D = np.ones(d.shape) * np.infty
    D[0, 0] = d[0, 0]
    for n in range(1, N):
        D[n, 0] = d[n, 0] + D[n - 1, 0]
    for m in range(1, M):
        D[0, m] = d[0, m] + D[0, m - 1]

    encode_cost = 1
    mcost = encode_cost * np.std(np.hstack(vec_existing), ddof=1) * np.log2(M)
    ncost = encode_cost * np.std(np.hstack(vec_new), ddof=1) * np.log2(N)
    for n in range(1, N):
        m_min = max(1, n - max_dist + 1)
        m_max = min(M, n + max_dist + 1)
        for m in range(m_min, m_max):
            D[n, m] = d[n, m] + min(
                min(D[n - 1, m] + mcost, D[n - 1, m - 1]), D[n, m - 1] + ncost
            )

    Dist = D[N - 1, M - 1]
    n = N
    m = M
    k = 1
    w = np.zeros((1, 2))
    w[0, :] = [N, M]
    while n + m != 2:
        if n - 1 == 0:
            m = m - 1
        elif m - 1 == 0:
            n = n - 1
        else:
            number = np.argmin([D[n - 2, m - 1], D[n - 1, m - 2], D[n - 2, m - 2]])
            if number == 0:
                n = n - 1
            elif number == 1:
                m = m - 1
            elif number == 2:
                n = n - 1
                m = m - 1
        k = k + 1
        w = np.vstack([w, [[n, m]]])
    return Dist, D, k, w
