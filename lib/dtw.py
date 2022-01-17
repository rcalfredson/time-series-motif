from numba import jit
import numpy as np
from lib.distance import distance

import timeit


@jit(nopython=True)
def set_d_mat_elements(D, d, max_dist, N, M, ncost, mcost):
    for n in range(1, N):
        m_min = max(1, n - max_dist + 1)
        m_max = min(M, n + max_dist + 1)
        for m in range(m_min, m_max):
            D[n, m] = d[n, m] + min(
                min(D[n - 1, m] + mcost, D[n - 1, m - 1]), D[n, m - 1] + ncost,
            )
    return D


class DTW:
    def __init__(self, vec_existing: np.ndarray, vec_new: np.ndarray, max_dist):
        self.vec_existing = vec_existing
        self.vec_new = vec_new
        self.max_dist = max_dist

    def dtw(self):
        time_1 = timeit.default_timer()
        vec_existing, vec_new = [
            np.expand_dims(arr, 0) if arr.ndim == 1 else arr
            for arr in (self.vec_existing, self.vec_new)
        ]
        rows, self.N = vec_existing.shape
        _, self.M = vec_new.shape
        self.d = distance(vec_existing, vec_new) / rows
        self.D = np.ones(self.d.shape) * np.infty
        self.D[0, 0] = self.d[0, 0]
        for n in range(1, self.N):
            self.D[n, 0] = self.d[n, 0] + self.D[n - 1, 0]
        for m in range(1, self.M):
            self.D[0, m] = self.d[0, m] + self.D[0, m - 1]
        time_2 = timeit.default_timer()
        print("upper block:", time_2 - time_1)
        encode_cost = 1
        mcost = encode_cost * np.std(np.hstack(vec_existing), ddof=1) * np.log2(self.M)
        ncost = encode_cost * np.std(np.hstack(vec_new), ddof=1) * np.log2(self.N)
        time_3 = timeit.default_timer()
        print("std dev calc:", time_3 - time_2)
        self.D = set_d_mat_elements(
            self.D, self.d, self.max_dist, self.N, self.M, ncost, mcost
        )
        # np.savetxt("D_orig.csv", self.D)
        time_4 = timeit.default_timer()
        print("assigning to D:", time_4 - time_3)
        Dist = self.D[self.N - 1, self.M - 1]
        n = self.N
        m = self.M
        k = 1
        w = np.zeros((1, 2))
        w[0, :] = [self.N, self.M]
        while n + m != 2:
            if n - 1 == 0:
                m = m - 1
            elif m - 1 == 0:
                n = n - 1
            else:
                number = np.argmin(
                    [self.D[n - 2, m - 1], self.D[n - 1, m - 2], self.D[n - 2, m - 2]]
                )
                if number == 0:
                    n = n - 1
                elif number == 1:
                    m = m - 1
                elif number == 2:
                    n = n - 1
                    m = m - 1
            k = k + 1
            w = np.vstack([w, [[n, m]]])
        time_5 = timeit.default_timer()
        print("lower block:", time_5 - time_4)
        return Dist, self.D, k, w
