from time import time
import numpy as np

from lib.dtw import DTW

import timeit


def new_segment_size(x, cur, models, s_min, s_max, max_dist):
    num_models = len(models)
    avg_costs = np.ones((s_max, num_models + 1, s_max)) * np.infty
    for s in range(s_min, s_max + 1):
        print('s:', s)
        if cur + s + 1 >= x.shape[1]:
            continue
        for k in range(num_models + 2):
            # start_t = timeit.default_timer()
            if k + 1 <= num_models:
                cur_model = models[k]
            else:
                cur_model = x[:, cur : cur + s]
            # time_1 = timeit.default_timer()
            # print('upper block:', time_1 - start_t)
            x_cur = x[:, cur + s : min(x.shape[1], cur + s + s_max - 1) + 1]
            time_2 = timeit.default_timer()
            # print('finding min:', time_2 - time_1)
            _, dtw_mat, _, _ = DTW(cur_model, x_cur, max_dist).dtw()
            time_3 = timeit.default_timer()
            print('calling dtw:', time_3 - time_2)
            dtw_costs = dtw_mat[-1, :]
            avg_costs[s - 1, k - 1, 0 : x_cur.shape[1]] = dtw_costs / np.arange(
                1, x_cur.shape[1] + 1
            )
            avg_costs[s - 1, k - 1, 0:s_min] = np.inf
            # print('setting values of avg costs:', timeit.default_timer() - time_3)
    best_idx = np.argmin(avg_costs)
    best_s1, best_k, _ = np.unravel_index(best_idx, avg_costs.shape)
    return best_s1, best_k
