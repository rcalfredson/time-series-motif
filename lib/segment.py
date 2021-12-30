import numpy as np

from lib.dtw import dtw


def new_segment_size(x, cur, models, s_min, s_max, max_dist):
    num_models = len(models)
    avg_costs = np.ones((s_max, num_models + 1, s_max)) * np.infty
    for s in range(s_min, s_max + 1):
        if cur + s + 1 >= x.shape[1]:
            continue
        for k in range(num_models + 2):
            if k + 1 <= num_models:
                cur_model = models[k]
            else:
                cur_model = x[:, cur : cur + s - 1]
            x_cur = x[:, cur + s : min(x.shape[1], cur + s + s_max - 1)]
            _, dtw_mat, _, _ = dtw(cur_model, x_cur, max_dist)
            dtw_costs = dtw_mat[-1, :]
            avg_costs[s - 1, k - 1, 0 : x_cur.shape[1]] = dtw_costs / np.arange(
                1, x_cur.shape[1] + 1
            )
            avg_costs[s - 1, k - 1, 0:s_min] = np.inf
    _, best_idx = np.nanmin(avg_costs[:])
    best_s1, best_k, _ = np.unravel_index(best_idx, avg_costs.shape)
    return best_s1, best_k
