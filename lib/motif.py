import numpy as np

from lib.dtw import dtw
from lib.list_helpers import pad_insert
from lib.markov import MarkovChain
from lib.segment import new_segment_size


def forecast_motifs(x, pred_len, s_min, s_max, max_dist):
    ndim = x.shape[0]
    N = x.shape[1]
    tot_len = pred_len + N
    models, starts, ends, idx, best_prefix_length, _ = find_motifs(
        x, s_min, s_max, max_dist
    )
    print(f"prefix length: {best_prefix_length}")

    x_p = x
    p_starts = []
    p_ends = []
    suffix = models[idx[-1]][:, best_prefix_length:]
    if suffix.size != 0:
        x_p = np.vstack((x_p, (suffix + x[:, -1] - suffix[:, 0])))
        p_starts[0] = x.shape[1] + 1
        p_ends[0] = x_p.shape[1]
    m = MarkovChain(3)
    for i in range(idx):
        m.update(idx[:i])
    p_idx = []
    while x_p.shape[1] < tot_len:
        best_char = m.predict(idx + p_idx)
        p_idx.append(best_char)


def find_motifs(x, s_min, s_max, max_dist):
    """Transforms time-series data into a sequence of motifs.
  
  Parameters:
      x (ndarray): a d-by-N time series, where d is dimensionality and N is length
      s_min (int): min segment length
      s_max (int): max segment length
      max_dist (float): max warping distance

  Output:
      models (list): list of motifs
      starts (list): starts of each segment
      ends (list): ends of each segment
      idx (list): index of vocabulary term assigned to each segment
      best_prefix_length (int): can usually be ignored; this indicates the partial matching of a vocabulary term at the end of the data
      tot_err (float): total error in terms of description length
  """
    starts = []
    ends = []
    starts.append(0)
    best_initial, _ = new_segment_size(
        x, 0, [], s_min=s_min, s_max=s_max, max_dist=max_dist
    )
    ends.append(best_initial - 1)
    models = [x[:, starts[0] : ends[0]]]
    idx = [0]
    model_momentum = 0.8
    max_vocab = 5
    termination_threshold = 0
    new_cluster_threshold = 0.3
    mean_dev = np.mean(np.power(x - np.mean(x), 2))
    best_prefix_length = np.nan
    tot_err = 0
    while ends[-1] + termination_threshold < x.shape[1]:
        cur_idx = len(starts) + 1
        cur = ends[-1] + 1
        starts = pad_insert(starts, cur, cur_idx)

        print(f"Segment {cur_idx} at position {cur}========\n")
        num_models = len(models)
        avg_costs = np.ones((num_models, s_max)) * np.infty
        cur_end = min(cur + s_max - 1, x.shape[1])
        x_cur = x[:, cur:cur_end]
        for k in range(num_models):
            dtw_dist, dtw_mat, _, dtw_trace = dtw(models[k], x_cur, max_dist)
            dtw_costs = dtw_mat[-1, :]
            avg_costs[k, 0 : x_cur.shape[1]] = dtw_costs / np.arange(x_cur.shape[1])
            avg_costs[k, 0 : s_min - 1] = np.nan
        best_idx = np.argmin(avg_costs)
        best_cost = avg_costs[np.unravel_index(best_idx, avg_costs.shape)]
        best_k, best_size = np.unravel_index(best_idx, avg_costs.shape)

        if cur + s_max >= x.shape[1]:
            good_prefix_costs = np.ones((num_models, 1)) * np.nan
            good_prefix_lengths = np.ones((num_models, 1)) * np.nan
            for k in range(num_models):
                _, dtw_mat, _, _ = dtw(models[k], x_cur, max_dist)
                prefix_costs = dtw_mat[:, -1]
                avg_prefix_costs = prefix_costs / np.arange(
                    start=1, stop=len(models[k]) + 1
                )
                good_prefix_lengths[k] = np.argmin(avg_prefix_costs)
                good_prefix_costs[k] = avg_prefix_costs[
                    np.unravel_index(good_prefix_lengths[k], avg_prefix_costs.shape)
                ]
            best_prefix_k = np.argmin(good_prefix_costs)
            best_prefix_cost = good_prefix_costs[
                np.unravel_index(best_prefix_k, good_prefix_costs.shape)
            ]
            best_prefix_length = good_prefix_lengths[best_prefix_k]
            print(
                f"end state: best k is {best_k}, best cost is {best_cost:.3f},"
                f" best prefix cost is {best_prefix_cost:.3f}"
            )
            if best_prefix_cost < best_cost:
                print(f"ending with prefix")
                ends = pad_insert(ends, np.max(x.shape), cur_idx)
                idx = pad_insert(idx, best_prefix_k, cur_idx)
        print(f"cluster costs: {avg_costs[:, best_size]:.2f}")
        print(
            f"new cluster costs for {x.shape[0]}: ",
            f"{new_cluster_threshold*mean_dev*x.shape[0]:.2f}",
        )
        print(f"size chosen: {best_size}")
        x_best = x[:, cur : cur + best_size - 2]
        if best_cost > new_cluster_threshold * mean_dev and len(models) < max_vocab:
            print(f"=> new cluster")
            best_s1, _ = new_segment_size(x, cur, models, s_min, s_max, max_dist)
            ends = pad_insert(ends, cur + best_s1 - 1, cur_idx)
            idx = pad_insert(idx, num_models + 1, cur_idx)
            models[num_models + 1] = x[:, starts[cur_idx] : ends[cur_idx]]
            tot_err = tot_err + new_cluster_threshold * mean_dev * best_s1
        else:
            print(f"=> cluster {best_k}")
            ends = pad_insert(ends, cur + best_size - 1, cur_idx)
            idx = pad_insert(idx, best_k, cur_idx)
            tot_err = tot_err + best_cost * best_size
            _, _, _, dtw_trace = dtw(models[best_k], x_best, max_dist)
            trace_summed = np.zeros(models[best_k].shape)
            for t in range(dtw_trace.shape[0]):
                trace_summed[:, dtw_trace[t, 0]] = (
                    trace_summed[:, dtw_trace[t, 0]] + x_best[:, dtw_trace[t, 1]]
                )
            trace_counts = np.unique(dtw_trace[:, 0], return_counts=True)[:, 1].T
            trace_avg = trace_summed / trace_counts
            models[best_k] = (
                model_momentum * models[best_k] + (1 - model_momentum) * trace_avg
            )
    return models, starts, ends, idx, best_prefix_length, tot_err
