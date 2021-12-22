import numpy as np

from lib.dtw import dtw
from lib.segment import new_segment_size


def find_motifs(x, s_min, s_max, max_dist, verbose):
    """Transforms time-series data into a sequence of motifs.
  
  Parameters:
      x (ndarray): a d-by-N time series, where d is dimensionality and N is length
      s_min (int): min segment length
      s_max (int): max segment length
      max_dist (float): max warping distance
      verbose: verbosity level (0 for low and 1 for high)

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
        if cur_idx + 1 < len(starts):
            starts.extend([0] * (cur_idx + 1 - len(starts)))
        starts[cur_idx] = cur
        print(f"Segment {cur_idx} at position {cur}========\n")
        num_models = len(models)
        ave_costs = np.ones((num_models, s_max)) * np.infty
        cur_end = min(cur + s_max - 1, x.shape[1])
        x_cur = x[:, cur:cur_end]
        for k in range(num_models):
            dtw_dist, dtw_mat, _, dtw_trace = dtw(models[k], x_cur, max_dist)
            dtw_costs = dtw_mat[-1, :]
            ave_costs[k, 0 : x_cur.shape[1]] = dtw_costs / np.arange(x_cur.shape[1])
            ave_costs[k, 0:s_min-1] = np.nan
        best_idx = np.argmin(ave_costs)
        best_cost = ave_costs[np.unravel_index(best_idx, ave_costs.shape)]
