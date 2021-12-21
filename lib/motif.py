from lib.segment import new_segment_size

def find_motifs(x):
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
  best_initial, _ = new_segment_size(x, )