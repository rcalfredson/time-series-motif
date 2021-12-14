from lib.segment import new_segment_size

def find_motifs(x):
  """Transforms time-series data into a sequence of motifs.
  
  Parameters:
      x (list): time-series data points

  Returns:
      motifs (ndarray): array of detected motifs
      sequence (ndarray): motif indices of the sequenced time-series data
  """
  starts = []
  ends = []
  starts.append(0)
  best_initial, _ = new_segment_size(x, )