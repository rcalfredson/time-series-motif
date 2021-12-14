from numpy import ndarray


def dtw(vec_existing: ndarray, vec_new, max_dist):
    rows, N = vec_existing.shape
    _, M = vec_new.shape

    
