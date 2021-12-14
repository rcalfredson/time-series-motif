import numpy as np


def distance(a: np.ndarray, b: np.ndarray):
    """compute Euclidean distance matrix

    adapted from Matlab code written by Roland Bunschoten,
    University of Amsterdam
    """
    a, b = [np.expand_dims(arr, 0) if arr.ndim == 1 else arr for arr in (a, b)]
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B should be of same dimensionality")
    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    aa, bb = [np.expand_dims(arr, 0) if arr.ndim == 1 else arr for arr in (aa, bb)]
    ab = np.matmul(a.conj().T, b)
    d = np.abs(
        np.tile(aa.conj().T, (1, bb.shape[1])) + np.tile(bb, (aa.shape[1], 1)) - 2 * ab
    )
    return d
