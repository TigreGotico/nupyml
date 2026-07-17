"""1-NN with dynamic time warping -- the classic strong TSC baseline.

DTW aligns two series by warping the time axis, so a shape at a different phase
still matches. 1-nearest-neighbour under DTW is the reference every fancier
method is measured against.
"""
import numpy as np

from nupyml.sequence import dtw_distance


def solve(X_train, y_train, X_test):
    y_train = np.asarray(y_train)
    preds = np.empty(len(X_test), dtype=y_train.dtype)
    # a Sakoe-Chiba band keeps DTW O(n*band) and rarely hurts accuracy
    band = max(3, X_train.shape[1] // 10)
    for i, xt in enumerate(X_test):
        best_d, best_j = np.inf, 0
        for j, xr in enumerate(X_train):
            d = dtw_distance(xt, xr, window=band)
            if d < best_d:
                best_d, best_j = d, j
        preds[i] = y_train[best_j]
    return preds
