"""DBA: the average of a set of series under DYNAMIC TIME WARPING (Petitjean, 2011)."""
import numpy as np


def dtw_barycenter_averaging(series, n_iter=10):
    """DBA: the average of a set of series under DYNAMIC TIME WARPING (Petitjean, 2011).

    Averaging time series pointwise blurs features that occur at slightly different
    TIMES (two heartbeats a few samples apart average into a smear). DBA instead
    averages under DTW: align every series to a candidate average by warping, then
    update each average point to the mean of the points aligned to it, and iterate.
    The result preserves the shared SHAPE rather than smearing it -- the correct
    centroid for DTW-based clustering (k-means with DTW). Returns the barycenter
    series.
    """
    from ..sequence import dtw_path
    series = [np.asarray(s, float) for s in series]
    avg = series[len(series) // 2].copy()              # init from a medoid-ish member
    for _ in range(n_iter):
        assoc = [[] for _ in range(len(avg))]
        for s in series:
            path, _ = dtw_path(avg, s)
            for i, j in path:
                assoc[i].append(s[j])                  # points aligned to avg[i]
        avg = np.array([np.mean(a) if a else avg[k] for k, a in enumerate(assoc)])
    return avg


__all__ = ["dtw_barycenter_averaging"]
