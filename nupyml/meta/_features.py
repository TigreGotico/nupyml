"""Meta-features: describe a whole dataset with a handful of numbers.

AutoML has to pick an algorithm for a dataset it has never seen. It cannot try
them all, so it describes the dataset with META-FEATURES -- its shape, its class
balance, the statistics of its columns, and cheap "landmark" accuracies from
trivial models -- and lets a meta-model, trained on many past datasets, predict
what will work. These are the classic groups (simple, statistical, and
landmarking) that drive algorithm selection and warm-starting.
"""
import numpy as np
from scipy import stats

from ..utils import check_array


def extract_meta_features(X, y=None):
    """Return a dict of dataset meta-features (simple, statistical, landmarking)."""
    X = check_array(X)
    n, d = X.shape
    feats = {
        "n_samples": float(n),
        "n_features": float(d),
        "samples_per_feature": float(n / d),
        # statistical: the shape of the feature distributions, averaged
        "mean_skew": float(np.nanmean(stats.skew(X, axis=0))),
        "mean_kurtosis": float(np.nanmean(stats.kurtosis(X, axis=0))),
        "mean_std": float(np.nanmean(X.std(axis=0))),
        # average absolute pairwise correlation (feature redundancy)
        "mean_abs_correlation": _mean_abs_corr(X),
    }
    if y is not None:
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        feats["n_classes"] = float(len(classes))
        feats["class_entropy"] = float(-(p * np.log2(p)).sum())
        # imbalance ratio: largest class / smallest class (1.0 = balanced)
        feats["imbalance_ratio"] = float(counts.max() / counts.min())
        # landmark: accuracy of a 1-nearest-neighbour leave-one-out classifier
        feats["landmark_1nn"] = _landmark_1nn(X, y)
    return feats


def _mean_abs_corr(X):
    if X.shape[1] < 2:
        return 0.0
    C = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices_from(C, k=1)
    vals = np.abs(C[iu])
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else 0.0


def _landmark_1nn(X, y, max_n=300):
    # a cheap capacity probe: how separable is the data to a nearest neighbour?
    from scipy.spatial.distance import cdist
    if len(X) > max_n:
        idx = np.random.RandomState(0).choice(len(X), max_n, replace=False)
        X, y = X[idx], y[idx]
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)                          # leave-one-out
    nn = D.argmin(axis=1)
    return float((y[nn] == y).mean())


__all__ = ["extract_meta_features"]
