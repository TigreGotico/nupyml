"""Distance to the centre in the metric of the data's covariance."""
import numpy as np
from ..utils import check_array, check_random_state
from .detectors import _Detector


class MahalanobisDetector(_Detector):
    """Distance to the centre in the metric of the data's covariance.

    The oldest multivariate outlier score: the Mahalanobis distance
    ``sqrt((x-mu)' S^{-1} (x-mu))`` measures how many "standard deviations" a point
    is from the mean AFTER accounting for feature correlations -- so it flags
    points that are unusual relative to the ellipse the data actually occupies,
    not just far in raw units. Exactly right for a single roughly-Gaussian blob;
    blind to structure a mixture would need.
    """

    def __init__(self, contamination=0.1):
        self.contamination = contamination

    def fit(self, X):
        X = check_array(X)
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        self.precision_ = np.linalg.inv(cov)
        self._set_threshold(self.decision_function(X))
        return self

    def decision_function(self, X):
        Xc = check_array(X) - self.mean_
        return np.sqrt(np.einsum("ij,jk,ik->i", Xc, self.precision_, Xc))


__all__ = ["MahalanobisDetector"]
