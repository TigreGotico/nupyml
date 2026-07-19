"""Anomaly = how badly PCA (fit on the data) RECONSTRUCTS the point."""
import numpy as np
from ..utils import check_array, check_random_state
from .detectors import _Detector


class PCAReconstructionDetector(_Detector):
    """Anomaly = how badly PCA (fit on the data) RECONSTRUCTS the point.

    Fit PCA keeping the top components; a normal point lives near that principal
    subspace and is reconstructed with tiny error, while an anomaly -- off the
    manifold the bulk of the data spans -- is reconstructed poorly. The
    reconstruction error IS the score. The linear cousin of an autoencoder anomaly
    detector, and a strong baseline when the inliers really do lie near a
    low-dimensional subspace.
    """

    def __init__(self, n_components=2, contamination=0.1):
        self.n_components = n_components
        self.contamination = contamination

    def fit(self, X):
        from ..decomposition import PCA
        X = check_array(X)
        self.pca_ = PCA(n_components=self.n_components).fit(X)
        self._set_threshold(self.decision_function(X))
        return self

    def decision_function(self, X):
        X = check_array(X)
        recon = self.pca_.inverse_transform(self.pca_.transform(X))
        return np.sqrt(np.sum((X - recon) ** 2, axis=1))


# --- automatic thresholding -----------------------------------------------


__all__ = ["PCAReconstructionDetector"]
