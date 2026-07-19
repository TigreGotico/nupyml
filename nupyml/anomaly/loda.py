"""Lightweight On-line Detector of Anomalies (Pevný, 2016)."""
import numpy as np
from ..utils import check_array, check_random_state
from .detectors import _Detector


class LODA(_Detector):
    """Lightweight On-line Detector of Anomalies (Pevný, 2016).

    THE IDEA
    --------
    A single 1-D histogram is a weak density model, but an ENSEMBLE of histograms
    on many random 1-D PROJECTIONS is a strong one -- and dazzlingly cheap. Project
    the data onto ``n_projections`` random (sparse) directions, build a histogram
    per projection, and score a point by the AVERAGE negative log-density it gets
    across the projections. A point that lands in a low-density bin on many
    projections is anomalous.

    WHY SPARSE PROJECTIONS
    ----------------------
    Each projection vector is mostly zeros (a random subset of features with
    random weights), which makes the projections cheap AND diverse, and lets the
    per-projection contributions double as a crude feature-importance for WHY a
    point is anomalous. No distances, trivially streamable, embarrassingly cheap.
    """

    def __init__(self, n_projections=100, n_bins=10, contamination=0.1,
                 random_state=None):
        self.n_projections = n_projections
        self.n_bins = n_bins
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        d = X.shape[1]
        n_nonzero = max(1, int(np.sqrt(d)))          # sparse projections
        self.projections_, self.edges_, self.logdens_ = [], [], []
        for _ in range(self.n_projections):
            w = np.zeros(d)
            idx = rng.choice(d, n_nonzero, replace=False)
            w[idx] = rng.randn(n_nonzero)
            proj = X @ w
            hist, edges = np.histogram(proj, bins=self.n_bins, density=True)
            self.projections_.append(w)
            self.edges_.append(edges)
            self.logdens_.append(np.log(hist + 1e-12))
        self._set_threshold(self.decision_function(X))
        return self

    def decision_function(self, X):
        X = check_array(X)
        score = np.zeros(len(X))
        for w, edges, logd in zip(self.projections_, self.edges_, self.logdens_):
            proj = X @ w
            b = np.clip(np.searchsorted(edges, proj) - 1, 0, len(logd) - 1)
            score += -logd[b]                        # negative log-density
        return score / self.n_projections


__all__ = ["LODA"]
