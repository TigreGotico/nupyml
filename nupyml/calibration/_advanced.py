"""Calibration beyond Platt/isotonic: temperature scaling, histogram binning,
beta calibration, and the expected calibration error.

A classifier can be accurate yet MISCALIBRATED -- its 0.9 confidences right only
70% of the time. These re-map confidences to match reality, and ECE measures how
far off they were.
"""
import numpy as np

from ..base import BaseEstimator


def expected_calibration_error(y_true, probs, n_bins=10):
    """ECE: the gap between confidence and accuracy, averaged over bins.

    Bin predictions by their confidence (max probability); in a well-calibrated
    model, the average confidence in each bin equals the accuracy in that bin. ECE
    is the average absolute confidence-minus-accuracy gap, weighted by bin size --
    a single number for "how much should I trust the probabilities". ``probs`` is
    the (n, n_classes) probability matrix (or (n,) for binary positive-class prob).
    """
    probs = np.asarray(probs, float)
    y_true = np.asarray(y_true)
    if probs.ndim == 1:
        conf = np.maximum(probs, 1 - probs)
        pred = (probs >= 0.5).astype(int)
    else:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() > 0:
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


class TemperatureScaling(BaseEstimator):
    """Divide the logits by a single learned temperature ``T`` (Guo et al., 2017).

    Modern networks are systematically OVER-confident. Temperature scaling is the
    minimal fix: rescale ALL logits by one scalar ``T > 1`` before the softmax,
    which softens every probability without changing the argmax -- so ACCURACY is
    untouched and only the confidences move. ``T`` is fit by minimising NLL on a
    validation set. One parameter, no accuracy cost, and it fixes most of the
    miscalibration -- which is why it is the default post-hoc calibrator.

    ``fit`` takes validation LOGITS and labels; ``transform`` returns calibrated
    probabilities.
    """

    def __init__(self, max_iter=200):
        self.max_iter = max_iter

    def _nll(self, logits, y, T):
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(y)), y].mean()

    def fit(self, logits, y):
        logits = np.asarray(logits, float)
        y = np.asarray(y)
        # 1-D golden-section-ish search over T in [0.05, 10]
        Ts = np.linspace(0.05, 10.0, 400)
        nlls = [self._nll(logits, y, T) for T in Ts]
        self.temperature_ = float(Ts[int(np.argmin(nlls))])
        return self

    def transform(self, logits):
        z = np.asarray(logits, float) / self.temperature_
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


class HistogramBinning(BaseEstimator):
    """Calibrate binary probabilities by BINNING and using each bin's accuracy.

    The most direct non-parametric calibrator: partition predicted probabilities
    into bins, and map every prediction in a bin to the OBSERVED positive rate of
    that bin. Assumption-free and easy, but blocky (piecewise-constant) and
    hungry for data per bin -- the trade against smooth methods like isotonic or
    beta. ``fit`` on scores + binary labels; ``transform`` maps new scores.
    """

    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def fit(self, scores, y):
        scores = np.asarray(scores, float)
        y = np.asarray(y)
        self.edges_ = np.linspace(0, 1, self.n_bins + 1)
        self.bin_prob_ = np.zeros(self.n_bins)
        for b in range(self.n_bins):
            lo, hi = self.edges_[b], self.edges_[b + 1]
            mask = (scores >= lo) & (scores < hi) if b < self.n_bins - 1 else \
                (scores >= lo) & (scores <= hi)
            self.bin_prob_[b] = y[mask].mean() if mask.sum() > 0 else (lo + hi) / 2
        return self

    def transform(self, scores):
        scores = np.asarray(scores, float)
        idx = np.clip(np.searchsorted(self.edges_, scores) - 1, 0, self.n_bins - 1)
        return self.bin_prob_[idx]


class BetaCalibration(BaseEstimator):
    """Beta calibration: a flexible 2-parameter map for binary probabilities
    (Kull et al., 2017).

    Platt scaling (a logistic map) is symmetric and often too rigid; isotonic is
    flexible but can overfit and is non-smooth. Beta calibration fits a logistic
    regression on ``log(p)`` and ``log(1-p)``::

        calibrated = sigmoid(a*log(p) - b*log(1-p) + c)

    which is the family of maps derived from Beta likelihoods -- smooth, and able
    to represent the asymmetric S-curves real classifiers produce, with only three
    parameters. A strong default between Platt and isotonic.
    """

    def fit(self, scores, y):
        from ..linear_model import LogisticRegression
        p = np.clip(np.asarray(scores, float), 1e-6, 1 - 1e-6)
        feats = np.column_stack([np.log(p), -np.log(1 - p)])
        self.lr_ = LogisticRegression(max_iter=500).fit(feats, np.asarray(y))
        return self

    def transform(self, scores):
        p = np.clip(np.asarray(scores, float), 1e-6, 1 - 1e-6)
        feats = np.column_stack([np.log(p), -np.log(1 - p)])
        return self.lr_.predict_proba(feats)[:, 1]


__all__ = ["expected_calibration_error", "TemperatureScaling", "HistogramBinning",
           "BetaCalibration"]
