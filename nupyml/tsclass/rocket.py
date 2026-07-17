"""ROCKET: RandOm Convolutional KErnel Transform.

THE SURPRISE
------------
For a decade, time-series classification meant expensive elastic distances or
learned shapelets. ROCKET showed you can beat almost all of them with kernels you
NEVER TRAIN: generate thousands of RANDOM convolutional kernels, convolve each
with the series, and summarise every convolution by just two numbers -- its
maximum and its ``ppv`` (the proportion of positive values). Feed that feature
vector to a linear classifier. That's it. No kernel learning, no backprop.

WHY IT WORKS
------------
A random kernel is a random feature detector; with enough of them, some will by
chance align with whatever local patterns discriminate the classes, at whatever
scale (the random DILATION spreads the kernels across scales) and position. The
``ppv`` summary is the crucial ingredient -- it captures HOW MUCH of the series
responds to a pattern, not just how strongly, and it is what lifted ROCKET above
using the max alone. Thousands of cheap weak features + a linear model beats a
few expensive learned ones.

Dempster, Petitjean & Webb (2020).
"""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, ClassifierMixin, check_is_fitted
from ..utils import check_array, check_X_y, check_random_state


def _apply_kernel(series, weights, bias, dilation, padding):
    """Dilated 1-D convolution of one series with one kernel; return (ppv, max).

    The kernel taps are spaced ``dilation`` apart (so a length-9 kernel at
    dilation 4 spans 33 samples -- this is how one kernel sees a coarse scale),
    and ``padding`` zeros extend both ends so short series still produce outputs.
    """
    n = len(series)
    k = len(weights)
    if padding > 0:
        series = np.concatenate([np.zeros(padding), series, np.zeros(padding)])
    out_len = len(series) - (k - 1) * dilation
    if out_len <= 0:
        return 0.0, 0.0
    # convolution as a sum of dilated, shifted, weighted copies
    conv = np.full(out_len, bias, dtype=float)
    for j in range(k):
        conv += weights[j] * series[j * dilation:j * dilation + out_len]
    return float(np.mean(conv > 0)), float(conv.max())


class Rocket(BaseEstimator, TransformerMixin):
    """Transform each series into ``2 * n_kernels`` random-kernel features.

    Each kernel contributes its ``ppv`` and its ``max``. The kernels are sampled
    once at ``fit`` time (their number, weights, biases, dilations, and paddings
    are fixed thereafter), so ``transform`` is a deterministic feature map -- pair
    it with any linear classifier (see :class:`RocketClassifier`).
    """

    def __init__(self, n_kernels=1000, random_state=None):
        self.n_kernels = n_kernels
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        input_length = X.shape[1]
        self.kernels_ = []
        candidate_lengths = np.array([7, 9, 11])
        for _ in range(self.n_kernels):
            k = int(rng.choice(candidate_lengths))
            w = rng.normal(0, 1, k)
            w = w - w.mean()                        # zero-mean weights (no DC)
            bias = rng.uniform(-1, 1)
            # dilation spread log-uniformly, so kernels cover many time scales
            max_exp = np.log2((input_length - 1) / (k - 1)) if input_length > k else 0
            dilation = int(2 ** rng.uniform(0, max(0.0, max_exp)))
            dilation = max(1, dilation)
            padding = ((k - 1) * dilation) // 2 if rng.randint(2) else 0
            self.kernels_.append((w, bias, dilation, padding))
        return self

    def transform(self, X):
        check_is_fitted(self, "kernels_")
        X = check_array(X)
        feats = np.empty((len(X), 2 * len(self.kernels_)))
        for i, series in enumerate(X):
            for j, (w, bias, dil, pad) in enumerate(self.kernels_):
                ppv, mx = _apply_kernel(series, w, bias, dil, pad)
                feats[i, 2 * j] = ppv
                feats[i, 2 * j + 1] = mx
        return feats


class RocketClassifier(BaseEstimator, ClassifierMixin):
    """ROCKET features + a ridge classifier -- the canonical ROCKET pipeline.

    A ridge (L2) linear classifier is the recommended head: with thousands of
    weak, correlated features it is fast, closed-form-ish, and does not overfit
    the way an unregularised model would. Features are standardised first, as the
    ppv and max summaries live on very different scales.
    """

    def __init__(self, n_kernels=1000, alpha=1.0, random_state=None):
        self.n_kernels = n_kernels
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        from ..linear_model import RidgeClassifier
        from ..preprocessing import StandardScaler
        X, y = check_X_y(X, y)
        self.rocket_ = Rocket(self.n_kernels, self.random_state).fit(X)
        F = self.rocket_.transform(X)
        self.scaler_ = StandardScaler().fit(F)
        self.clf_ = RidgeClassifier(alpha=self.alpha).fit(
            self.scaler_.transform(F), y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self, "clf_")
        F = self.scaler_.transform(self.rocket_.transform(check_array(X)))
        return self.clf_.predict(F)


__all__ = ["Rocket", "RocketClassifier"]
