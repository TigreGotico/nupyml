"""Detect objects with a boosted cascade of Haar features (Viola & Jones, 2001)."""
import numpy as np
from ..base import BaseEstimator
from . import integral_image, rectangle_sum


def _haar_features(ii, window, step=4):
    # two-rectangle horizontal/vertical Haar features at several sizes and
    # positions across the window (so features cover corners AND the centre)
    feats = []
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    for fs in (window // 4, window // 2):
        if fs < 2:
            continue
        h = fs // 2
        for y in range(0, H - fs + 1, step):
            for x in range(0, W - fs + 1, step):
                left = rectangle_sum(ii, y, x, y + fs, x + h)
                right = rectangle_sum(ii, y, x + h, y + fs, x + fs)
                feats.append(right - left)
                top = rectangle_sum(ii, y, x, y + h, x + fs)
                bot = rectangle_sum(ii, y + h, x, y + fs, x + fs)
                feats.append(bot - top)
    return np.array(feats)


class ViolaJones(BaseEstimator):
    """Detect objects with a boosted cascade of Haar features (Viola & Jones, 2001).

    The breakthrough that made real-time face detection possible. Features are
    simple HAAR rectangles (sums of pixel intensities in adjacent boxes), each
    computable in constant time from the INTEGRAL IMAGE. AdaBoost picks the few
    features that best separate object from background and weights them; arranging
    the resulting stumps as a CASCADE lets the easy negatives be rejected in the
    first cheap stages. Here: Haar features over the integral image + an AdaBoost
    stump classifier trained to detect a pattern (e.g. a bright central region).
    """

    def __init__(self, window=24, n_rounds=30, step=4):
        self.window = window
        self.n_rounds = n_rounds
        self.step = step

    def _features(self, images):
        return np.array([_haar_features(integral_image(im), self.window, self.step)
                         for im in images])

    def fit(self, images, y):
        X = self._features(images)
        y = np.where(np.asarray(y) > 0, 1.0, -1.0)
        n, m = X.shape
        w = np.ones(n) / n
        self.stumps_ = []
        for _ in range(self.n_rounds):
            best = None
            for j in range(m):
                thr = np.median(X[:, j])
                for polarity in (1, -1):
                    pred = np.where(polarity * (X[:, j] - thr) > 0, 1.0, -1.0)
                    err = w[pred != y].sum()
                    if best is None or err < best[0]:
                        best = (err, j, thr, polarity)
            err, j, thr, pol = best
            err = min(max(err, 1e-6), 1 - 1e-6)
            alpha = 0.5 * np.log((1 - err) / err)
            self.stumps_.append((j, thr, pol, alpha))
            pred = np.where(pol * (X[:, j] - thr) > 0, 1.0, -1.0)
            w = w * np.exp(-alpha * y * pred)
            w /= w.sum()
        return self

    def decision_function(self, images):
        X = self._features(images)
        score = np.zeros(len(X))
        for j, thr, pol, alpha in self.stumps_:
            score += alpha * np.where(pol * (X[:, j] - thr) > 0, 1.0, -1.0)
        return score

    def predict(self, images):
        return (self.decision_function(images) > 0).astype(int)


__all__ = ["ViolaJones"]
