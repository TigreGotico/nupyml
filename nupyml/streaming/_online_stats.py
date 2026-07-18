"""Online (streaming) statistics: covariance and quantiles in one pass.

Computing statistics over a data STREAM -- where you see each point once and
cannot store them all -- needs incremental estimators. These update in O(1) (or
O(d^2)) per point and never revisit the data.
"""
import numpy as np


class OnlineCovariance:
    """Mean and covariance in ONE PASS, numerically stably (Welford, 1962).

    The textbook covariance formula ``E[xx'] - E[x]E[x]'`` catastrophically loses
    precision when the mean is large relative to the spread (it subtracts two big
    nearly-equal numbers). Welford's method updates the mean and the co-moment
    matrix incrementally -- each new point nudges them using its deviation from the
    CURRENT mean -- so it is both single-pass (no data stored) and numerically
    stable. ``update`` adds a point; ``covariance`` reads the current estimate.
    """

    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features)
        self.M2 = np.zeros((n_features, n_features))   # co-moment

    def update(self, x):
        x = np.asarray(x, float)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += np.outer(delta, x - self.mean)      # uses pre- and post-update mean
        return self

    def update_batch(self, X):
        for x in np.asarray(X, float):
            self.update(x)
        return self

    def covariance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else np.zeros_like(self.M2)


class OnlineQuantile:
    """Estimate a quantile from a stream in O(1) memory (the P^2 algorithm).

    You cannot sort a stream you never fully store. The P^2 (piecewise-parabolic)
    algorithm keeps just FIVE markers -- running estimates of the min, the target
    quantile, and points around it -- and, as each observation arrives, nudges the
    marker positions and heights along a parabola so the middle marker converges to
    the desired quantile. Constant memory, no data retained, and accurate for
    streaming percentiles (latency SLAs, anomaly thresholds). ``update`` per point,
    ``quantile`` reads the estimate.
    """

    def __init__(self, q=0.5):
        self.q = q
        self.n = []            # marker positions (counts)
        self.heights = []      # marker heights (values)
        self._init = []

    def update(self, x):
        x = float(x)
        if len(self._init) < 5:
            self._init.append(x)
            if len(self._init) == 5:
                self._init.sort()
                self.heights = list(self._init)
                self.n = [1, 2, 3, 4, 5]
                self.np_ = [1, 1 + 2 * self.q, 1 + 4 * self.q, 3 + 2 * self.q, 5]
                self.dn = [0, self.q / 2, self.q, (1 + self.q) / 2, 1]
            return self
        # find the cell k that x falls into
        if x < self.heights[0]:
            self.heights[0] = x; k = 0
        elif x >= self.heights[4]:
            self.heights[4] = x; k = 3
        else:
            k = next(i for i in range(4) if self.heights[i] <= x < self.heights[i + 1])
        for i in range(k + 1, 5):
            self.n[i] += 1
        for i in range(5):
            self.np_[i] += self.dn[i]
        for i in range(1, 4):                          # adjust the interior markers
            d = self.np_[i] - self.n[i]
            if (d >= 1 and self.n[i + 1] - self.n[i] > 1) or \
               (d <= -1 and self.n[i - 1] - self.n[i] < -1):
                d = int(np.sign(d))
                h = self._parabolic(i, d)
                if self.heights[i - 1] < h < self.heights[i + 1]:
                    self.heights[i] = h
                else:
                    self.heights[i] = self._linear(i, d)
                self.n[i] += d
        return self

    def _parabolic(self, i, d):
        n = self.n; h = self.heights
        return h[i] + d / (n[i + 1] - n[i - 1]) * (
            (n[i] - n[i - 1] + d) * (h[i + 1] - h[i]) / (n[i + 1] - n[i])
            + (n[i + 1] - n[i] - d) * (h[i] - h[i - 1]) / (n[i] - n[i - 1]))

    def _linear(self, i, d):
        return self.heights[i] + d * (self.heights[i + d] - self.heights[i]) \
            / (self.n[i + d] - self.n[i])

    def update_batch(self, xs):
        for x in np.ravel(xs):
            self.update(x)
        return self

    def quantile(self):
        if len(self._init) < 5:
            return float(np.quantile(self._init, self.q)) if self._init else 0.0
        return float(self.heights[2])


__all__ = ["OnlineCovariance", "OnlineQuantile"]
