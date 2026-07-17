"""BOSS: Bag-Of-SFA-Symbols -- classify a series by its histogram of "words".

THE IDEA
--------
Slide a window along the series; turn each window into a short SYMBOLIC word via
SFA (Symbolic Fourier Approximation); count the words. The histogram of words is
the feature, and two series are compared by how similar their word histograms
are. This is the text-classification bag-of-words idea transplanted onto signals.

WHY SFA, AND WHY IT IS ROBUST
-----------------------------
SFA takes the low-frequency Fourier coefficients of each window (a low-pass
filter -- it keeps the shape and discards the jitter) and DISCRETISES them into a
few symbols. The low-pass step denoises; the discretisation makes near-identical
windows collapse to the same word, so noise and small warps stop mattering. That
built-in tolerance is why BOSS is strong on noisy, phase-shifted data where a raw
distance struggles.

NUMEROSITY REDUCTION
--------------------
Consecutive identical words (a slowly varying stretch produces the same word many
times) are counted once, so a stable region does not swamp the histogram. This
small trick is important to BOSS's accuracy.

The breakpoints that discretise each Fourier coefficient are learned from the
training windows (MCB -- multiple coefficient binning: per-coefficient quantiles),
so the alphabet is data-adaptive.

Schäfer (2015).
"""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, check_is_fitted
from ..utils import check_array, check_X_y, check_random_state
from ..preprocessing import LabelEncoder


def _znorm(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else x - x.mean()


def _dft_coeffs(window, word_length):
    """The first ``word_length`` real DFT values (real/imag interleaved),
    dropping the DC term -- the low-frequency shape of the window."""
    f = np.fft.rfft(_znorm(window))
    vals = []
    for c in f[1:]:                                 # skip DC (coefficient 0)
        vals.append(c.real)
        vals.append(c.imag)
        if len(vals) >= word_length:
            break
    while len(vals) < word_length:
        vals.append(0.0)
    return np.array(vals[:word_length])


class BOSS(BaseEstimator, ClassifierMixin):
    """1-NN over BOSS word histograms, using the (asymmetric) BOSS distance."""

    def __init__(self, window_size=None, word_length=4, alphabet_size=4,
                 random_state=None):
        self.window_size = window_size
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.random_state = random_state

    # --- SFA ---------------------------------------------------------------
    def _all_windows(self, series, w):
        return [series[s:s + w] for s in range(len(series) - w + 1)]

    def _fit_breakpoints(self, X, w):
        """MCB: per-coefficient quantile breakpoints over all training windows."""
        coeffs = []
        for series in X:
            for win in self._all_windows(series, w):
                coeffs.append(_dft_coeffs(win, self.word_length))
        coeffs = np.array(coeffs)
        qs = np.linspace(0, 100, self.alphabet_size + 1)[1:-1]
        # breakpoints_[j] are the interior bin edges for Fourier coefficient j
        self.breakpoints_ = [np.percentile(coeffs[:, j], qs)
                             for j in range(self.word_length)]

    def _word(self, window):
        c = _dft_coeffs(window, self.word_length)
        return tuple(int(np.searchsorted(self.breakpoints_[j], c[j]))
                     for j in range(self.word_length))

    def _histogram(self, series, w):
        hist = {}
        prev = None
        for win in self._all_windows(series, w):
            word = self._word(win)
            if word == prev:                        # numerosity reduction
                continue
            prev = word
            hist[word] = hist.get(word, 0) + 1
        return hist

    @staticmethod
    def _boss_distance(a, b):
        # asymmetric: only words PRESENT in the query a contribute
        return sum((cnt - b.get(word, 0)) ** 2 for word, cnt in a.items())

    # --- estimator API -----------------------------------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        L = X.shape[1]
        self.w_ = self.window_size or max(self.word_length + 1, L // 4)
        self._fit_breakpoints(X, self.w_)
        self.hists_ = [self._histogram(s, self.w_) for s in X]
        self.y_ = self._le.transform(y)
        return self

    def predict(self, X):
        check_is_fitted(self, "hists_")
        X = check_array(X)
        preds = np.empty(len(X), dtype=int)
        for i, series in enumerate(X):
            h = self._histogram(series, self.w_)
            dists = [self._boss_distance(h, ref) for ref in self.hists_]
            preds[i] = self.y_[int(np.argmin(dists))]
        return self.classes_[preds]


__all__ = ["BOSS"]
