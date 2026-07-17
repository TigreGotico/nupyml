"""A compact tsfresh-style feature bank for time series.

Rather than a bespoke time-series model, often the pragmatic win is to CRUSH each
series down to a fixed vector of interpretable summary statistics -- moments,
change statistics, autocorrelation, spectral energy, complexity -- and hand that
to any ordinary tabular classifier or regressor. Libraries like tsfresh compute
hundreds of such features; this is a curated, phase-invariant subset that covers
the main axes: level, spread, shape, dynamics, periodicity, and complexity.
"""
import numpy as np

from ..base import BaseEstimator, TransformerMixin
from ..utils import check_array


FEATURE_NAMES = [
    "mean", "std", "min", "max", "median", "iqr", "skew", "kurtosis",
    "abs_energy", "mean_abs_change", "mean_change", "count_above_mean",
    "longest_above_mean", "n_zero_crossings", "n_peaks",
    "autocorr_lag1", "autocorr_lag2", "autocorr_lag3",
    "fft_dominant_bin", "fft_dominant_mag", "spectral_entropy",
    "sample_entropy_proxy",
]


def _features(s):
    s = np.asarray(s, dtype=float)
    n = len(s)
    mean, std = s.mean(), s.std()
    diffs = np.diff(s)
    above = s > mean
    # longest run above the mean
    longest = cur = 0
    for a in above:
        cur = cur + 1 if a else 0
        longest = max(longest, cur)

    def autocorr(lag):
        if n <= lag or std < 1e-12:
            return 0.0
        return float(np.corrcoef(s[:-lag], s[lag:])[0, 1])

    fft = np.abs(np.fft.rfft(s - mean))
    power = fft ** 2
    p = power / (power.sum() + 1e-12)
    spectral_entropy = float(-(p * np.log(p + 1e-12)).sum())
    # a crude complexity proxy: normalised sum of squared successive differences
    complexity = float(np.sqrt(np.sum(diffs ** 2))) if n > 1 else 0.0
    # peak count: strictly greater than both neighbours
    peaks = int(np.sum((s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]))) if n > 2 else 0

    return np.array([
        mean, std, s.min(), s.max(), np.median(s),
        np.percentile(s, 75) - np.percentile(s, 25),
        float(((s - mean) ** 3).mean() / (std ** 3 + 1e-12)),
        float(((s - mean) ** 4).mean() / (std ** 4 + 1e-12)),
        float(np.sum(s ** 2)),
        float(np.mean(np.abs(diffs))) if n > 1 else 0.0,
        float(np.mean(diffs)) if n > 1 else 0.0,
        float(above.sum()),
        float(longest),
        float(np.sum((s[:-1] * s[1:]) < 0)) if n > 1 else 0.0,
        float(peaks),
        autocorr(1), autocorr(2), autocorr(3),
        float(fft.argmax()), float(fft.max()),
        spectral_entropy, complexity,
    ])


class TSFeatureExtractor(BaseEstimator, TransformerMixin):
    """Transform each series (row) into the fixed feature vector above.

    Stateless -- ``fit`` does nothing but validate. Compose it with any estimator
    (``make_pipeline(TSFeatureExtractor(), RandomForestClassifier())``) to get a
    strong, cheap, interpretable time-series model.
    """

    def fit(self, X, y=None):
        check_array(X)
        self.feature_names_ = list(FEATURE_NAMES)
        return self

    def transform(self, X):
        X = check_array(X)
        return np.array([_features(s) for s in X])

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)


__all__ = ["TSFeatureExtractor", "FEATURE_NAMES"]
