"""Streaming concept-drift detectors.

Each detector is fed one value at a time via ``update(x)`` and exposes
``drift_detected_`` (and, for the classifier-error ones, ``warning_detected_``).
The stream is usually a sequence of prediction errors (1/0) or a monitored
feature; the detector holds only a small running summary, never the history.
"""
import numpy as np
from scipy import stats

from ..base import BaseEstimator


class ADWIN(BaseEstimator):
    """ADaptive WINdowing: keep a window, split it where the means diverge.

    THE IDEA
    --------
    Maintain a window of recent values. After each new value, check every way of
    splitting the window into an older part and a newer part: if the two parts'
    means differ by more than statistical noise allows (a Hoeffding-style bound),
    conclude the process changed at the split, DROP the older part, and flag
    drift. So the window automatically grows while the stream is stable and
    shrinks the moment it shifts -- detection and forgetting in one mechanism.

    WHY IT IS THE PRINCIPLED CHOICE
    -------------------------------
    The split test comes with a guarantee: the false-alarm and missed-detection
    rates are bounded by the confidence ``delta``, derived, not tuned. And because
    the window adapts to the current rate of change, there is no window-size to
    pick -- the data sets it. This simplified version keeps the exact window and
    tests splits directly; the published one uses an exponential-histogram
    summary to do it in log space.

    Bifet & Gavalda (2007).
    """

    def __init__(self, delta=0.002):
        self.delta = delta
        self.reset()

    def reset(self):
        self.window = []
        self.drift_detected_ = False
        return self

    def update(self, value):
        self.window.append(float(value))
        self.drift_detected_ = False
        n = len(self.window)
        if n < 2:
            return self
        arr = np.array(self.window)
        total = arr.sum()
        # try every cut point; a significant mean gap means drift at that cut
        for i in range(1, n):
            n0, n1 = i, n - i
            m0 = arr[:i].mean()
            m1 = arr[i:].mean()
            # harmonic window size for the Hoeffding bound
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            var = arr.var() + 1e-12
            eps = np.sqrt(2.0 / m * var * np.log(2.0 / self.delta)) \
                + 2.0 / (3.0 * m) * np.log(2.0 / self.delta)
            if abs(m0 - m1) > eps:
                # drift: forget everything before the cut and flag it
                self.window = self.window[i:]
                self.drift_detected_ = True
                break
        return self


class DDM(BaseEstimator):
    """Drift Detection Method: watch a classifier's error rate rise.

    THE IDEA
    --------
    Feed it the stream of 0/1 errors from a classifier in production. Under a
    binomial model the error rate ``p`` and its standard deviation ``s`` are
    tracked, and the method remembers the BEST (lowest) ``p + s`` seen so far --
    the model's high-water mark. When the current ``p + s`` climbs several
    standard deviations above that minimum, the model is doing worse than it ever
    did, which means the concept has drifted:

        p + s > p_min + 2 s_min   -> WARNING (start buffering new data)
        p + s > p_min + 3 s_min   -> DRIFT   (retrain)

    The two-level warning/alarm design is DDM's contribution: the warning gives
    you time to collect fresh training data before the alarm forces a retrain, so
    you are not caught flat-footed. It assumes error rates only need to be watched
    for INCREASES and reacts slowly to gradual drift -- which is what EDDM
    improves.

    Gama et al. (2004).
    """

    def __init__(self, warning_level=2.0, drift_level=3.0, min_samples=30):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_samples = min_samples
        self.reset()

    def reset(self):
        self.n = 0
        self.p = 1.0
        self.p_min = np.inf
        self.s_min = np.inf
        self.warning_detected_ = False
        self.drift_detected_ = False
        return self

    def update(self, error):
        self.n += 1
        # running error rate and its binomial standard deviation
        self.p += (error - self.p) / self.n
        s = np.sqrt(self.p * (1 - self.p) / self.n)
        self.warning_detected_ = False
        self.drift_detected_ = False
        if self.n < self.min_samples:
            return self
        if self.p + s < self.p_min + self.s_min:    # a new best: update the mark
            self.p_min, self.s_min = self.p, s
        if self.p + s > self.p_min + self.drift_level * self.s_min:
            self.reset()                            # retrain => fresh statistics
            self.drift_detected_ = True             # AFTER reset (which clears it)
        elif self.p + s > self.p_min + self.warning_level * self.s_min:
            self.warning_detected_ = True
        return self


class EDDM(BaseEstimator):
    """Early Drift Detection Method: watch the SPACING between errors.

    DDM tracks the error RATE, which changes slowly under gradual drift. EDDM
    instead tracks the DISTANCE between consecutive errors: as a model degrades,
    errors bunch closer together, and the average gap between them SHRINKS before
    the overall rate has moved much. Watching that gap catches slow, gradual drift
    earlier than DDM's rate test -- hence "early". Same warning/alarm two-level
    scheme, keyed off the ratio of the current mean-gap-plus-spread to its best.

    Baena-Garcia et al. (2006).
    """

    def __init__(self, warning_level=0.95, drift_level=0.90, min_errors=30):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_errors = min_errors
        self.reset()

    def reset(self):
        self.n = 0
        self.n_errors = 0
        self.last_error_pos = 0
        self.mean_dist = 0.0
        self.m2 = 0.0                               # for Welford variance
        self.max_score = 0.0
        self.warning_detected_ = False
        self.drift_detected_ = False
        return self

    def update(self, error):
        self.n += 1
        self.warning_detected_ = False
        self.drift_detected_ = False
        if not error:
            return self
        self.n_errors += 1
        dist = self.n - self.last_error_pos        # gap since the last error
        self.last_error_pos = self.n
        # Welford update of the mean and variance of the gaps
        delta = dist - self.mean_dist
        self.mean_dist += delta / self.n_errors
        self.m2 += delta * (dist - self.mean_dist)
        if self.n_errors < self.min_errors:
            return self
        std = np.sqrt(self.m2 / self.n_errors)
        score = self.mean_dist + 2 * std           # larger = errors more spread
        if score > self.max_score:                 # best (most-spread) so far
            self.max_score = score
        ratio = score / (self.max_score + 1e-12)
        if ratio < self.drift_level:               # gaps shrank a lot => drift
            self.reset()
            self.drift_detected_ = True             # AFTER reset (which clears it)
        elif ratio < self.warning_level:
            self.warning_detected_ = True
        return self


class PageHinkley(BaseEstimator):
    """The Page-Hinkley test: a cumulative sum that trips on a sustained shift.

    THE IDEA
    --------
    Accumulate how far each value falls BELOW (or above) the running mean, minus a
    small tolerance ``delta`` that absorbs noise. As long as the stream is
    stationary this cumulative sum hovers near its running minimum; once the mean
    shifts, the sum drifts steadily away from that minimum, and when the gap
    exceeds ``threshold`` it alarms. It is the classic sequential change-point
    test (CUSUM's cousin), watching for a PERSISTENT change rather than a single
    spike.

    ITS TWO KNOBS ARE IN THE DATA'S UNITS
    -------------------------------------
    ``delta`` is a tolerance -- the size of change to IGNORE as noise -- and
    ``threshold`` is how much accumulated evidence to DEMAND before alarming.
    Both live in the units of the stream, so they must match its scale: a
    ``delta`` far below the noise level lets the cumulative sum random-walk past
    any fixed threshold on noise alone (a false alarm), and one far above it
    swallows the real change. Rule of thumb: ``delta`` about half the shift you
    care about, ``threshold`` a small multiple of the noise standard deviation.
    Standardise the stream first and the defaults here are reasonable.

    Page (1954).
    """

    def __init__(self, delta=0.5, threshold=10.0):
        self.delta = delta
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.cumsum = 0.0
        self.min_cumsum = 0.0
        self.drift_detected_ = False
        return self

    def update(self, value):
        self.n += 1
        self.mean += (value - self.mean) / self.n
        # accumulate deviation above the mean, past the noise tolerance delta. A
        # genuine upward shift makes this climb steadily away from its running
        # minimum; noise alone keeps it hovering near it. (An earlier version
        # multiplied by a forgetting factor, which damped the sum so it never
        # reached the threshold -- removed.)
        self.cumsum += value - self.mean - self.delta
        self.min_cumsum = min(self.min_cumsum, self.cumsum)
        detected = (self.cumsum - self.min_cumsum) > self.threshold
        if detected:
            self.reset()                            # reset() clears the flag, so
        self.drift_detected_ = detected             # set it afterwards
        return self


class KSWIN(BaseEstimator):
    """Kolmogorov-Smirnov Windowing: distribution-free drift on window shape.

    THE IDEA
    --------
    Keep a sliding window of the most recent values. Compare a RECENT sub-window
    against an OLDER reference sub-window with the two-sample Kolmogorov-Smirnov
    test, which asks whether two samples come from the same distribution -- using
    the largest gap between their empirical CDFs, with no assumption about the
    shape. When the KS statistic says the two windows differ significantly, flag
    drift.

    Because KS is distribution-free, KSWIN catches changes in the DISTRIBUTION'S
    SHAPE (variance, skew, multimodality), not just its mean -- the kind of drift
    that the mean-based detectors (ADWIN, Page-Hinkley) sail straight past.

    Raab, Heusinger & Schleif (2020).
    """

    def __init__(self, alpha=0.005, window_size=100, stat_size=30):
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.reset()

    def reset(self):
        self.window = []
        self.drift_detected_ = False
        return self

    def update(self, value):
        self.window.append(float(value))
        self.drift_detected_ = False
        if len(self.window) > self.window_size:
            self.window.pop(0)                      # slide the window
        if len(self.window) < self.window_size:
            return self
        # the most recent `stat_size` values vs an older reference sample
        recent = np.array(self.window[-self.stat_size:])
        reference = np.array(self.window[:-self.stat_size])
        stat, p = stats.ks_2samp(reference, recent)
        if p < self.alpha:
            self.drift_detected_ = True
            # keep only the recent window as the new reference after a change
            self.window = self.window[-self.stat_size:]
        return self


__all__ = ["ADWIN", "DDM", "EDDM", "PageHinkley", "KSWIN"]
