"""Time-series classification: label a whole sequence by its SHAPE.

Each example is a 1-D time series (one row); the label is the pattern that
generated it. Unlike tabular classification the columns are ordered and the
discriminative signal is a shape (a frequency, a bump's presence) that appears at
a RANDOM phase/location, so a model that keys on absolute column values struggles
-- the point of a dedicated time-series method.

Scored by plain accuracy on a held-out split.
"""
import numpy as np

from nupyml.metrics import accuracy_score

KIND = "supervised"
GOAL = ("Classify each 1-D time series by its generating pattern "
        "(phase/location randomised); scored by accuracy.")
METRIC = "accuracy"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.80


def _make_series(rng, label, length=64):
    """All classes share the SAME carrier and noise; they differ only in a small
    local shape at a RANDOM position. Absolute column values are therefore
    uninformative -- only the shape (found by alignment/local features) tells the
    classes apart, which is exactly what a time-series method should exploit and a
    flat tabular model cannot."""
    t = np.linspace(0, 1, length)
    phase = rng.uniform(0, 2 * np.pi)               # shared random phase
    carrier = 0.7 * np.sin(2 * np.pi * 2 * t + phase)
    noise = 0.6 * rng.randn(length)                 # heavy noise vs the mark
    centre = rng.uniform(0.25, 0.75)                # random position of the mark
    local = np.exp(-((t - centre) ** 2) / (2 * 0.01))
    if label == 0:
        mark = 0.0 * local                          # no mark
    elif label == 1:
        mark = 1.1 * local                          # an upward bump
    else:
        mark = -1.1 * local                         # a downward notch
    return carrier + mark + noise


def load():
    rng = np.random.RandomState(0)
    n_per_class = 120
    X, y = [], []
    for label in (0, 1, 2):
        for _ in range(n_per_class):
            X.append(_make_series(rng, label))
            y.append(label)
    X, y = np.array(X), np.array(y)
    # deterministic shuffle + 70/30 split
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    n_tr = int(0.7 * len(y))
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


def metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
