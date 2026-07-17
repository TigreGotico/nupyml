"""Image classification from HAND-CRAFTED features (no deep nets).

The submission receives raw 8x8 digit images (as an ``(n, 8, 8)`` array) and must
extract its own features before classifying -- the point is to exercise the
``image`` descriptors (HOG/LBP/etc.), not a pre-flattened matrix. A pre-deep-learning
pipeline on a small dataset. Scored by accuracy.
"""
import numpy as np

from nupyml.datasets import load_digits
from nupyml.metrics import accuracy_score

KIND = "supervised"
GOAL = ("Classify 8x8 digit images from hand-crafted features; scored by "
        "accuracy.")
METRIC = "accuracy"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.85


def load():
    X, y = load_digits(return_X_y=True)
    images = X.reshape(-1, 8, 8)                      # restore 2-D image shape
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(y))
    images, y = images[perm], y[perm]
    n_tr = int(0.7 * len(y))
    return images[:n_tr], y[:n_tr], images[n_tr:], y[n_tr:]


def metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
