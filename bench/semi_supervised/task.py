"""Semi-supervised classification: few labels, many unlabeled points.

Only ~8% of the training points are labeled; the rest carry the sentinel label
``-1``. A method that EXPLOITS the unlabeled structure (self-training, label
spreading) should beat one that trains on the labeled points alone. Scored by
accuracy on a fully-labeled held-out set.
"""
import numpy as np

from nupyml.datasets import make_classification
from nupyml.metrics import accuracy_score

KIND = "supervised"
GOAL = ("Classify with only ~8% of training labels (rest are -1); use the "
        "unlabeled points. Scored by accuracy.")
METRIC = "accuracy"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.80


def load():
    X, y = make_classification(n_samples=1000, n_features=8, n_informative=5,
                               n_classes=3, class_sep=1.0, random_state=0)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    n_tr = 700
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr].copy(), X[n_tr:], y[n_tr:]
    # hide ~95% of the training labels with the -1 sentinel
    hide = rng.rand(n_tr) > 0.05
    ytr[hide] = -1
    return Xtr, ytr, Xte, yte


def metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
