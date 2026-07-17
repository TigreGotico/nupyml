"""Imbalanced fraud detection: a rare positive class (~8%).

Accuracy is useless here -- predicting "not fraud" always scores ~92% and
catches nothing -- so the metric is MACRO-F1, which averages the per-class F1 and
therefore punishes ignoring the rare class. The task exists to reward handling
the imbalance (resampling, or a model that does not collapse to the majority).
"""
import numpy as np

from nupyml.datasets import make_classification
from nupyml.metrics import f1_score
from nupyml.model_selection import train_test_split

KIND = "supervised"
GOAL = "Detect a rare fraud class (~8% positive) -- scored by macro-F1, not accuracy."
METRIC = "macro_f1"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.55


def load():
    # a hard, overlapping base (low class separation), then subsample the
    # positives to ~5% so the rare class is both rare AND not cleanly separable
    X, y = make_classification(n_samples=4000, n_features=12, n_informative=3,
                               class_sep=0.5, random_state=0)
    rng = np.random.RandomState(0)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    keep_pos = rng.choice(pos, size=int(0.05 * len(neg)), replace=False)
    idx = rng.permutation(np.concatenate([neg, keep_pos]))
    X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    return X_train, y_train, X_test, y_test


def metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")
