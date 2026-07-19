"""Baseline: mine frequent subsequences per class, use them as features."""
import numpy as np

from nupyml.patterns import prefixspan
from nupyml.linear_model import LogisticRegression


def _is_subseq(pattern, seq):
    it = iter(seq)
    return all(sym in it for sym in pattern)


def solve(X_train, y_train, X_test):
    y_train = np.asarray(y_train)
    # mine the frequent subsequences that characterise each class
    patterns = set()
    for cls in np.unique(y_train):
        seqs = [list(s) for s in X_train[y_train == cls]]
        for pat, _ in prefixspan(seqs, min_support=0.3):
            if len(pat) >= 2:                              # ordered patterns only
                patterns.add(pat)
    patterns = sorted(patterns)

    def featurize(X):
        return np.array([[int(_is_subseq(p, list(s))) for p in patterns] for s in X])

    clf = LogisticRegression(max_iter=500).fit(featurize(X_train), y_train)
    return clf.predict(featurize(X_test))
