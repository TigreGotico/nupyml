"""Baseline: order-blind unigram + bigram counts, then logistic regression."""
import numpy as np

from nupyml.linear_model import LogisticRegression


def solve(X_train, y_train, X_test):
    vocab = sorted({sym for s in X_train for sym in s})
    bigrams = sorted({(a, b) for s in X_train for a, b in zip(s, s[1:])})
    v_idx = {v: i for i, v in enumerate(vocab)}
    b_idx = {b: i for i, b in enumerate(bigrams)}

    def featurize(X):
        F = np.zeros((len(X), len(vocab) + len(bigrams)))
        for i, s in enumerate(X):
            for sym in s:
                if sym in v_idx:
                    F[i, v_idx[sym]] += 1
            for a, b in zip(s, s[1:]):
                if (a, b) in b_idx:
                    F[i, len(vocab) + b_idx[(a, b)]] += 1
        return F

    clf = LogisticRegression(max_iter=500).fit(featurize(X_train), np.asarray(y_train))
    return clf.predict(featurize(X_test))
