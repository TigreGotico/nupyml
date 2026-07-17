"""Baseline: plain logistic regression -- the one that struggles on imbalance.

Included as the reference the resampling baselines should beat: with no help for
the rare class, it leans toward the majority and its macro-F1 suffers.
"""
from nupyml.linear_model import LogisticRegression


def solve(X_train, y_train, X_test):
    return LogisticRegression(max_iter=500).fit(X_train, y_train).predict(X_test)
