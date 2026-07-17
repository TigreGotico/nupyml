"""Baseline: SMOTE the training set, then logistic regression.

Synthesising minority points before fitting is the textbook fix for imbalance --
but note on this hard, overlapping task it barely moves macro-F1 over plain
logistic, and can even hurt when the synthetic points land across the boundary.
SMOTE is not a guaranteed win; the scoreboard is the honest judge.
"""
from nupyml.imbalance import SMOTE
from nupyml.linear_model import LogisticRegression


def solve(X_train, y_train, X_test):
    # resample ONLY the training data -- never the test set
    X_res, y_res = SMOTE(random_state=0).fit_resample(X_train, y_train)
    return LogisticRegression(max_iter=500).fit(X_res, y_res).predict(X_test)
