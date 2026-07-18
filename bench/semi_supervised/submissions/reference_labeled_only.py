"""Train on the labeled points ONLY -- the floor the SSL methods should beat."""
import numpy as np

from nupyml.linear_model import LogisticRegression


def solve(X_train, y_train, X_test):
    mask = y_train != -1
    clf = LogisticRegression(max_iter=500).fit(X_train[mask], y_train[mask])
    return clf.predict(X_test)
