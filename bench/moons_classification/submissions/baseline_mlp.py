"""Baseline: a small MLP."""
from nupyml.nn import MLPClassifier


def solve(X_train, y_train, X_test):
    return MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400,
                         random_state=0).fit(X_train, y_train).predict(X_test)
