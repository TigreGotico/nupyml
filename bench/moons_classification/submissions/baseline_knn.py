"""Baseline: k-nearest-neighbours -- a local method handles curved boundaries."""
from nupyml.neighbors import KNeighborsClassifier


def solve(X_train, y_train, X_test):
    return KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train).predict(X_test)
