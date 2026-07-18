"""Baseline: k-NN -- nodes with similar neighbourhoods share a community."""
from nupyml.neighbors import KNeighborsClassifier


def solve(X_train, y_train, X_test):
    return KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train).predict(X_test)
