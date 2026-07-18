"""Baseline: a random forest on adjacency features."""
from nupyml.ensemble import RandomForestClassifier


def solve(X_train, y_train, X_test):
    return RandomForestClassifier(n_estimators=100, random_state=0).fit(
        X_train, y_train).predict(X_test)
