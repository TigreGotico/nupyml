"""Baseline: random forest -- trees cope with imbalance better than a plain
linear model, without explicit resampling."""
from nupyml.ensemble import RandomForestClassifier


def solve(X_train, y_train, X_test):
    return RandomForestClassifier(n_estimators=200, random_state=0).fit(
        X_train, y_train).predict(X_test)
