"""Baseline: ridge regression -- a strong linear baseline on this dataset."""
from nupyml.linear_model import Ridge


def solve(X_train, y_train, X_test):
    return Ridge(alpha=1.0).fit(X_train, y_train).predict(X_test)
