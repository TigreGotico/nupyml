"""Baseline: kernel ridge regression (RBF)."""
from nupyml.nonparametric import KernelRidge


def solve(X_train, y_train, X_test):
    return KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1).fit(
        X_train, y_train).predict(X_test)
