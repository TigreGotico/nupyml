"""Baseline: exact Gaussian process regression."""
from nupyml.gaussian_process import GaussianProcessRegressor


def solve(X_train, y_train, X_test):
    gp = GaussianProcessRegressor(alpha=1e-2, optimize=False).fit(X_train, y_train)
    return gp.predict(X_test)
