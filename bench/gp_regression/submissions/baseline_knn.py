"""Baseline: k-nearest-neighbours regression -- a simple non-parametric reference."""
from nupyml.neighbors import KNeighborsRegressor


def solve(X_train, y_train, X_test):
    return KNeighborsRegressor(n_neighbors=7).fit(X_train, y_train).predict(X_test)
