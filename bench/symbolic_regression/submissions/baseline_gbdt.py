"""Gradient-boosted trees -- a strong nonparametric approximator."""
from nupyml.ensemble import GradientBoostingRegressor


def solve(X_train, y_train, X_test):
    return GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                     random_state=0).fit(X_train, y_train).predict(X_test)
