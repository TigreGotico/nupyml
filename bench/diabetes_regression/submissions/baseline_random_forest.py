"""Baseline: random forest regressor."""
from nupyml.ensemble import RandomForestRegressor


def solve(X_train, y_train, X_test):
    return RandomForestRegressor(n_estimators=200, random_state=0).fit(
        X_train, y_train).predict(X_test)
