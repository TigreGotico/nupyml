"""Baseline: gradient boosting regressor."""
from nupyml.ensemble import GradientBoostingRegressor


def solve(X_train, y_train, X_test):
    return GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                     learning_rate=0.05, random_state=0).fit(
        X_train, y_train).predict(X_test)
