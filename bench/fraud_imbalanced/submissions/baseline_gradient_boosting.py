"""Baseline: gradient boosting."""
from nupyml.ensemble import GradientBoostingClassifier


def solve(X_train, y_train, X_test):
    return GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                      learning_rate=0.1, random_state=0).fit(
        X_train, y_train).predict(X_test)
