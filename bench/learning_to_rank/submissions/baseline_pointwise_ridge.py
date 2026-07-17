"""Pointwise baseline: regress the relevance grade directly with ridge.

Ignores the query structure entirely -- just predicts each item's grade and sorts
by it. The floor a real pairwise/listwise ranker should beat.
"""
from nupyml.linear_model import Ridge


def solve(X_train, y_train, groups_train, X_test, groups_test):
    return Ridge(alpha=1.0).fit(X_train, y_train.astype(float)).predict(X_test)
