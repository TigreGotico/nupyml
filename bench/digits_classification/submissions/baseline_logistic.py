"""Baseline: multinomial logistic regression."""
from nupyml.linear_model import LogisticRegression


def solve(X_train, y_train, X_test):
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    return model.predict(X_test)
