"""Baseline: logistic regression on the raw adjacency rows."""
from nupyml.linear_model import LogisticRegression
from nupyml.multiclass import OneVsRestClassifier


def solve(X_train, y_train, X_test):
    clf = OneVsRestClassifier(LogisticRegression(C=1.0)).fit(X_train, y_train)
    return clf.predict(X_test)
