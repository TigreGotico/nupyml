"""Binary relevance: one independent classifier per label.

The simplest multi-label method -- train a separate binary classifier for each
label and ignore any correlation between them. The floor the correlation-aware
methods should beat.
"""
from nupyml.linear_model import LogisticRegression
from nupyml.multilabel import BinaryRelevance


def solve(X_train, y_train, X_test):
    clf = BinaryRelevance(estimator=LogisticRegression(max_iter=500))
    clf.fit(X_train, y_train)
    return clf.predict(X_test)
