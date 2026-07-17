"""Classifier chain: each label's model also sees the earlier labels.

A chain threads the binary classifiers together -- label k's features are the
inputs PLUS the predictions for labels 1..k-1 -- so it can exploit the
correlation binary relevance throws away.
"""
from nupyml.linear_model import LogisticRegression
from nupyml.multilabel import ClassifierChain


def solve(X_train, y_train, X_test):
    clf = ClassifierChain(estimator=LogisticRegression(max_iter=500),
                          random_state=0)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)
