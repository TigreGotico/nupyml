"""Baseline: RBF-kernel SVC -- the natural fit for a curved boundary."""
from nupyml.svm import SVC


def solve(X_train, y_train, X_test):
    return SVC(kernel="rbf", C=1.0, gamma=1.0).fit(X_train, y_train).predict(X_test)
