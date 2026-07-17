"""Treat the series as a flat feature vector: StandardScaler + RBF SVC.

The naive tabular approach -- ignore the time ordering entirely. It works when
the pattern sits at a fixed location, and is a useful floor: a real TSC method
should beat it precisely because the pattern here is phase/location shifted.
"""
from nupyml.pipeline import make_pipeline
from nupyml.preprocessing import StandardScaler
from nupyml.svm import SVC


def solve(X_train, y_train, X_test):
    model = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))
    model.fit(X_train, y_train)
    return model.predict(X_test)
