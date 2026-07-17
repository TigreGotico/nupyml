"""ROCKET: thousands of random convolutional kernels + a ridge classifier.

Random (untrained) kernels, each summarised by its max and proportion-of-positive
values, then a linear head. The modern state-of-the-art-for-cheap in time-series
classification -- no kernel learning at all.
"""
from nupyml.tsclass import RocketClassifier


def solve(X_train, y_train, X_test):
    clf = RocketClassifier(n_kernels=500, random_state=0).fit(X_train, y_train)
    return clf.predict(X_test)
