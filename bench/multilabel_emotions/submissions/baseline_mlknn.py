"""ML-kNN: a Bayesian nearest-neighbour multi-label classifier.

For each test point it looks at the label counts among its k neighbours and
applies a per-label maximum-a-posteriori rule. A lazy, non-linear alternative to
the chain that captures local label structure.
"""
from nupyml.multilabel import MLkNN


def solve(X_train, y_train, X_test):
    clf = MLkNN(k=12).fit(X_train, y_train)
    return clf.predict(X_test)
