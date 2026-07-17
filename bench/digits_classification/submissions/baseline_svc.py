"""Baseline: support vector classifier with an RBF kernel."""
from nupyml.svm import SVC


def solve(X_train, y_train, X_test):
    model = SVC(kernel="rbf", C=10.0, gamma=0.05, random_state=0)
    model.fit(X_train, y_train)
    return model.predict(X_test)
