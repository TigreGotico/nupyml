"""Baseline: a small multilayer perceptron on the autograd engine."""
from nupyml.nn import MLPClassifier


def solve(X_train, y_train, X_test):
    model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
                          random_state=0)
    model.fit(X_train, y_train)
    return model.predict(X_test)
