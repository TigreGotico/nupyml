"""Baseline: the Adamic-Adar score alone (feature 3), no learning."""
def solve(X_train, y_train, X_test):
    return X_test[:, 2]
