"""Raw flattened pixels + random forest -- the no-feature-engineering floor."""
import numpy as np

from nupyml.ensemble import RandomForestClassifier


def solve(X_train, y_train, X_test):
    Xtr = X_train.reshape(len(X_train), -1)
    Xte = X_test.reshape(len(X_test), -1)
    return RandomForestClassifier(n_estimators=150, random_state=0).fit(
        Xtr, y_train).predict(Xte)
