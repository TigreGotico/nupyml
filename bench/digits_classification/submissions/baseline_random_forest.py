"""Baseline: random forest."""
from nupyml.ensemble import RandomForestClassifier


def solve(X_train, y_train, X_test):
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    return model.predict(X_test)
