"""Baseline: random forest on the link-prediction features."""
from nupyml.ensemble import RandomForestClassifier


def solve(X_train, y_train, X_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
    return rf.predict_proba(X_test)[:, 1]
