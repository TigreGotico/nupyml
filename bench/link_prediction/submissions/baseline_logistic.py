"""Baseline: logistic regression on the link-prediction features (returns probs)."""
from nupyml.linear_model import LogisticRegression
from nupyml.preprocessing import StandardScaler


def solve(X_train, y_train, X_test):
    sc = StandardScaler().fit(X_train)
    clf = LogisticRegression().fit(sc.transform(X_train), y_train)
    return clf.predict_proba(sc.transform(X_test))[:, 1]
