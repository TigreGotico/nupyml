"""Baseline: hand-crafted permutation-invariant statistics + random forest."""
import numpy as np

from nupyml.ensemble import RandomForestClassifier


def _features(clouds):
    r = np.linalg.norm(clouds, axis=2)                     # per-point radius
    return np.column_stack([
        r.mean(axis=1), r.std(axis=1), r.max(axis=1), r.min(axis=1),
        clouds.std(axis=1).mean(axis=1),                   # spread
        np.abs(clouds).max(axis=1).mean(axis=1),           # bounding extent
    ])


def solve(X_train, y_train, X_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(_features(X_train), y_train)
    return rf.predict(_features(X_test))
