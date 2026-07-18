"""Isolation forest as an OOD scorer (shorter isolation path = more OOD)."""
import numpy as np

from nupyml.outlier import IsolationForest


def solve(X):
    return -IsolationForest(n_estimators=100, random_state=0).fit(X).score_samples(X)
