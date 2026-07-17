"""Predict the global mean rating for everything -- the trivial floor.

No personalisation at all. Any real recommender must beat this; it exists to
anchor the scoreboard and confirm the task carries learnable signal.
"""
import numpy as np


def solve(X_train, y_train, X_test):
    return np.full(len(X_test), float(np.mean(y_train)))
