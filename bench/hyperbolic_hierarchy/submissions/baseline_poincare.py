"""Baseline: Poincare-ball embedding, then hyperbolic distance."""
import numpy as np

from nupyml.embed import PoincareEmbedding


def solve(X_train, y_train, X_test):
    edges = [(int(u), int(v)) for (u, v), d in zip(X_train, y_train) if d == 1]
    pe = PoincareEmbedding(dim=2, epochs=200, lr=0.1, random_state=0).fit(edges)
    return np.array([pe.distance(int(u), int(v)) for u, v in X_test])
