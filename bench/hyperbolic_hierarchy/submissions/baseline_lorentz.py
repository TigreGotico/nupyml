"""Baseline: Lorentz (hyperboloid) embedding, then hyperbolic distance."""
import numpy as np

from nupyml.embed import LorentzEmbedding


def solve(X_train, y_train, X_test):
    edges = [(int(u), int(v)) for (u, v), d in zip(X_train, y_train) if d == 1]
    le = LorentzEmbedding(dim=3, epochs=500, lr=1.0, n_negative=10,
                          random_state=0).fit(edges)
    return np.array([le.distance(int(u), int(v)) for u, v in X_test])
