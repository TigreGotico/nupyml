"""Density estimation: fit p(x) and score held-out points by log-likelihood.

An unsupervised task: fit a density on ``X_train`` and return the log-density of
each ``X_test`` point. A model that captures the multi-modal structure assigns
higher likelihood to held-out data than one that assumes a single blob. Scored by
the mean held-out log-likelihood (higher is better).

Entries must return TRUE (normalised) log-densities -- the baselines all do; the
score is only meaningful for proper density models.
"""
import numpy as np

KIND = "density"
GOAL = ("Fit a density and score held-out points by mean log-likelihood "
        "(higher is better).")
METRIC = "mean_log_likelihood"
HIGHER_IS_BETTER = True
MIN_SCORE = -9.0          # a proper density must beat this held-out log-likelihood


def load():
    rng = np.random.RandomState(0)
    # a three-component Gaussian mixture in 3-D
    centers = np.array([[0, 0, 0], [5, 5, 0], [0, 5, 5]], float)
    n = 900
    comp = rng.randint(0, 3, n)
    X = centers[comp] + rng.randn(n, 3)
    perm = rng.permutation(n)
    X = X[perm]
    n_tr = 600
    return X[:n_tr], X[n_tr:]


def metric(log_probs):
    return float(np.mean(log_probs))
