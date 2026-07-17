"""Baseline: a Gaussian mixture fitted by EM."""
from nupyml.mixture import GaussianMixture


def solve(X):
    return GaussianMixture(n_components=5, random_state=0).fit(X).predict(X)
