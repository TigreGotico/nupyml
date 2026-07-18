"""Baseline: a Gaussian mixture over the pixel features."""
from nupyml.mixture import GaussianMixture


def solve(X):
    return GaussianMixture(n_components=4, random_state=0).fit(X).predict(X)
