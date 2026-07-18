"""Baseline: a 2-component Gaussian mixture over the features."""
from nupyml.mixture import GaussianMixture
from nupyml.preprocessing import StandardScaler


def solve(X):
    Xs = StandardScaler().fit_transform(X)
    return GaussianMixture(n_components=2, random_state=0).fit(Xs).predict(Xs)
