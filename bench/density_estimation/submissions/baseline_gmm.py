"""A Gaussian mixture -- the right model for multi-modal Gaussian data."""
from nupyml.mixture import GaussianMixture


def solve(X_train, X_test):
    gm = GaussianMixture(n_components=3, random_state=0).fit(X_train)
    return gm.score_samples(X_test)
